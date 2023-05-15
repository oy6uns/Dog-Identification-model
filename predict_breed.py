import uvicorn
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import io
import pandas as pd 
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
# for dog icon model
import boto3

# init app
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", status_code=201)
async def predict(file: UploadFile = File(...)):
    # file_ext = file.filename.split(".").pop() # jpeg, png등 확장자 무시
    # file_name = "test picture"
    # file_path = f"{file_name}.{file_ext}"
    # run_model("test picture.jpg")
    # with open(file_path, "wb") as f:
    #     content = await file.read()
    #     f.write(content)
    breed_num = await convert_image_to_tensor(file)
    breed_labels = swap_dict(labels)
    breed = breed_labels[breed_num]
    print("breed: ", breed)
    return {"statusCode": 201, "success": True, "message":"File uploaded successfully", "breed": breed}

model = torch.load('resnet50_fintuned_epoch50_v1.pt', map_location=torch.device('cpu'))

# need to get the ids from the sample_submission csv so we can match it up 
labels = dict()
for index, value in enumerate(pd.read_csv('dog-breed-identification/sample_submission.csv').columns[1:]):
    labels[value] = index
print(labels)

def swap_dict(d):
    return {v: k for k, v in d.items()}

async def convert_image_to_tensor(upload_file):
    image_bytes = await upload_file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.95, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    b= transform(img)
    tensor_image = torch.unsqueeze(b, 0)

    input = Variable(tensor_image)
    output = model(input)
    _, preds = torch.max(output.data, 1)

    return preds.item()

@app.post("/icon")
async def get_images_from_s3(texts: list[str]):
    #AWS S3 스토리지에 접근
    s3 = boto3.client('s3')

    # image를 S3버킷으로부터 불러오는 함수
    def generate_images():
        images = []
        # request body의 배열의 원소와 파일명이 동일한 이미지를 S3버킷으로부터 불러온다. 
        for text in texts:
            try:
                response = s3.get_object(Bucket='dog-icon-component-bucket', Key=f'{text}.png')
                image_data = response['Body'].read()
                # byte형태를 image형으로 변환해준다. 
                image = Image.open(io.BytesIO(image_data))
                images.append(image)
            # 찾고자 하는 이미지가 S3버킷에 존재하지 않을 때, 에러메시지 출력
            except s3.exceptions.NoSuchKey:
                raise HTTPException(status_code=404, detail=f"No image found for text: {text}")
        return images
    
    images = generate_images()

    # 실제로 불러온 이미지 개수와 request body에서 전달한 텍스트의 개수가 다를 때 에러메시지 출력
    if len(images) < len(texts):
        raise HTTPException(status_code=500, detail="Some images could not be retrieved.")

    # images 배열의 각 이미지를 순서대로 새로운 배열에 저장해준다. 
    ear_image = images[0]
    fur_image = images[1]
    pattern_image = images[2]

    # ear(귀)의 위치를 지정해주기 위한 함수
    def make_ear_position(ear_image, fur_image):
        ear_x1 = int((fur_image.size[0] - ear_image.size[0]) / 2)
        ear_x2 = fur_image.size[0] - ear_x1
        ear_y1 = 50
        ear_y2 = ear_y1 + ear_image.size[1]

        area = (ear_x1, ear_y1, ear_x2, ear_y2)
        return area

    # pattern(무늬)의 위치를 지정해주기 위한 함수
    def make_pattern_position(pattern_image, fur_image):
        x1 = int((fur_image.size[0] - pattern_image.size[0]) / 2)
        x2 = pattern_image.size[0] + x1
        y1 = int((fur_image.size[1] - pattern_image.size[1]) / 2)
        y2 = pattern_image.size[1] + y1

        area = (x1, y1, x2, y2)
        return area

    # 배경색을 제거해주기 위한 함수
    def make_color_transparent(image, target_color):
        # 이미지에 알파 채널(투명도) 추가
        image = image.convert("RGBA")

        # 이미지의 픽셀 데이터 가져오기
        data = image.getdata()

        # 새로운 픽셀 데이터 생성
        new_data = []
        for item in data:
            # 대상 색상과 일치하는 경우 알파 값을 0으로 설정하여 투명하게 만듦
            if item[:3] == target_color:
                new_data.append((*target_color, 0))
            else:
               new_data.append(item)

        # 이미지에 새로운 픽셀 데이터 적용
        image.putdata(new_data)

    # 대상 이외의 배경을 제거하는 함수
    make_color_transparent(ear_image, (255, 255, 255))
    make_color_transparent(pattern_image, (255, 255, 255))

    area_ear = make_ear_position(ear_image, fur_image)
    area_pattern = make_pattern_position(pattern_image, fur_image)
    print(area_pattern)

    # 귀 이미지와 패턴 이미지를 알맞은 위치에 삽입해주는 함수
    fur_image.paste(ear_image, area_ear, mask=ear_image)
    fur_image.paste(pattern_image, area_pattern, mask=pattern_image)

    image_bytes = io.BytesIO()
    fur_image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    return StreamingResponse(image_bytes, media_type="image/png")

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port = 8000)
