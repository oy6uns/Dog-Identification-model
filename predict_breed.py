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

# for dog color model
import dlib
import cv2
import numpy as np
from imutils import face_utils
from sklearn.cluster import KMeans

# for dog detection model


# init app
app = FastAPI()

'''Web CORS 관련 문제 해결 코드'''
origins = [
    "http://localhost",
    "http://localhost:8000",
    "https://dog-mbti.pages.dev"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

'''강아지 종 분류 관련 코드'''
# 예전 모델로 일단 올려놓음. 
breed_model = torch.load('selected_breed_epoch100.pth', map_location=torch.device('cpu'))

'''강아지의 색상 추출해주는 함수 코드'''

'''강아지 face detection'''
ear_model = torch.load('ear_resnet50.pth', map_location=torch.device('cpu'))
fur_model = torch.load('fur_resnet50.pth', map_location=torch.device('cpu'))
dot_model = torch.load('dot_revised_data_v1.pth', map_location=torch.device('cpu'))

# detection model로 부터 얻은 index 값으로부터 동일한 파일명을 S3로부터 불러온다. 
# 그러기 위해 각 detection feature의 종류를 배열에 저장해놓는다. 
ear_type = ['down', 'up']
fur_type = ['fur', 'no_fur']
pattern_type = ['no', 'ear_dot', 'many', 'nose', 'pattern3']

# 새로운 모델의 개 품종
selected_classes = ['beagle','labrador_retriever','golden_retriever','french_bulldog','german_shepherd',
                    'toy_poodle','rottweiler','yorkshire_terrier','siberian_husky','shih-tzu','doberman',
                   'maltese_dog','border_collie','chihuahua','pomeranian','italian_greyhound','boxer',
                    'standard_schnauzer']

# need to get the ids from the sample_submission csv so we can match it up 
labels = dict()
for index, value in enumerate(selected_classes):
    labels[value] = index
print(labels)

def swap_dict(d):
    return {v: k for k, v in d.items()}

# ear, fur, pattern detect를 위한 image-to-tensor 코드
async def convert_image_to_tensor_detect(upload_file, model):
    image_bytes = await upload_file.read()
    dataBytesIO = io.BytesIO(image_bytes)

    img = Image.open(dataBytesIO).convert('RGB')
    transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=300, scale=(0.8, 1.0)),
                    transforms.CenterCrop(size=300),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
    
    b= transform(img)
    tensor_image = torch.unsqueeze(b, 0)

    input = Variable(tensor_image)
    output = model(input)
    return output

# breed identification을 위한 image-to-tensor 코드
async def convert_image_to_tensor_breed(upload_file, model):
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
    
    return output
    

detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')
predictor = dlib.shape_predictor('landmarkDetector.dat')

@app.post("/color", status_code=201)
async def predict_color(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_np = np.array(img)

    def defineColor(img_np):
        detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')

        # yuv_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2YUV)
        # yuv_img[:,:,0] = cv2.equalizeHist(yuv_img[:,:,0])
        # img_output = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
        dets = detector(img_np, upsample_num_times=1)

        colors = []
        for i, d in enumerate(dets):
            # 얼굴 영역 추출
            x1, y1 = d.rect.left(), d.rect.top()
            x2, y2 = d.rect.right(), d.rect.bottom()

            face_img = img_np[y1:y2, x1:x2]

            # RGB로 변환
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # 이미지를 2차원 배열로 변경
            pixels = face_img.reshape(-1, 3)

            # k-means 클러스터링을 사용하여 가장 일반적인 색상 구하기
            kmeans = KMeans(n_clusters=5)
            kmeans.fit(pixels)

            # 중심점의 RGB 값
            RGB_colors = kmeans.cluster_centers_

            # RGB 값을 정수로 변환
            RGB_colors = RGB_colors.round(0).astype(int)

            # RGB 값을 50 단위로 정규화
            for i in range(len(RGB_colors)):
                RGB_colors[i] = ((RGB_colors[i] + 25) // 50) * 50

            # 각 dominant color의 비율 계산
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            sorted_colors = sorted(zip(RGB_colors, counts), key=lambda x: x[1], reverse=True)
            
            # , 단위로 출력
            for color, count in sorted_colors:
                colors.append(','.join(map(str, color)))

        return colors[:2]

    return defineColor(img_np)

@app.post("/predict", status_code=201)
async def predict(file: UploadFile = File(...)):
    # file_ext = file.filename.split(".").pop() # jpeg, png등 확장자 무시
    # file_name = "test picture"
    # file_path = f"{file_name}.{file_ext}"
    # run_model("test picture.jpg")
    # with open(file_path, "wb") as f:
    #     content = await file.read()
    #     f.write(content)
    tensorImage = await convert_image_to_tensor_breed(file, breed_model)
    _, preds = torch.max(tensorImage.data, 1)
    breed_num = preds.item()
    breed_labels = swap_dict(labels)
    breed = breed_labels[breed_num]
    print("breed: ", breed)
    return {"statusCode": 201, "success": True, "message":"File uploaded successfully", "breed": breed}

@app.post("/face", status_code=201)
async def detectFace(file: UploadFile = File(...)):
    # file_ext = file.filename.split(".").pop() # jpeg, png등 확장자 무시
    # file_name = "test picture"
    # file_path = f"{file_name}.{file_ext}"
    # run_model("test picture.jpg")
    # with open(file_path, "wb") as f:
    #     content = await file.read()
    #     f.write(content)
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=300, scale=(0.8, 1.0)),
                    transforms.CenterCrop(size=300),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
    
    b= transform(img)
    tensor_image = torch.unsqueeze(b, 0)

    input = Variable(tensor_image)
    ear_output = ear_model(input)
    fur_output = fur_model(input)
    pattern_output = dot_model(input)

    _, ear_preds = torch.max(ear_output.data, 1)
    _, fur_preds = torch.max(fur_output.data, 1)
    _, pattern_preds = torch.max(pattern_output.data, 1)

    return ear_preds.item(), fur_preds.item(), pattern_preds.item()

@app.post("/icon", status_code=201)
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
    pattern_image = pattern_image.resize((220, 180))
    face_image = images[3]

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
    
    def make_face_position(face_image, fur_image):
        x1 = int((fur_image.size[0] - face_image.size[0]) / 2)
        x2 = x1 + face_image.size[0]
        y1 = 220
        y2 = y1 + face_image.size[1]

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
    make_color_transparent(face_image, (255, 255, 255))

    area_ear = make_ear_position(ear_image, fur_image)
    area_pattern = make_pattern_position(pattern_image, fur_image)
    area_face = make_face_position(face_image, fur_image)

    # 귀 이미지와 패턴 이미지를 알맞은 위치에 삽입해주는 함수
    fur_image.paste(ear_image, area_ear, mask=ear_image)
    fur_image.paste(pattern_image, area_pattern, mask=pattern_image)
    fur_image.paste(face_image, area_face, mask=face_image)

    image_bytes = io.BytesIO()
    fur_image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    return StreamingResponse(image_bytes, media_type="image/png")

# final version respose with Image
@app.post("/finalImg", status_code=201)
async def makeIcon(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=300, scale=(0.8, 1.0)),
                    transforms.CenterCrop(size=300),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
    
    b= transform(img)
    tensor_image = torch.unsqueeze(b, 0)

    input = Variable(tensor_image)
    ear_output = ear_model(input)
    fur_output = fur_model(input)
    pattern_output = dot_model(input)

    _, ear_preds = torch.max(ear_output.data, 1)
    _, fur_preds = torch.max(fur_output.data, 1)
    _, pattern_preds = torch.max(pattern_output.data, 1)

    #AWS S3 스토리지에 접근
    s3 = boto3.client('s3')
    print(pattern_preds.item())
    texts = ["250,250,250-" + ear_type[ear_preds.item()], "250,250,250-" + fur_type[fur_preds.item()], "200,200,200-" + pattern_type[pattern_preds.item()], "dog-face"]

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

    # 배경에 깔 이미지를 하나 생성해준다. 
    background_image = Image.new('RGBA', (600, 600), (0, 0, 0, 0))

    # images 배열의 각 이미지를 순서대로 새로운 배열에 저장해준다. 
    ear_image = images[0]
    fur_image = images[1]
    if fur_preds.item() == 1:
        fur_image = fur_image.resize((300, 300))
    pattern_image = images[2]
    face_image = images[3]

    # ear_type = ['down', 'up']
    # fur_type = ['fur', 'no_fur']
    # pattern_type = ['no', 'ear_dot', 'many', 'nose', 'pattern3']

    # ear(귀)의 위치를 지정해주기 위한 함수
    def make_ear_position(ear_preds, ear_image, background_image):
        ear_x1 = int((background_image.size[0] - ear_image.size[0]) / 2)
        ear_x2 = ear_image.size[0] + ear_x1
        if(ear_preds == 0):
            ear_y1 = 160
        else:
            ear_y1 = 80
        ear_y2 = ear_y1 + ear_image.size[1]

        area = (ear_x1, ear_y1, ear_x2, ear_y2)
        return area

    # pattern(무늬)의 위치를 지정해주기 위한 함수
    def make_pattern_position(pattern_preds, pattern_image, background_image):
        print(pattern_preds)
        if pattern_preds == 1:
            pattern_image = pattern_image.resize((250, 150))
            x1 = int((background_image.size[0] - pattern_image.size[0]) / 2)
            print(x1)
            x2 = pattern_image.size[0] + x1
            print(x2)
            y1 = int((background_image.size[1] - pattern_image.size[1]) / 2) - 30
            print(y1)
            y2 = pattern_image.size[1] + y1
            print(y2)
        elif pattern_preds == 2:
            pattern_image = pattern_image.resize((250, 220))
            x1 = int((background_image.size[0] - pattern_image.size[0]) / 2)
            x2 = pattern_image.size[0] + x1
            y1 = int((background_image.size[1] - pattern_image.size[1]) / 2)
            y2 = pattern_image.size[1] + y1
        elif pattern_preds == 3:
            x1 = int((background_image.size[0] - pattern_image.size[0]) / 2)
            x2 = pattern_image.size[0] + x1
            y1 = int((background_image.size[1] - pattern_image.size[1]) / 2) + 55
            y2 = pattern_image.size[1] + y1
        elif pattern_preds == 4:
            pattern_image = pattern_image.resize((315, 195))
            x1 = int((background_image.size[0] - pattern_image.size[0]) / 2)
            x2 = pattern_image.size[0] + x1
            y1 = int((background_image.size[1] - pattern_image.size[1]) / 2) - 60
            y2 = pattern_image.size[1] + y1
        else:
            x1 = int((background_image.size[0] - pattern_image.size[0]) / 2)
            x2 = pattern_image.size[0] + x1
            y1 = int((background_image.size[1] - pattern_image.size[1]) / 2)
            y2 = pattern_image.size[1] + y1

        area = (x1, y1, x2, y2)
        return area
    
    def make_fur_position(fur_image, background_image):
    
        x1 = int((background_image.size[0] - fur_image.size[0]) / 2)
        x2 = fur_image.size[0] + x1
        y1 = int((background_image.size[1] - fur_image.size[1]) / 2)
        y2 = fur_image.size[1] + y1

        area = (x1, y1, x2, y2)
        return area
    
    def make_face_position(face_image, background_image):
        x1 = int((background_image.size[0] - face_image.size[0]) / 2)
        x2 = x1 + face_image.size[0]
        y1 = 270
        y2 = y1 + face_image.size[1]

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
    make_color_transparent(fur_image, (255, 255, 255))
    make_color_transparent(pattern_image, (255, 255, 255))
    make_color_transparent(face_image, (255, 255, 255))

    area_ear = make_ear_position(ear_preds, ear_image, background_image)
    area_pattern = make_pattern_position(pattern_preds, pattern_image, background_image)
    print(area_pattern)
    area_fur = make_fur_position(fur_image, background_image)
    area_face = make_face_position(face_image, background_image)

    # 배경에 이미지를 겹쳐서 붙이기
    background_image.paste(fur_image, area_fur, mask=fur_image)
    if pattern_preds.item() != 0:
        if pattern_preds == 1:
            pattern_image = pattern_image.resize((250, 150))
        elif pattern_preds == 2:
            pattern_image = pattern_image.resize((250, 220))
        elif pattern_preds == 4:
            pattern_image = pattern_image.resize((315, 195))
        background_image.paste(pattern_image, area_pattern, mask=pattern_image)

    background_image.paste(ear_image, area_ear, mask=ear_image)
    background_image.paste(face_image, area_face, mask=face_image)

    image_bytes = io.BytesIO()
    background_image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    return StreamingResponse(image_bytes, media_type="image/png")

# final version respose with URL
@app.post("/final", status_code=201)
async def makeIcon_URL(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=300, scale=(0.8, 1.0)),
                    transforms.CenterCrop(size=300),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
    
    b= transform(img)
    tensor_image = torch.unsqueeze(b, 0)

    input = Variable(tensor_image)
    ear_output = ear_model(input)
    fur_output = fur_model(input)
    pattern_output = dot_model(input)

    _, ear_preds = torch.max(ear_output.data, 1)
    _, fur_preds = torch.max(fur_output.data, 1)
    _, pattern_preds = torch.max(pattern_output.data, 1)

    #AWS S3 스토리지에 접근
    s3 = boto3.client('s3')
    print(pattern_preds.item())
    
    detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')
    img_np = np.array(img)

    def getNearest50(num):
        # 주어진 숫자를 가장 가까운 50의 배수로 반올림
        return round((num + 25) // 50) * 50

    def defineColor(img_np):

        dets = detector(img_np, upsample_num_times=1)

        for i, d in enumerate(dets):
            # 얼굴 영역 추출
            x1, y1 = d.rect.left(), d.rect.top()
            x2, y2 = d.rect.right(), d.rect.bottom()

            face_img = img_np[y1+15:y2-15, x1+15:x2-15]

            # RGB로 변환
            # face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # 이미지를 2차원 배열로 변경
            pixels = face_img.reshape(-1, 3)

            # k-means 클러스터링을 사용하여 가장 일반적인 색상 구하기
            kmeans = KMeans(n_clusters=5)
            kmeans.fit(pixels)

            numLabels = np.arange(0, len(np.unique(kmeans.labels_)) + 1)
            (hist, _) = np.histogram(kmeans.labels_, bins=numLabels)
            hist = hist.astype("float")
            hist /= hist.sum()

            # 각 색상 블록의 크기와 해당 색상을 사전에 저장
            dictionary = {}
            for (percent, color) in zip(hist, kmeans.cluster_centers_):
                block = percent * 300
                dictionary[block] = color.astype("uint8").tolist()

            # 사전을 빈도수에 따라 정렬하고 가장 빈번한 두 가지 색상을 선택
            sorted_dict = sorted(dictionary.items(), reverse=True)
            top_two_colors = [','.join(map(str, [getNearest50(value) for value in color])) for block, color in sorted_dict[:2]]

            return top_two_colors

    colorArray = defineColor(img_np) # test, 지우면 됨
    print(colorArray)
    texts = [colorArray[0] + "-" + ear_type[ear_preds.item()], colorArray[0] + "-" + fur_type[fur_preds.item()], colorArray[1] + "-" + pattern_type[pattern_preds.item()], "dog-face"]

    face_bg_arr = ['0,0,0-fur', '0,0,0-no_fur']

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
                image = Image.new('RGBA', (600, 600), (0, 0, 0, 0))
                images.append(image)
                # raise HTTPException(status_code=404, detail=f"No image found for text: {text}")
        return images
    
        # image를 S3버킷으로부터 불러오는 함수
    def generate_bg_images():
        images = []
        # request body의 배열의 원소와 파일명이 동일한 이미지를 S3버킷으로부터 불러온다. 
        for text in face_bg_arr:
            try:
                response = s3.get_object(Bucket='dog-icon-component-bucket', Key=f'{text}.png')
                image_data = response['Body'].read()
                # byte형태를 image형으로 변환해준다. 
                image = Image.open(io.BytesIO(image_data))
                images.append(image)
            # 찾고자 하는 이미지가 S3버킷에 존재하지 않을 때, 에러메시지 출력
            except s3.exceptions.NoSuchKey:
                image = Image.new('RGBA', (600, 600), (0, 0, 0, 0))
                images.append(image)
                # raise HTTPException(status_code=404, detail=f"No image found for text: {text}")
        return images
    
    images = generate_images()
    face_bgs = generate_bg_images()

    # 실제로 불러온 이미지 개수와 request body에서 전달한 텍스트의 개수가 다를 때 에러메시지 출력
    if len(images) < len(texts):
        raise HTTPException(status_code=500, detail="Some images could not be retrieved.")

    # 배경에 깔 이미지를 하나 생성해준다. 
    background_image = Image.new('RGBA', (600, 600), (0, 0, 0, 0))

    # images 배열의 각 이미지를 순서대로 새로운 배열에 저장해준다. 
    ear_image = images[0]
    fur_image = images[1]
    if fur_preds.item() == 0:
        # face_bgs[0] = face_bgs[0].resize((500, 500))
        face_bgs[0] = face_bgs[0].resize((510, 510))
    if fur_preds.item() == 1:
        fur_image = fur_image.resize((300, 300))
        # face_bgs[1] = face_bgs[1].resize((304, 304))
        face_bgs[1] = face_bgs[1].resize((310, 310))
    pattern_image = images[2]
    face_image = images[3]

    # ear_type = ['down', 'up']
    # fur_type = ['fur', 'no_fur']
    # pattern_type = ['no', 'ear_dot', 'many', 'nose', 'pattern3']

    # ear(귀)의 위치를 지정해주기 위한 함수
    def make_ear_position(ear_preds, ear_image, background_image):
        ear_x1 = int((background_image.size[0] - ear_image.size[0]) / 2)
        ear_x2 = ear_image.size[0] + ear_x1
        if(ear_preds == 0):
            ear_y1 = 160
        else:
            ear_y1 = 100
        ear_y2 = ear_y1 + ear_image.size[1]

        area = (ear_x1, ear_y1, ear_x2, ear_y2)
        return area

    # pattern(무늬)의 위치를 지정해주기 위한 함수
    def make_pattern_position(pattern_preds, pattern_image, background_image):
        print(pattern_preds)
        if pattern_preds == 1:
            # pattern_image = pattern_image.resize((250, 150))
            x1 = int((background_image.size[0] - pattern_image.size[0]) / 2)
            print(x1)
            x2 = pattern_image.size[0] + x1
            print(x2)
            y1 = int((background_image.size[1] - pattern_image.size[1]) / 2) - 60
            print(y1)
            y2 = pattern_image.size[1] + y1
            print(y2)
        elif pattern_preds == 2:
            pattern_image = pattern_image.resize((250, 220))
            x1 = int((background_image.size[0] - pattern_image.size[0]) / 2)
            x2 = pattern_image.size[0] + x1
            y1 = int((background_image.size[1] - pattern_image.size[1]) / 2)
            y2 = pattern_image.size[1] + y1
        elif pattern_preds == 3:
            x1 = int((background_image.size[0] - pattern_image.size[0]) / 2)
            x2 = pattern_image.size[0] + x1
            y1 = int((background_image.size[1] - pattern_image.size[1]) / 2) + 50
            y2 = pattern_image.size[1] + y1
        elif pattern_preds == 4:
            pattern_image = pattern_image.resize((305, 195))
            x1 = int((background_image.size[0] - pattern_image.size[0]) / 2)
            x2 = pattern_image.size[0] + x1
            y1 = int((background_image.size[1] - pattern_image.size[1]) / 2) - 60
            y2 = pattern_image.size[1] + y1
        else:
            x1 = int((background_image.size[0] - pattern_image.size[0]) / 2)
            x2 = pattern_image.size[0] + x1
            y1 = int((background_image.size[1] - pattern_image.size[1]) / 2)
            y2 = pattern_image.size[1] + y1

        area = (x1, y1, x2, y2)
        return area
    
    def make_fur_position(fur_image, background_image):
    
        x1 = int((background_image.size[0] - fur_image.size[0]) / 2)
        x2 = fur_image.size[0] + x1
        y1 = int((background_image.size[1] - fur_image.size[1]) / 2)
        y2 = fur_image.size[1] + y1

        area = (x1, y1, x2, y2)
        return area
    
    def make_face_position(face_image, background_image):
        x1 = int((background_image.size[0] - face_image.size[0]) / 2)
        x2 = x1 + face_image.size[0]
        y1 = 270
        y2 = y1 + face_image.size[1]

        area = (x1, y1, x2, y2)
        return area
    
    def make_face_background_position(face_background_image, background_image):
        x1 = int((background_image.size[0] - face_background_image.size[0]) / 2)
        x2 = x1 + face_background_image.size[0]
        y1 = int((background_image.size[1] - face_background_image.size[1]) / 2)
        y2 = y1 + face_background_image.size[1]

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
    make_color_transparent(fur_image, (255, 255, 255))
    make_color_transparent(pattern_image, (255, 255, 255))
    make_color_transparent(face_image, (255, 255, 255))

    area_ear = make_ear_position(ear_preds, ear_image, background_image)
    area_pattern = make_pattern_position(pattern_preds, pattern_image, background_image)
    print(area_pattern)
    area_fur = make_fur_position(fur_image, background_image)
    area_face = make_face_position(face_image, background_image)
    if fur_preds == 0 :
        # face_bgs[0] = face_bgs[0].resize((500, 500))
        face_bgs[0] = face_bgs[0].resize((510, 510))
        area_face_bg = make_face_background_position(face_bgs[0], background_image)
        face_bg_image = face_bgs[0]
    if fur_preds == 1:
        # face_bgs[1] = face_bgs[1].resize((304, 304))
        face_bgs[1] = face_bgs[1].resize((310, 310))
        area_face_bg = make_face_background_position(face_bgs[1], background_image)
        face_bg_image = face_bgs[1]


    # 배경에 얼굴 배경 먼저 붙이기
    background_image.paste(face_bg_image, area_face_bg, mask=face_bg_image)

    background_image.paste(ear_image, area_ear, mask=ear_image)

    # 얼굴 이미지를 이어서 붙이기 
    background_image.paste(fur_image, area_fur, mask=fur_image)
    if pattern_preds.item() != 0:
        # if pattern_preds == 1:
        #     pattern_image = pattern_image.resize((250, 150))
        if pattern_preds == 2:
            pattern_image = pattern_image.resize((250, 220))
        elif pattern_preds == 4:
            pattern_image = pattern_image.resize((305, 195))
        background_image.paste(pattern_image, area_pattern, mask=pattern_image)

   
    background_image.paste(face_image, area_face, mask=face_image)
    

    image_bytes = io.BytesIO()
    background_image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # S3에 업로드 할 이미지 이름 생성함수
    def join_text_elements(text_list):
        return ''.join(text_list[:3])

    # Specify your bucket name and image key.
    bucket = 'dog-icon-component-bucket'
    key = join_text_elements(texts) + '.jpg'
    folder = "generated"

    # Upload the image to S3
    s3.put_object(Bucket=bucket, Key=f"{folder}/{key}", Body=image_bytes.getvalue(), ContentType='image/jpeg')

    # Generate the URL for the uploaded image
    image_url = f"https://{bucket}.s3.amazonaws.com/{folder}/{key}"

    # Return the URL
    return {"image_url": image_url}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port = 8000)