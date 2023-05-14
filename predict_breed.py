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
    s3 = boto3.client('s3')

    def generate_images():
        for text in texts:
            try:
                response = s3.get_object(Bucket='dog-icon-component-bucket', Key=f'{text}.png')
                yield response['Body'].read()
            except s3.exceptions.NoSuchKey:
                raise HTTPException(status_code=404, detail=f"No image found for text: {text}")

    return StreamingResponse(generate_images(), media_type="image/png")



if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port = 8000)
