# Dog-Identification-model
개 이미지의 품종, 특징 등을 분류, 인식하여 아이콘을 만들어주는 모델을 FastAPI와 AWS EC2로 서빙합니다.

### Ver 23/04/29.
POST 요청으로 사진을 MultipartFormData에 담아 보내면 response로 보낸 개 사진의 품종을 받을 수 있습니다.
현재 EC2에 서빙된 FastAPI 코드에는 statusCode가 없는 상태라, 만약 모델을 다시 서빙한다면, "ImageResponseDto.swift"파일의 responseBody에 statusCode를 추가해줘야합니다. 

