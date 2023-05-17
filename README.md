# Dog-Identification-model
개 이미지의 품종, 특징 등을 분류, 인식하여 아이콘을 만들어주는 pytorch 모델을 FastAPI와 AWS EC2로 서빙합니다.
모델은 NVIDIA GPU SERVER를 통해 학습시켰으며, pretrained model로 resnet50을 사용했습니다.

아래는 제작할 전체적인 프로젝트 파이프라인 구조입니다. 
<img width="881" alt="스크린샷 2023-05-02 오전 12 11 31" src="https://github.com/oy6uns/Dog-Identification-model/assets/45239582/f64e6b1c-8695-403c-bee7-554a592267a7">

### Ver 23/04/29.
POST 요청으로 사진을 MultipartFormData에 담아 보내면 response로 보낸 개 사진의 품종을 받을 수 있습니다.
현재 EC2에 서빙된 FastAPI 코드에는 statusCode가 없는 상태라, 만약 모델을 다시 서빙한다면, "ImageResponseDto.swift"파일의 responseBody에 statusCode를 추가해줘야합니다. 

### Ver 23/05/04.
- responseBody에 statusCode, success Message, breed를 추가했습니다.
- AWS EC2 인스턴스가 계속 혼자 중지되는 문제를 확인하고, 이를 해결했습니다. 
<img width="710" alt="스크린샷 2023-05-16 오후 1 46 26" src="https://github.com/oy6uns/Dog-Identification-model/assets/45239582/a53aa87a-176a-4e87-a35b-2a8ef09c9511">

### Ver 23/05/11.
- 강아지 아이콘(귀, 털 종류, 패턴) 이미지를 받은 뒤 이를 합성해주는 python 모델을 제작했습니다. 
- 아이콘을 합성할 때 이미지의 배경이 다른 이미지를 가리는 것을 막기 위해 각 아이콘 이미지의 배경을 제거해주었습니다. 
<img width="1512" alt="스크린샷 2023-05-16 오후 1 49 59" src="https://github.com/oy6uns/Dog-Identification-model/assets/45239582/a52e5572-cc7d-4fd0-a11d-61f63432db2b">


### Ver 23/05/15.
- 'BASE_URL/icon' 과의 POST 요청을 추가했습니다. 
- AWS S3와의 연결을 추가했습니다. (Amazon IAM, Amazon Cognito 추가)
["ear", "fur", "pattern"] 과 같이 받고자 하는 이미지 이름을 String 배열로 request body에 담아 POST 요청을 보내면, 
AWS S3에서 배열의 원소와 동일한 파일 이름을 가지는 이미지를 탐색합니다. 
이미지는 ver 23/05/11.에서 만든 아이콘 합성 모델을 거쳐 완성된 dog-icon으로써 response body에 담아 보내줍니다. 

<img width="709" alt="스크린샷 2023-05-16 오후 1 44 45" src="https://github.com/oy6uns/Dog-Identification-model/assets/45239582/7636820b-3cab-42db-8eca-d21b03633a6c">
