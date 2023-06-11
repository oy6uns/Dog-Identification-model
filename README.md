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

### Ver 23/05/18.
- ear, fur, pattern detection 모델을 추가했습니다. (pretrained model = resnet50)
- 사진을 request body에 담아 POST 요청을 보내면, 각각 ear, fur, dot detection 모델을 돌린 뒤에 어떤 종류의 icon에 해당하는지를 detection 합니다. 
- detection한 각 ear, fur, pattern의 종류를 바탕으로 S3 storage에서 그에 맞는 이미지를 찾아서 합성해서 resoponse body에 담아 보내줍니다. 
  - ear_type = ['down', 'up']
  - fur_type = ['fur', 'no_fur']
  - pattern_type = ['no', 'ear_dot', 'many', 'nose', 'pattern3']
<img width="736" alt="스크린샷 2023-05-18 오후 2 54 52" src="https://github.com/oy6uns/Dog-Identification-model/assets/45239582/485eea4f-f5f2-410d-a0a6-44959ce6818e">

### Ver 23/05/30.
- 강아지 얼굴의 color를 추출해낸 뒤, 그 색에 맞는 icon을 찾아주는 코드를 추가했습니다. 
- dogHeadDetector.dat 이라는 dlib의 pretrained model을 사용했습니다. 
  - dlib을 ubuntu EC2 클라우드 컴퓨터에 설치하는데 많은 애를 먹었습니다. 
  - cmake를 먼저 install 해준 뒤에 dlib을 설치하려고 시도했지만, 85%에서 계속 다운로드가 멈췄습니다.(maybe RAM 부족 문제, swap Memory도 사용했지만 해결되지 않았습니다ㅠㅠ)
  - 결국, EC2 인스턴스 유형을 t2 micro -> t3a medium 으로 변경해줌으로써 해결했습니다. 
- 생성된 강아지 아이콘을 아래와 같은 url로 S3에 업로드합니다. 
- texts = [colorArray[0] + "-" + ear_type[ear_preds.item()], colorArray[0] + "-" + fur_type[fur_preds.item()], colorArray[1] + "-" + pattern_type[pattern_preds.item()], "dog-face"]
<img width="677" alt="스크린샷 2023-06-11 오후 6 33 49" src="https://github.com/oy6uns/Dog-Identification-model/assets/45239582/dbf499c2-c01d-4d02-8bd7-70cafcc3e754">

### Ver 23/06/11.
> 아래와 같은 에러메시지를 발견했습니다. <br>
xhr.js:247 Mixed Content: The page at 'https://dog-mbti.pages.dev/' was loaded over HTTPS, but requested an insecure XMLHttpRequest endpoint 'http://3.23.60.50:8000/final'. This request has been blocked; the content must be served over HTTPS.
Web은 보안 문제로 인해 HTTP 엔트포인트와 통신하는데 있어서 위와 같은 오류가 발생했습니다. 따라서, 이를 조치해주기 위해 API URL을 HTTPS로 변경해주었습니다. 
- SSL 인증서를 발급받기 위해 nginx를 설정해주었습니다. 
- ip 주소가 아닌 도메인 주소를 발급받고, ip 주소와 연동시켜주었습니다. (도메인 주소 발급 : https://www.duckdns.org/domains)
<img width="805" alt="스크린샷 2023-06-11 오후 6 42 39" src="https://github.com/oy6uns/Dog-Identification-model/assets/45239582/54c34a44-61cc-49a1-b2e2-1a662eee1b4f">
