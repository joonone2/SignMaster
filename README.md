#    LSTM_SignLanguage_Tensorflow




## OpenCV와 MediaPipe를 사용한 각도 계산

이 프로젝트에서는 OpenCV와 MediaPipe를 사용하여 각 손가락 관절 간의 각도를 계산합니다. MediaPipe를 사용하여 손의 랜드마크를 추출한 후, 각 랜드마크 간의 벡터를 계산하고, 벡터 간의 각도를 구하는 방식으로 각도를 측정합니다. 이러한 각도 데이터는 각 손의 상대거리와 함께 모델의 입력 데이터로 사용됩니다.

<br>
<br>


## 프로젝트의 전체적인 흐름
데이터 준비: 손의 랜드마크 데이터와 관절 간의 각도를 준비합니다. 각 데이터 포인트는 왼손과 오른손의 랜드마크 좌표 및 각도 데이터로 구성된 1차원 배열입니다. <br>

모델 학습: LSTM 모델을 사용하여 데이터를 학습합니다. LSTM 레이어는 시퀀스 데이터를 처리하고, Dense 레이어는 학습된 패턴을 기반으로 손의 제스처를 분류합니다.<br>

제스처 예측: 학습된 모델을 사용하여 새로운 손의 랜드마크 데이터와 각도 데이터에 대해 제스처를 예측합니다.<br>


## Dataset

  - Dataset은 10개의 class, class 당 2개의 video, video당 약 900 frames (30초, 1초당 30frame)
  
    0. hello<br>
    1. pretty<br>
    2. shy<br>
    3. introduce<br>
    4. sorry<br>
    5. good<br>
    6. how much<br>
    7. fine<br>
    8. thanks<br>
    9. please<br>



    <hr>
### 이 프로젝트의 데이터셋은 손의 랜드마크간의 각도와 각 손의 상대거리를 담고 있습니다. 
### 각 데이터 포인트는 관절간의 각도를 포함하는 31개 요소의 1차원 배열로 표현됩니다. 데이터셋은 다음과 같은 방식으로 구성됩니다:


  왼손: 배열의 처음 15개의 인덱스에 왼손의 손가락간의 각도들이 할당됩니다.<br>

  오른손: 배열의 16번째 인덱스부터 15개의 인덱스에 오른손의 손가락간의 각도들이 할당됩니다.
<br>
  왼손과 오른손사이의 거리 : 마지막 인덱스에 입력됩니다.
<br>


  
  만약 왼손만 감지되었다면 오른손의 인덱스는 0으로 채워집니다.
  
  ![image](https://github.com/joonone2/SignMaster/assets/129241680/7c9e7534-1753-407f-87f1-df19e78319fe)

  
  만약 오른손만 감지되었다면 왼손의 인덱스는 0으로 채워집니다.
  
  ![image](https://github.com/joonone2/SignMaster/assets/129241680/105e5a4b-33c7-41d8-bf7c-22e2a9a8723d)

  
  두 손이 모두 감지되었다면 양쪽 손의 데이터가 모두 배열에 포함됩니다.
  
  ![image](https://github.com/joonone2/SignMaster/assets/129241680/392668ec-66db-462b-98c1-67e54e90de8f)



  
  <br>

  Dataset은 임의로 train 7099개 (90%), test 789개 (10%) 로 나눴습니다.
  
  31개 요소의 1차원 배열을 30 frame씩 앞뒤로 29 frame씩 overlap 시킨 것을 한 clip으로 하고 한 video 를 여러 clip으로 나누어 clip 을 하나씩 model에 넣었습니다.


  
  
- # Model
  
  손의 랜드마크 데이터를 학습하기 위해 LSTM(Long Short-Term Memory) 모델을 사용합니다.
  

  ### LSTM (Long Short-Term Memory) 레이어
  설명: LSTM은 순환 신경망(RNN)의 한 종류로, 시계열 데이터나 순차 데이터를 처리하는 데 효과적입니다.
  사용 이유:
  기억력: LSTM은 긴 시퀀스 데이터에서 중요한 정보를 잊지 않고 기억할 수 있습니다.
  손 랜드마크 시퀀스 처리: 손의 랜드마크 데이터는 시간에 따라 연속적으로 변하기 때문에, 시퀀스 데이터로 취급할 수 있습니다. LSTM은 이러한 시퀀스 데이터의 패턴을 학습하는 데 적합합니다.
  
  ### Dense 레이어
  사용 이유:
  분류: LSTM 레이어의 출력은 Dense 레이어를 통해 각 클래스에 대한 확률로 변환됩니다. 이 프로젝트에서는 손의 움직임이나 제스처를 분류하기 위해 사용됩니다.
  Softmax 활성화 함수: Dense 레이어의 활성화 함수로 softmax를 사용하여, 각 클래스에 대한 확률을 계산합니다. 이를 통해 모델이 각 입력 시퀀스에 대해 가장 가능성이 높은 클래스를 예측할 수 있습니다.
  


  - LSTM과 FC layer를 거쳐 지나온 feature에 대한 Activation function으로 Softmax와 Logsoftmax를 비교하여 실험해 성능이 더 잘나온 softmax를 사용하였습니다.
  
  - Loss function은 categorical_crossentropy 와 MSELoss를 비교하여 실험한 결과 더 성능이 잘 나온 categorical_crossentropy를 사용했습니다.

  
  
- # Performance
  
  1. 각각의 동작당 30초의 데이터를 학습 시켰을때의 그래프입니다.
  
    ![image](https://github.com/joonone2/SignMaster/assets/129241680/96f3b244-62a1-4823-80d2-c7700faad023)
_가로축 = epoch , 왼쪽 세로축 = loss, 오른쪽 세로축 = accuracy_

<br><br>


  2. 각각의 동작당 60초의 데이터를 학습 시켰을때의 그래프입니다.
    ![image](https://github.com/joonone2/SignMaster/assets/129241680/a32714c0-216d-4f43-9756-032899d705e5)
_가로축 = epoch , 왼쪽 세로축 = loss, 오른쪽 세로축 = accuracy_




  ### 2번이 1번보다 성능이 좋았기 때문에 학습데이터는 60초의 dataset으로 구성하였습니다. 
  
# actions
  
|                                                        how much                                                        |                                                         inroduce                                                         |
| :---------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: |
| <img src='https://github.com/joonone2/SignMaster/assets/129241680/a98ae2b9-1108-43a2-bd42-09eb8de39b50'> | <img src='https://github.com/joonone2/SignMaster/assets/129241680/2f899986-7237-4a87-9f94-9545b57930d2'> |
|                                                     <b>please</b>                                                      |                                                <b>sorry</b>                                                |
| <img src='https://github.com/joonone2/SignMaster/assets/129241680/ad239822-0ae5-48e4-bfa0-8e29f25baf85'> | <img src='https://github.com/joonone2/SignMaster/assets/129241680/4f779a5e-8b69-4581-a73a-5666cce89cf3'> |
|                                                     <b>thanks</b>                                                      |                                                <b>fine</b>                                                |
| <img src='https://github.com/joonone2/SignMaster/assets/129241680/be9e4471-5709-4174-a71f-fd3edefcd36f'> | <img src='https://github.com/joonone2/SignMaster/assets/129241680/15a0716d-2406-47bf-a573-0ba4311c1d50'> |
|                                                     <b>good</b>                                                      |                                                <b>pretty</b>                                                |
| <img src='https://github.com/joonone2/SignMaster/assets/129241680/4f313df2-2d84-4b1a-a18b-a9148d4babb3'> | <img src='https://github.com/joonone2/SignMaster/assets/129241680/1943e045-85d7-452f-8ae4-6990cb1fc63f'> |
|                                                     <b>hello</b>                                                      |                                                <b>shy</b>                                                |
| <img src='https://github.com/joonone2/SignMaster/assets/129241680/30051e5e-13ab-4a3a-b0f2-bc318137f135'> | <img src='https://github.com/joonone2/SignMaster/assets/129241680/5f01063c-5559-4fb4-b886-40f5716bbd2f'> |





## 🛠️ Skills

<img width="800px" src='https://github.com/joonone2/SignMaster/assets/129241680/5d9b8d2a-daf9-443f-81af-d313f6d1ca61'  alt="Skills"/>

    
- # Code

  - version
    
    python 3.11.7<br>
    tensorflow version 2.15.0<br>
    keras version 2.15.0<br>
    mediapipe version 0.10.11<br>
  
  - Practice
    
    [train_model.ipynb](https://github.com/joonone2/SignMaster/blob/main/train_model.ipynb) <- train과 test를 하는 code<br>
    [models.my_model.tflite](https://github.com/joonone2/SignMaster/blob/main/models/my_model.tflite) <- model<br>
    [check_model.py](https://github.com/joonone2/SignMaster/blob/main/check_model.py) <- 생성된 모델을 테스트하는 code<br>
    [create_data.py](https://github.com/joonone2/SignMaster/blob/main/create_data.py)  <- dataset을 만드는 code<br>



    
    
  
