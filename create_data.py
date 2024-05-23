import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math

actions = ['hello', 'pretty', 'shy', 'introduce', 'sorry', 'good', 'how much', 'fine', 'thanks', 'please']
seq_length = 30
secs_for_action = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,  # 두 손을 인식하도록 설정
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions): 
        data = []

        ret, img = cap.read()

        img = cv2.flip(img, 1)

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(7000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action: 
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)  
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                #add=np.zeros(32)
                add=np.full((32,), -1)
                left_loc_x = 0
                left_loc_y = 0
                right_loc_x = 0
                right_loc_y = 0
                
                
                for res, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    
                    hand_side = handedness.classification[0].label
                    #print(result.multi_handedness[0].classification[0])
                    
                    palm_center_x = int(res.landmark[mp_hands.HandLandmark.WRIST].x * img.shape[1])
                    palm_center_y = int(res.landmark[mp_hands.HandLandmark.WRIST].y * img.shape[0])

                 
                    
                    
                    #cv2.putText(img, hand_side, (palm_center_x, palm_center_y),
                    #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    joint = np.zeros((21, 4))  # 왼손의 정보 배열
                    
                    for j, lm in enumerate(res.landmark):
                        
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]  # 왼손의 정보에는 마지막에 각도 정보를 저장할 자리를 만듭니다.
                        
                    # 각 랜드마크 좌표들의 x,y,z좌표만 뽑아서 계산(slicing이용)
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # 벡터 내적(정규화)
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                    # v는 2차원 배열

                    # 벡터간의 각도 구하기
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
                    #여기서 angle은 1차원 배열

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    #angle_label = np.append(angle_label, idx) #angle_label에 action의 고유번호도 추가해 준다
                    # 따라서 angle_label은 각도값 여러개와 action 고유 번호를 포함하고 있는 1차원 배열
                    print("angle lable shape!!!",angle_label.shape)
                    
                    angle_label = angle_label.flatten()
                    
                    if(hand_side=="Left"):
                        #alpha=np.zeros(100)
                        #add=np.concatenate([d_left,alpha])
                        add[:15] = angle_label
                        #d=np.concatenate([d,add])
                        print("add_onehand:")
                        print(add.shape)
                        left_loc_x = palm_center_x
                        left_loc_y = palm_center_y
                    elif(hand_side=="Right"):
                        #alpha=np.zeros(100)
                        #add=np.concatenate([d_left,alpha])
                        add[15:-2] = angle_label
                        #d=np.concatenate([d,add])
                        print("add_onehand:")
                        print(add.shape)
                        right_loc_x = palm_center_x
                        right_loc_y = palm_center_y
                    # d는 모든 랜드마크들의 x,y,z,visibility, 우리가 원하는 각도값 이 포함된 1차원 배열 
                        
                            # 각 랜드마크 좌표들의 x,y,z좌표만 뽑아서 계산(slicing이용)
                    distance = math.sqrt((left_loc_x - right_loc_x)**2 + (left_loc_y - right_loc_y)**2)
                    if(left_loc_x==0 or right_loc_x==0):
                        distance = 0
                    add[-2] = distance
                    add[-1] = idx
        
            
  
                data.append(add)
                print(add)
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break

cap.release()
cv2.destroyAllWindows()
