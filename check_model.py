import cv2
import mediapipe as mp
import numpy as np
import math
import uuid
import os
from tensorflow.keras.models import load_model

actions = ['hello', 'pretty', 'shy', 'introduce', 'sorry', 'good', 'how much', 'fine', 'thanks', 'please']
seq_length = 30

model = load_model('models/0518_02.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
# out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

seq = []
action_seq = []

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        #add=np.zeros(31)
        add=np.full((31,), -1)
        left_loc_x = 0
        left_loc_y = 0
        right_loc_x = 0
        right_loc_y = 0
        print(result.multi_hand_landmarks)
        #print(result.multi_handness)
        #result.multi_handness
        #print(result.multi_handedness)

        for res, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            print(result.multi_handedness)
            a = result.multi_handedness
            hand_side = handedness.classification[0].label
            print(result.multi_handedness[0].classification[0])
            
            #print( a[0].classification[0].label)
            left_hand=np.zeros((21, 4)) 
            right_hand=np.zeros((21, 4)) 
            # 손바닥 중심 좌표 계산
            palm_center_x = int(res.landmark[mp_hands.HandLandmark.WRIST].x * img.shape[1])
            palm_center_y = int(res.landmark[mp_hands.HandLandmark.WRIST].y * img.shape[0])

            #hand_label = res.classification[0].index
            #print("LOKK!!!!!!!!",hand_label)
            # 카메라 뷰에서 손바닥이 좌측에 있는지 우측에 있는지 확인하여 왼손 또는 오른손으로 구분
            
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree
            
            #d = np.concatenate([joint.flatten(), angle])
            print("angle_shape : ", angle.shape) #       => shape: (19, )
            
            if(hand_side == "Left"):
                
                #alpha=np.zeros(100)
                #add=np.concatenate([d_left,alpha])
                add[:15] = angle 
                #d=np.concatenate([d,add])
                print("add_left:")
                print(add.shape)
                left_loc_x = palm_center_x
                left_loc_y = palm_center_y
                
            if(hand_side == "Right"):
                add[15:-1]=angle
                
                print("add_right:")
                print(add.shape)
                right_loc_x = palm_center_x
                right_loc_y = palm_center_y
            
            distance = math.sqrt((left_loc_x - right_loc_x)**2 + (left_loc_y - right_loc_y)**2)

            if(left_loc_x==0 or right_loc_x==0):
                distance = 0
            add[-1] = distance
            print(add)
            #이부분에서 해보자

            
            #data.append(add)



            seq.append(add)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue
            
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            print(input_data.shape)
            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:  
                continue
     
            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action

            cv2.putText(img, f'{this_action.upper()}', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)


    # out.write(img0)
    # out2.write(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break