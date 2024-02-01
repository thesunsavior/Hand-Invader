import cv2 as cv

import numpy as np
from keras.models import load_model

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import pyautogui

from app.models.auto_agent.agent import Agent

if __name__ == "__main__":
    # load hand gesture classification model
    base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)
    

    mp_hand = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hand_gesture_model = load_model('mp_hand_gesture')

    class_names =['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']

    # initiate camera
    cap = cv.VideoCapture(0)
    agent = Agent()
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Game loop 
with mp_hand.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Error tracking for camera 
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        x , y, c = frame.shape

        screen_x, screen_y =pyautogui.size()
        
        # Flip the frame vertically
        frame = cv.flip(frame, 1)
        framergb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Detections
        results = hands.process(framergb)

        class_name = ''
        
        # get hand position
        if results.multi_hand_landmarks:
            landmarks = []
            for handslms in results.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])
                    
            prediction = hand_gesture_model.predict([landmarks])
            classID = np.argmax(prediction)
            className = class_names[classID]
            cv.putText(frame, className, (10, 50), cv.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv.LINE_AA)

            palm = results.multi_hand_landmarks[0]
            palm_x = palm.landmark[0].x
            palm_y = palm.landmark[0].y
            
            shape = frame.shape 
            relative_x = int(palm_x * shape[1])
            relative_y = int(palm_y * shape[0])

            # movement control
            if className == "stop" or className == "live long" or className == "peace":
                shoot = True
                shoot2 = False
            elif className == "rock":
                shoot = False
                shoot2 = True
            else:
                shoot = False
                shoot2 = False

            agent.shoot_first_gun(shoot)
            agent.shoot_second_gun(shoot2)
            agent.move_mouse (int(palm_x*screen_x), int(palm_y*screen_y))
            
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

