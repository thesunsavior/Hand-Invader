import os
import cv2 as cv

import torch
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PIL import Image
from app.models.hand_detector import get_transform
import pyautogui

# from app.models.hand_detector import hand_detection_model, images_valid_dataloader, get_transform
# from app.util.train_util import inference, draw_bbox
# from app.util.eval_util import evaluate, plot_loss, plot_image


if __name__ == "__main__":
    # if need for training 
    # model = train_hand_detection(hand_detection_model)
    
    # load hand detection model
    # checkpoint = torch.load('hand_detection.ckpt')
    # hand_detection_model.load_state_dict(checkpoint['model_state_dict'])

    # val_loss =evaluate(hand_detection_model, images_valid_dataloader,'cpu')
    # plot_loss(valid_loss=val_loss)


    # load hand gesture classification model
    base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)
    

    mp_hand = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hand_gesture_model = load_model('mp_hand_gesture')

    class_names =['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']

    # preprocessing
    transform = get_transform()

    # initiate camera
    cap = cv.VideoCapture(0)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 300)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 300)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Game loop 
    
def shoot_first_gun(shoot):
    try:
        if shoot:
            pyautogui.mouseDown()
        else:
            pyautogui.mouseUp()
    except KeyboardInterrupt:
        print('\n')    

def shoot_second_gun(shoot):
    try:
        if shoot:
            pyautogui.mouseDown(button="right")
        else:
            pyautogui.mouseUp(button="right")
    except KeyboardInterrupt:
        print('\n')
        
def move_mouse(x_pos, y_pos):
    try:
        pyautogui.moveTo(x_pos, y_pos)
    except KeyboardInterrupt:
        print('\n')
        
shoot = False
shoot2 = False
with mp_hand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while True:
        shoot_first_gun(shoot)
        shoot_second_gun(shoot2)
        # Capture frame-by-frame
        ret, frame = cap.read()
        # frame = cv.flip(frame, 1)
        
        # Error tracking for camera 
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        x , y, c = frame.shape
        # print(x, y, c)
        size = pyautogui.size()
        # print(size)
        ratio = np.array([size[0]/x, size[1]/y])
        
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
                    
            pos = np.array([landmarks[0],landmarks[9]]).mean(axis=0)
            x_pos, y_pos = pos*ratio
            print(x_pos, y_pos)
            move_mouse(x_pos, y_pos)
            # break
            prediction = hand_gesture_model.predict([landmarks[:21]])
            classID = np.argmax(prediction)
            className = class_names[classID]
            print(className)
            if className == "stop" or className == "live long" or className == "peace":
                shoot = True
                shoot2 = False
            elif className == "rock":
                shoot = True
                shoot2 = True
            else:
                shoot = False
                shoot2 = False
                
            # cv.putText(frame, className, (10, 50), cv.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv.LINE_AA)

            # palm = results.multi_hand_landmarks[0]
            # palm_x = palm.landmark[0].x
            # palm_y = palm.landmark[0].y
            
            # shape = frame.shape 
            # relative_x = int(palm_x * shape[1])
            # relative_y = int(palm_y * shape[0])

            # image = cv.circle(frame, (relative_x, relative_y), radius=20, color=(225, 0, 100), thickness=1)    
            
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

