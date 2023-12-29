import cv2
import numpy as np 
from keras.models import load_model

model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5') #load the machine learning model that detects the emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] #emotions we wanna detect
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #haarcascade classifier loaded

cap = cv2.VideoCapture(0) #Turn on Camera

while True: #Capture every frame
    ret, frame = cap.read()
    if not ret: #if its not successful break
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #make the picture gray
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)) #face detection model
