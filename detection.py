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

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w] #region of interest which is our rectangle
        roi = cv2.resize(roi, (64,64)) #resize to 64 by 64 pixels which is normal size
        roi = roi.astype('float') / 255.0 #normalize it
        roi = np.expand_dims(roi, axis =0) #make it more flexible for the model

        emotion = model.predict(roi)[0] #Predict
        emotion_index = np.argmax(emotion) #predict the greatest probability of the emotion made and get the emotion
        emotion_text = emotion_labels[emotion_index] #To actually generate the image

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
