# emotion-detection
Emotion Recognition using OpenCV and Keras
This repository contains a simple Python script for real-time emotion recognition using OpenCV and Keras. The emotion detection model used in this script is fer2013_mini_XCEPTION.102-0.66.hdf5, and it can classify facial expressions into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

Prerequisites
Before running the script, make sure you have the required libraries installed:

bash
Copy code
pip install opencv-python numpy keras
Usage
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/emotion-recognition.git
cd emotion-recognition
Download the pre-trained emotion detection model (fer2013_mini_XCEPTION.102-0.66.hdf5) and place it in the project directory.

Run the script:

bash
Copy code
python emotion_recognition.py
The script will open your camera and display the real-time emotion recognition results.
Description
The script captures video frames from your camera using OpenCV.
It uses Haar Cascade for face detection.
Detected faces are processed, resized to 64x64 pixels, and normalized.
The normalized face is fed into the pre-trained Keras model for emotion prediction.
The predicted emotion is overlaid on the video feed along with a bounding box around the detected face.
References
Haar Cascade Classifier
Keras
Feel free to modify the script and experiment with different models for improved emotion recognition accuracy. If you find any issues or have suggestions, please create an issue or submit a pull request.

