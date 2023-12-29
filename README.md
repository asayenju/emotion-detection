# Emotion Detection System using OpenCV, Tensorflow Keras and Numpy

This repository contains a simple Python script for real-time emotion recognition using OpenCV and Keras. The emotion detection model used in this script is fer2013_mini_XCEPTION.102-0.66.hdf5, and it can classify facial expressions into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

# Prerequisites
Before running the script make sure you have required libraries installed

```console
pip install opencv-python keras numpy
```

# Usage
1. Clone the repository
```console
git clone https://github.com/your-username/emotion-recognition.git
cd emotion-recognition
```
2. Download the pre-trained emotion detection model (fer2013_mini_XCEPTION.102-0.66.hdf5) and place it in the project directory.
3. Run the Script:
```console
python detection.py
```

The script will open your camera and display real-time emotion recognition results

# Description
* The script captures video frames from your camera using OpenCV.
* It uses Haar Cascade for face detection.
* Detected faces are processed, resized to 64x64 pixels, and normalized.
* The normalized face is fed into the pre-trained Keras model for emotion prediction.
* The predicted emotion is overlaid on the video feed along with a bounding box around the detected face.



