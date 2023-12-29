import cv2
import numpy as np 
from keras.models import load_model

model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5') #load the machine learning model that detects the emotions

