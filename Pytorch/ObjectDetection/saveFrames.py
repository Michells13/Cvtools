import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Path
videoPath = "../AICity_data/train/S03/c010/vdo.avi"
videoFrames = "./frames/"

# Load video
cap = cv2.VideoCapture(videoPath)

# Create folder of new images
if not os.path.exists(videoFrames):
    os.makedirs(videoFrames)

while True:
    
    # Read frame
    ret, frame = cap.read()
    
    # Check if frame was successfully read
    if not ret:
        break
    
    # Get frame
    imageId = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
    print(imageId)
    
    # Save image
    cv2.imwrite(videoFrames + imageId + ".jpg", frame)

print("Done!")