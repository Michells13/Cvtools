import torchvision.transforms as T
import cv2
import numpy as np
from metrics import voc_eval, mIoU
from utils import readXMLtoAnnotation
from detrModel import initDetr, inferDetr
from yolov8model import initYoloV8, inferYoloV8
from fasterModel import initFaster, inferFaster
import matplotlib.pyplot as plt
import time




# Set device    
device = "cuda"
# Path
videoPath = "../AICity_data/train/S03/c010/vdo.avi"
modelWeights = "./rand1_output/checkpoint.pth"

# Load video
cap = cv2.VideoCapture(videoPath)

# Load model
model = initDetr(device, modelWeights)

# Open file
f = open("det_detr.txt", "w")

# Init detections
BB = np.zeros((0, 4))
imgIds = []
confs = np.array([])

# Not line jump in first line
initial = True

while True:
    
    # Read frame
    ret, frame = cap.read()
    
    # Check if frame was successfully read
    if not ret:
        break
    
    # Get frame
    imageId = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    print(imageId)
    
    # Inference model
    conf, BBoxes = inferDetr(model, frame, device, 0.5)
    BBoxes = BBoxes.cpu().detach().numpy()
    conf = conf.cpu().detach().numpy()
    
    for i in range(conf.shape[0]):
        x = BBoxes[i, 0]
        y = BBoxes[i, 1]
        w = BBoxes[i, 2] - BBoxes[i, 0]
        h = BBoxes[i, 3] - BBoxes[i, 1]
        if initial:
            line = imageId + ",-1,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},-1,-1,-1".format(x, y, w, h, conf[i])
            initial = False
        else:
            line = "\n" + imageId + ",-1,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},-1,-1,-1".format(x, y, w, h, conf[i])

        f.write(line)
# Close txt
f.close()
    

    
    
