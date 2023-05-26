import torchvision.transforms as T
import cv2
import numpy as np
from metrics import voc_eval, mIoU
from utils import readXMLtoAnnotation, drawBoxes
from detrModel import initDetr, inferDetr
from yolov8model import initYoloV8, inferYoloV8
from fasterModel import initFaster, inferFaster
import matplotlib.pyplot as plt
import time
import imageio

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


models = {"detr": (initDetr, inferDetr), "faster": (initFaster, inferFaster), "yolov8": (initYoloV8, inferYoloV8)}
modelType = "detr"
initModel, inferModel = models[modelType]

# Set device    
device = "cuda"
# Path
videoPath = "C:/Users/MICHE/Documents/Datasets/aic19-track1-mtmc-train/train/S03/c010/vdo.avi"
annotsPath = "C:/Users/MICHE/Documents/Datasets/ai_challenge_s03_c010-full_annotation.xml"

# Load annotations
annots, imageNames = readXMLtoAnnotation(annotsPath, remParked = False)

# Load video
cap = cv2.VideoCapture(videoPath)

# Load model
model = initModel(device)

# Init detections
BB = np.zeros((0, 4))
imgIds = []
confs = np.array([])

inferenceTimes = []

# Init gif frame list
gif_boxes = []

int_id = -1

while True:
    int_id += 1
    
    # Read frame
    ret, frame = cap.read()
    
    # Check if frame was successfully read
    if not ret:
        break
    
    # Get frame
    imageId = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
    print(imageId)
    
    # Inference model
    startTime = time.time()
    conf, BBoxes = inferModel(model, frame, device, 0.5)
    stopTime = time.time()
    inferenceTimes.append(stopTime - startTime)
    BBoxes = BBoxes.cpu().detach().numpy()
    conf = conf.cpu().detach().numpy()
    imageIds = [imageId]*len(conf)
    
    # Plot
    if int_id % 10 == 0:
        print(imageId)
        
        if imageId in annots.keys():
            plotBoxes = drawBoxes(frame, BBoxes, annots[imageId], [255, 0, 0], [0, 255, 0])
        else:
            plotBoxes = drawBoxes(frame, BBoxes, [], [255, 0, 0], [0, 255, 0])
        
        plotBoxes = cv2.resize(plotBoxes, (1000, 500))
        
        # Store plots
        gif_boxes.append(plotBoxes)
    
    # Store predictions
    imgIds = imgIds + imageIds
    BB = np.vstack((BB,BBoxes))
    confs = np.concatenate((confs, conf))
    
    
# Compute mAP
_,_, ap = voc_eval((imgIds, confs, BB), annots, imageNames)
print("mAP: ", ap)

# Estimate mIoU
miou = mIoU((imgIds, confs, BB), annots, imageNames)
print("mIoU: ", miou)

# Mean inference time
print("Mean inference time (sec/frame): ", np.mean(np.array(inferenceTimes)))

imageio.mimsave('resultsPretrained' + modelType + '.gif', gif_boxes, duration=500)