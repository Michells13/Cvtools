import torchvision.transforms as T
import cv2
import numpy as np
from metrics import voc_eval, mIoU
from utils import readXMLtoAnnotation, drawBoxes
from detrModel import initDetr, inferDetr
from yolov8model import initYoloV8, inferYoloV8
from fasterModel import initFaster, inferFaster
import matplotlib.pyplot as plt
import os
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

def removeKeys(dictionary, keys):
    """
    This function removes the keys that are not in the given list from the dictionary

    Parameters
    ----------
    dictionary : dict
        Original dictionary.
    keys : list
        List of keys not to remove.

    Returns
    -------
    dictionary : dict
        Dictionary with removed keys.

    """
    keyValues = list(dictionary.keys())
    for key in keyValues:
        if key not in keys:
            del dictionary[key]
    
    return dictionary

# Set device    
device = "cuda"
# Path
datasetPath = "C:/Users/MICHE/Documents/Datasets/datasetSplits/rand_0/train2017/"
annotsPath = "C:/Users/MICHE/Documents/Datasets/ai_challenge_s03_c010-full_annotation.xml"
modelFineTuningType = "rand0_output"
pathmodel='C:/Users/MICHE/Documents/Datasets/datasetSplits/rand0_output/checkpoint.pth'
modelWeights = pathmodel #"./" + modelFineTuningType + "/checkpoint.pth"

# Load annotations
annots, imageNames = readXMLtoAnnotation(annotsPath, remParked = False)
imageNames = os.listdir(datasetPath)
imageNames = [e[:-4] for e in imageNames]
imageIds = np.array(imageNames)
imageIds = imageIds.astype(np.int32)
imageOrder = np.argsort(imageIds)
imageNames = np.array(imageNames)[imageOrder]
annots = removeKeys(annots, imageNames)

# Load model
model = initDetr(device, modelWeights)

# Init detections
BB = np.zeros((0, 4))
imgIds = []
confs = np.array([])

# Init gif frame list
gif_boxes = []

int_id = -1

for img in imageNames:
    int_id += 1
    
    # Read image
    frame = cv2.imread(datasetPath + img + ".jpg")  
    imageId = img
    print(imageId)
    
    # Inference model
    conf, BBoxes = inferDetr(model, frame, device, 0.5)
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

imageio.mimsave('results_fine_tuned' + modelFineTuningType + '.gif', gif_boxes, duration=500)