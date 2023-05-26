import torchvision.transforms as T
import cv2
import numpy as np
from utils import readXMLtoAnnotation, drawBoxes
import matplotlib.pyplot as plt
import imageio
from pycocotools.coco import COCO

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


# Set device    
device = "cuda"
# Path
videoPath = "./AICity_data_S05_C010/validation/S05/c010/vdo.avi"
annotsPath = "./annotations_s05_c10/_annotations.coco.json"

# Load annotations
coco=COCO(annotsPath)

# Load video
cap = cv2.VideoCapture(videoPath)

# Init detections
BB = np.zeros((0, 4))
imgIds = []
confs = np.array([])

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
    imageId = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    print(imageId)
    
    if int(imageId) < 11 + 1:
        continue
    
    # Plot
    if int_id % 30 == 0:
        # Find id with filename (this should not be necessary if the order is correct)
        for imgKey in coco.imgs.keys():
            img = coco.imgs[imgKey]
            if img["file_name"].split("_")[1] == str(int(imageId) - 11):
                imgId = img['id']
                break
        annotsId = coco.getAnnIds(imgId)
        annots = coco.loadAnns(annotsId)
        
        BBoxes = np.zeros((0,4))
        for annot in annots:
            xmin = annot["bbox"][0]
            ymin = annot["bbox"][1]
            xmax = annot["bbox"][0] + annot["bbox"][2]
            ymax = annot["bbox"][1] + annot["bbox"][3]
            box = np.array([[xmin, ymin, xmax, ymax]])
            BBoxes = np.vstack((BBoxes,box))
        
        print(imageId)
        plotBoxes = drawBoxes(frame, BBoxes, [], [255, 0, 0], [0, 255, 0])
        plotBoxes = cv2.resize(plotBoxes, (500, 250))
        
        # Store plots
        gif_boxes.append(plotBoxes)
    

imageio.mimsave('annotations.gif', gif_boxes, fps=4)