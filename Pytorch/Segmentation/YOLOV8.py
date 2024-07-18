# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:19:05 2024

@author: MICHE
"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
import cv2
model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
#model.train(data='coco128.yaml', epochs=3)  # train the model

result=model('https://ultralytics.com/images/bus.jpg',show=True)  # predict on an image
print(result)

