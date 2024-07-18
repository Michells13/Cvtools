# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:19:05 2024

@author: MICHE
"""

import socket
from PyQt5.QtWidgets import QApplication, QLabel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
import cv2




model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
#model.train(data='coco128.yaml', epochs=3)  # train the model
cap = cv2.VideoCapture(0)

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print("Error al abrir el video")
    exit()

# Bucle para leer y mostrar cada frame del video
while True:
    # Leer un frame del video
    ret, frame = cap.read()

    # Verificar si se ha llegado al final del video
    if not ret:
        break

    # Mostrar el frame
    # from ndarray

    results = model.predict(source=frame, show=True)  # save predictions as labels
    #cv2.imshow('Video', frame)

    # Esperar 25 milisegundos antes de mostrar el siguiente frame
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Liberar el objeto de captura y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()

















# result=model('https://ultralytics.com/images/bus.jpg',show=True)  # predict on an image
# print(result)

