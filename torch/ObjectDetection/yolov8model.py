from ultralytics import YOLO
import cv2
from PIL import Image
import torch

def initYoloV8(device):
    """
    This function inits the YoloV8x model with COCO pretrained weights.

    Parameters
    ----------
    device : str
        Device where the model should be stored.

    Returns
    -------
    model : model
        YoloV8x model with COCO weights.

    """
    model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
    #model = model.cuda()
    
    return model

def inferYoloV8(model, image, device, confThresh = 0.5):
    """
    This function inferes the given image using the given YoloV8 model. 
    Outputs the bboxes of only confidence more than the threshold of confidence and
    only car class objects.

    Parameters
    ----------
    model : model
        Model to do the inference.
    image : torch tensor
        Image to get the detections from.
    device : str
        Device where the inferece should be done.
    confThresh : float, optional
        Minimum confidence value of the detections. The default is 0.5.

    Returns
    -------
    probas : torch tensor
        Confidence values of the detections.
    bboxes_scaled : torch tensor
        Detected car bboxes.

    """
    # Set to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    
    # Inference
    outputs = model.predict(image)
    
    # Obtain detections
    boxes = outputs[0].boxes.xyxy
    conf = outputs[0].boxes.conf
    classes = outputs[0].boxes.cls
    
    # Get detection with a confidence bigger than 0.5
    keep = conf > confThresh
    boxes = boxes[keep, :]
    conf = conf[keep]
    classes = classes[keep]
    
    # Get only bicycle and car predictions
    keep = torch.logical_or((classes == 2).cpu(), (classes == 1).cpu())
    conf = conf[keep]
    boxes = boxes[keep, :]
    
    
    return conf, boxes