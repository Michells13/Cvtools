import torch
import cv2
from PIL import Image
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import torchvision.transforms as T
import numpy as np

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    #T.Resize(1024),
    T.ToTensor()
])

def initFaster(device):
    """
    This function inits the Faster RCNN model with COCO pretrained weights.

    Parameters
    ----------
    device : str
        Device where the model should be stored.

    Returns
    -------
    model : model
        YoloV8x model with COCO weights.

    """
    # load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn_v2(pretrained = True)
    
    # Model to device
    model = model.to(device)
    model.eval()
    
    return model

def inferFaster(model, image, device, confThresh = 0.5):
    """
    This function inferes the given image using the given Faster RCNN model. 
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
    # Transform image
    img = transform(image).unsqueeze(0)
    img = img.to(device)
    
    # Inference
    outputs = model(img)
    
    # Obtain detections
    boxes = outputs[0]["boxes"]
    conf = outputs[0]["scores"]
    classes = outputs[0]["labels"]
    
    # Get detection with a confidence bigger than 0.5
    keep = conf > confThresh
    boxes = boxes[keep, :]
    conf = conf[keep]
    classes = classes[keep]
    
    # Get only bicycle and car predictions
    keep = torch.logical_or((classes == 3).cpu(), (classes == 2).cpu())
    conf = conf[keep]
    boxes = boxes[keep, :]
    
    
    return conf, boxes
