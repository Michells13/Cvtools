import torch
import cv2
from PIL import Image
import torchvision.transforms as T

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def box_cxcywh_to_xyxy(x):
    """
    This function parses from cxcywh bbox format to xyxy.

    Parameters
    ----------
    x : torch tensor
        Predicted bboxes.

    Returns
    -------
    torch tensor
        BBoxes in xyxy format.

    """
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    """
    This function parses the predicted bboxes to xyxy format (absolute positions).

    Parameters
    ----------
    out_bbox : torch tensor
        Model predicted bboxes.
    size : tuple
        Original image size.

    Returns
    -------
    b : torch tensor
        BBoxes in xyxy format in absolute positions.

    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def initDetr(device, fineTunedModel = None):
    """
    This function inits the DETR model with COCO pretrained weights.

    Parameters
    ----------
    device : str
        Device where the model should be stored.

    Returns
    -------
    model : model
        DETR model with COCO weights.

    """
    # Load model and set as eval
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    if fineTunedModel is not None:
        checkpoint = torch.load(fineTunedModel, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
    # Model to device
    model = model.to(device)
    model.eval()
    
    return model

def inferDetr(model, image, device, confThresh = 0.5):
    """
    This function inferes the given image using the given DETR model. 
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
    
    # Get detection with a confidence bigger than 0.5
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > confThresh
    probas = probas[keep]
    
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), image.size)
    
    # Get only bicycle and car predictions
    arg = probas.argmax(axis = 1)
    keep = torch.logical_or((arg == 3).cpu(), (arg == 2).cpu())
    probas = probas[keep, arg[keep]]
    bboxes_scaled = bboxes_scaled[keep, :]
    
    
    return probas, bboxes_scaled