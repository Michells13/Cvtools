from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image
from utils import readXMLtoAnnotation


def xyxy_to_xywh(xyxy):
    """
    Convert XYXY format (x,y top left and x,y bottom right) to XYWH format (x,y top left and width, height).
    :param xyxy: [X1, Y1, X2, Y2]
    :return: [X, Y, W, H]
    """
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])
    return [xyxy[0], xyxy[1], w_temp, h_temp]


def storeCOCOformatJson(annots, imageNames, classIds, framesPath, json_path):
    """
    This functions takes annotations and stores the json in COCO format.

    Parameters
    ----------
    annots : dict
        Annotations of images.
    imageNames : list
        Name of images.
    classIds : dict
        Class names with ids.
    framesPath : str
        Path of the frame images.
    json_path : str
        Path to store the result json.

    Returns
    -------
    None.

    """

    # Init coco
    coco = Coco()
    
    # Add classes
    for className in classIds.keys():
        
        coco.add_category(CocoCategory(id=classIds[className], name=className))
        coco.add_category(CocoCategory(id=classIds[className], name=className))
    
    # Add images
    for imageName in imageNames:
        
        # Read image and add image
        imageFullName = imageName + ".jpg"
        width, height = Image.open(framesPath + imageFullName).size
        coco_image = CocoImage(file_name=imageFullName, height=height, width=width)
        
        # Add annotations
        imageAnnots = annots[imageName]
        for ann in imageAnnots:
            coco_image.add_annotation(
                CocoAnnotation(
                bbox=xyxy_to_xywh(ann["bbox"]),
                category_id=classIds[ann["name"]],
                category_name=ann["name"]
                )
            )
    
        # Add image
        coco.add_image(coco_image)
    
    # Store json
    save_json(data=coco.json, save_path=json_path)