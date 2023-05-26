from utils import readXMLtoAnnotation
from sklearn.model_selection import KFold
import os
import copy
from storeCOCOformat import storeCOCOformatJson
import numpy as np
import cv2

def removeKeys(dictionary, keys):
    """
    This function removes the keys given from the dictionary

    Parameters
    ----------
    dictionary : dict
        Original dictionary.
    keys : list
        List of keys to remove.

    Returns
    -------
    dictionary : dict
        Dictionary with removed keys.

    """
    for key in keys:
        del dictionary[key]
    
    return dictionary

def copyImages(listImages, source, dest):
    """
    This functions copies the list of images from source to dest folder.

    Parameters
    ----------
    listImages : list
        Image name list.
    source : str
        Source folder path.
    dest : str
        Destination folder path.

    Returns
    -------
    None.

    """
    for image in listImages:
        img = cv2.imread(source + image + ".jpg")
        cv2.imwrite(dest + image + ".jpg", img)

# Define paths and classes
annotsPath = "../ai_challenge_s03_c010-full_annotation.xml"
classIds = {"bike": 2, "car": 3}
framesPath = "./frames/"
save_path = "train.json"
folderAnnots = "./datasetSplits/"

# Create folder of new images
if not os.path.exists(folderAnnots):
    os.makedirs(folderAnnots)

# Read images
images = list(range(len(os.listdir(framesPath))))
imagesStr = np.array([str(e) for e in images])

# Load annotations
annots, imageNames = readXMLtoAnnotation(annotsPath, remParked = False)

# 4-fold regular split
# Get folds
k = 4
kf = KFold(n_splits=k)
for i, (test, train) in enumerate(kf.split(images)):
    
    newFolder = folderAnnots + "reg_" + str(i) + "/"
    # Create folder of new split
    if not os.path.exists(newFolder):
        os.makedirs(newFolder)
    
    newFolderTrain = newFolder + "train2017/"
    # Create folder of new split train images
    if not os.path.exists(newFolderTrain):
        os.makedirs(newFolderTrain)
    
    # Copy images
    copyImages(imagesStr[train], framesPath, newFolderTrain)
    
    # Get train annotations
    annotsTrain = copy.deepcopy(annots)
    imageNamesTrain = imagesStr[train]
    annotsTrain = removeKeys(annotsTrain, imagesStr[test])
    
    newFolderVal = newFolder + "val2017/"
    # Create folder of new split train images
    if not os.path.exists(newFolderVal):
        os.makedirs(newFolderVal)
        
    # Copy images
    copyImages(imagesStr[test], framesPath, newFolderVal)
    
    # Get test annotations
    annotsTest = copy.deepcopy(annots)
    imageNamesTest = imagesStr[test]
    annotsTest = removeKeys(annotsTest, imagesStr[train])
    
    newFolderAnnots = newFolder + "annotations/"
    
    # Create folder of split annots
    if not os.path.exists(newFolderAnnots):
        os.makedirs(newFolderAnnots)
    
    # Save train COCO format json
    json_path = newFolderAnnots + "instances_train2017.json"
    storeCOCOformatJson(annotsTrain, imageNamesTrain, classIds, framesPath, json_path)
    
    # Save test COCO format json
    json_path = newFolderAnnots + "instances_val2017.json"
    storeCOCOformatJson(annotsTest, imageNamesTest, classIds, framesPath, json_path)
    
    #print("Train: ", train)
    #print("Test: ", test)

# 4-fold random split
# Get folds
k = 4
kf = KFold(n_splits=k, shuffle=True, random_state = 44)
for i, (test, train) in enumerate(kf.split(images)):
    
    newFolder = folderAnnots + "rand_" + str(i) + "/"
    # Create folder of new split
    if not os.path.exists(newFolder):
        os.makedirs(newFolder)
    
    newFolderTrain = newFolder + "train2017/"
    # Create folder of new split train images
    if not os.path.exists(newFolderTrain):
        os.makedirs(newFolderTrain)
    
    # Copy images
    copyImages(imagesStr[train], framesPath, newFolderTrain)
    
    # Get train annotations
    annotsTrain = copy.deepcopy(annots)
    imageNamesTrain = imagesStr[train]
    annotsTrain = removeKeys(annotsTrain, imagesStr[test])
    
    newFolderVal = newFolder + "val2017/"
    # Create folder of new split train images
    if not os.path.exists(newFolderVal):
        os.makedirs(newFolderVal)
        
    # Copy images
    copyImages(imagesStr[test], framesPath, newFolderVal)
    
    # Get test annotations
    annotsTest = copy.deepcopy(annots)
    imageNamesTest = imagesStr[test]
    annotsTest = removeKeys(annotsTest, imagesStr[train])
    
    newFolderAnnots = newFolder + "annotations/"
    
    # Create folder of split annots
    if not os.path.exists(newFolderAnnots):
        os.makedirs(newFolderAnnots)
    
    # Save train COCO format json
    json_path = newFolderAnnots + "instances_train2017.json"
    storeCOCOformatJson(annotsTrain, imageNamesTrain, classIds, framesPath, json_path)
    
    # Save test COCO format json
    json_path = newFolderAnnots + "instances_val2017.json"
    storeCOCOformatJson(annotsTest, imageNamesTest, classIds, framesPath, json_path)
    
    #print("Train: ", train)
    #print("Test: ", test)