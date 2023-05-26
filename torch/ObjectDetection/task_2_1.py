import cv2
from sort import iou_batch, convert_bbox_to_z, convert_x_to_bbox
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

class Detection:
    def __init__(self, frame, ID, coords):
        self.frame = frame
        self.coords = coords
        self.ID = ID

    def __str__(self):
        return f"{self.frame}, {self.ID}, {self.coords}"

def computeIoU(detection1, detection2):
    return bb_intersection_over_union(detection1[2:6], detection2[2:6])

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def computeOverlapMaximization(dict, current_id, nr_frames, threshold = 0.8):

    result = []
    [result.append(x) for x in dict[1]] # attach the first frames

    # check last 5 previous frames for a bounding box (in case of occlusions)
    BACK_IN_TIME_FRAMES = 5
    
    # start from second frame, as the first one has the IDs
    for current_frame in range(2, nr_frames+1):
        # Do not repeat assigments
        alreadyAssigId = []
        for current_frame_detection in dict[current_frame]: # take all bboxes from current frame
            IoUValue = float('-inf')
            assignedID = -1

            previous_frames = [*range(max(current_frame - BACK_IN_TIME_FRAMES, 1), current_frame)]
    
            previous_detections = [det for key in previous_frames for det in dict.get(key)]
            
            for previous_detection in previous_detections:
                IoU = computeIoU(current_frame_detection, previous_detection)
                
                if(IoU >= threshold and IoU > IoUValue and not (previous_detection[1] in alreadyAssigId)): # get the maximum IoU above the threshold
                    IoUValue = IoU
                    assignedID = previous_detection[1]
                    
            if assignedID != -1:
                current_frame_detection[1] = assignedID
                alreadyAssigId.append(assignedID)
            elif assignedID == -1: # didn't find anything that has an IoU big enought
                current_frame_detection[1] = current_id # new detection
                alreadyAssigId.append(current_id)
                current_id += 1
            
            result.append(current_frame_detection)

    return result

if __name__ == '__main__':
    display = True
    # colours = np.random.rand(32, 3) #used only for display
    colours = np.random.randint(0, 256, size=(32, 3))
    

    input_file = "det_detr.txt"
    seq_dets = np.loadtxt(input_file, delimiter=',') # array with every line in the file

    current_ID = 1
    # create dictionary to store the detections from each frame (faster than list) in paris <FRAME_NR, DETECTIONS_ARRAY>
    dict = {}
    nr_frames = 0

    for detection in seq_dets:
        frame = int(detection[0])
        nr_frames = max(nr_frames, frame)
        id = detection[1]
        rest = detection[2:7] 
        rest[2:4] += rest[0:2] # convert from [x, y, bb_width, bb_height] to [x1, y1, x2, y2]

        if(frame == 1):
            id = current_ID # set the first bboxes IDs in the first frame
            current_ID += 1
        
        if frame in dict:
            det = [frame, id, rest[0], rest[1], rest[2], rest[3], rest[4]]
            dict[frame] = dict[frame] + [det]
        else:
            det = [frame, id, rest[0], rest[1], rest[2], rest[3], rest[4]]
            dict[frame] = [det]

    results = computeOverlapMaximization(dict, current_ID, nr_frames)

    #print(result[0])
    """
    output_file = 'iou_based_tracking.txt'
    # Open file
    f = open(output_file, "w")
    
    initial = True
    
    for result in results:
        imageId = str(result[0])
        objId = str(result[1])
        x = result[2]
        y = result[3]
        w = result[4] - result[2]
        h = result[5] - result[3]
        conf = result[6]
        if initial:
            line = imageId + "," + objId + ",{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},-1,-1,-1".format(x, y, w, h, conf)
            initial = False
        else:
            line = "\n" + imageId + "," + objId + ",{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},-1,-1,-1".format(x, y, w, h, conf)

        f.write(line)

    # Close txt
    f.close()
    """

    if(display):
        
        # Set device    
        device = "cuda"
        # Path
        videoPath = "../AICity_data/train/S03/c010/vdo.avi"

        # Load video
        cap = cv2.VideoCapture(videoPath)

        # current frame number
        current_frame = 0

        while True:
             # Read frame
            ret, frame = cap.read()
            
            # Check if frame was successfully read
            if not ret:
                break

            for detection in results:
                if detection[0] == current_frame:
                    # draw the boxes on screen
                    
                    id = detection[1]
                    positions = detection[2:6]# [x1, y1, x2, y2]
                    color = np.uint8(colours[id%32, :])
                    c = tuple(map(int, color))

                    cv2.rectangle(frame, (int(positions[0]), int(positions[1])), (int(positions[2]), int(positions[3])), color=c, thickness=2)
                    cv2.putText(frame, str(id), (int(positions[0]), int(positions[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color=c, thickness=2)

            cv2.imshow('Frame', frame)
            
            current_frame += 1
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
