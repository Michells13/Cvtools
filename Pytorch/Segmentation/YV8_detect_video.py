# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:19:05 2024

@author: Michell
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
from ultralytics.solutions import heatmap
from collections import defaultdict
import numpy as np
import cv2

'''
YOLO V8  implementation (Ultralitics): 
        it uses opencv webcam caputure to feed the network
        there exist three methods to use YOLO V8, object detection, segmentation,
        clasification, all of them can be applied within this code by selecting
        the task  and there is also an option to perform traking by turning True 
        that option ..
        
        
        -Inputs:   
            MODE SELECTION OPTIONS 
                    MODE:   Inference       @This code only can be used to perform inference@
                    TASK:
                        "detection"     - Object detection
                        "segmentation"  - Segmentation
                        "PE"            - Pose estimation
                        "classification'- Clasification
                        "obb"           - Oriented object detection 
                        
             INFERENCE ARGUMENTS:       ...............
             VIZUALIZATION ARGUMENTS:   ...............
             OUTPUTS:                   ...............

'''

#Mode selection options
MODE= "Inference"   #Training, validation, export, track
TASK="detection"    #"detection", "segmentation", "PE", "classification",OBB     


#Inference arguments
tracker =1          # Enable tracking   0- Disable,  1- BoT-SORT, 2- ByteTrack,
conf=0.4            # Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.
iou=0.7             # Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.
half=False          # Enables half-precision (FP16) inference
traclet=False       # creates tracklets
display_traclets=False # displays tracklets
persist=True        
imgsz=640           # Defines the image size for inference. Can be a single integer 640 for square resizing or a (height, width) tuple. Proper sizing can improve detection accuracy and processing speed.
device= None        # Specifies the device for inference (e.g., cpu, cuda:0 or 0). Allows users to select between CPU, a specific GPU, or other compute devices for model execution.
max_det= 20         # Maximum number of detections allowed per image. Limits the total number of objects the model can detect in a single inference, preventing excessive outputs in dense scenes.
heatmap =False      # Heatmap activation 
vid_stride=1        # Frame stride for video inputs. Allows skipping frames in videos to speed up processing at the cost of temporal resolution. A value of 1 processes every frame, higher values skip frames.
stream_buffer=False # Determines if all frames should be buffered when processing video streams (True), or if the model should return the most recent frame (False). Useful for real-time applications.
visualize=False     # Activates visualization of model features during inference, providing insights into what the model is "seeing". Useful for debugging and model interpretation.
augment=False       # Enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed.
agnostic_nms=False  # Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common.
classes= None       # list[int] : Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned. Useful for focusing on relevant objects in multi-class detection tasks.
retina_masks= False # Uses high-resolution segmentation masks if available in the model. This can enhance mask quality for segmentation tasks, providing finer detail.
embed=None          # list[int] : Specifies the layers from which to extract feature vectors or embeddings. Useful for downstream tasks like clustering or similarity search.
Stream=False        # Use stream=True for processing long videos or large datasets to efficiently manage memory. When stream=False, the results for all frames or data points are stored in memory, which can quickly add up 
                    #and cause out-of-memory errors for large inputs. In contrast, stream=True utilizes a generator, which only keeps the results of the current frame or data point in memory, significantly reducing memory consumption and preventing out-of-memory issues.

#Visualization arguments:
Show        =True   # If True, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing.
save		=False	# Enables saving of the annotated images or videos to file. Useful for documentation, further analysis, or sharing results.
save_frames	=False	# When processing videos, saves individual frames as images. Useful for extracting specific frames or for detailed frame-by-frame analysis.
save_txt	=False	# Saves detection results in a text file, following the format [class] [x_center] [y_center] [width] [height] [confidence]. Useful for integration with other analysis tools.
save_conf	=False	# Includes confidence scores in the saved text files. Enhances the detail available for post-processing and analysis.
save_crop	=False	# Saves cropped images of detections. Useful for dataset augmentation, analysis, or creating focused datasets for specific objects.
show_labels	=True	# Displays labels for each detection in the visual output. Provides immediate understanding of detected objects.
show_conf   =True	# Displays the confidence score for each detection alongside the label. Gives insight into the model's certainty for each detection.
show_boxes  =True	# Draws bounding boxes around detected objects. Essential for visual identification and location of objects in images or video frames.
line_width	=None   # Specifies the line width of bounding boxes. If None, the line width is automatically adjusted based on the image size. Provides visual customization for clarity.

''' Results objects have the following attributes:

Attribute |     Type      | Description

orig_img	numpy.ndarray	The original image as a numpy array.
orig_shape	tuple	        The original image shape in (height, width) format.
boxes	    Boxes,          optional	A Boxes object containing the detection bounding boxes.
masks	    Masks,          optional	A Masks object containing the detection masks.
probs	    Probs,          optional	A Probs object containing probabilities of each class for classification task.
keypoints	Keypoints,      optional	A Keypoints object containing detected keypoints for each object.
obb	        OBB,            optional	 An OBB object containing oriented bounding boxes.
speed	    dict	        A dictionary of preprocess, inference, and postprocess speeds in milliseconds per image.
names	    dict	        A dictionary of class names.
path	    str	            The path to the image file.'''

'''Results objects have the following methods:

Method	   Return Type	      Description
update()	None	      Update the boxes, masks, and probs attributes of the Results object.
cpu()	    Results	      Return a copy of the Results object with all tensors on CPU memory.
numpy()	    Results	         Return a copy of the Results object with all tensors as numpy arrays.
cuda()	    Results	      Return a copy of the Results object with all tensors on GPU memory.
to()	    Results	      Return a copy of the Results object with tensors on the specified device and dtype.
new()	    Results	      Return a new Results object with the same image, path, and names.
plot()	    numpy.ndarray Plots the detection results. Returns a numpy array of the annotated image.
show()	    None	      Show annotated results to screen.
save()	    None	      Save annotated results to file.
verbose()	str	         Return log string for each task.
save_txt()	None	     Save predictions into a txt file.
save_crop()	None	     Save cropped predictions to save_dir/cls/file_name.jpg.
tojson()	str	        Convert the object to JSON format.'''





#Output
      


def traclets(results):
        # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Plot the tracks
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)
            
            
        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
    return annotated_frame

if TASK=="detection" :    
    model = YOLO('yolov8n.pt')
if TASK=="segmentation" :  
    model = YOLO('yolov8n-seg.pt')   
if TASK=="PE" :  
    model = YOLO('yolov8n-pose.pt')  
if TASK=="classification" :
    model = YOLO('yolov8n-cls.pt')  
if TASK=="obb" :    
    model = YOLO('yolov8n-obb.pt')  # load an official model
   

cap = cv2.VideoCapture(0)
# Store the track history for traclets
track_history = defaultdict(lambda: [])
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
if heatmap:
    # Init heatmap
    heatmap_obj = heatmap.Heatmap()
    heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                         imw=w,
                         imh=h,
                         view_img=True,
                         shape="circle")

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


# Perform tracking with the model

    if tracker==0 :
      results = model.predict(source=frame, show=Show, stream=Stream) 
    if tracker==1 :
       results = model.track(source=frame, conf=conf, iou=iou,show=Show,persist=persist, stream=Stream)  # Tracking with default tracker
    if tracker==2 :
       results = model.track(source=frame, conf=conf, iou=iou,show=Show, tracker="bytetrack.yaml", persist=persist, stream=Stream)  # Tracking with ByteTrack tracker
    if heatmap==True:
        im0 = heatmap_obj.generate_heatmap(frame, results)
        #cv2.imshow('heatmap', im0)

    if tracker > 0 and traclet==True:
       annotated_frame= traclets(results)
       # Display the annotated frame
       if display_traclets== True:
           cv2.imshow("YOLOv8 Traclets", annotated_frame)
    
    #boxes = results[0].boxes.xywh.cpu()
    #track_ids = results[0].boxes.id.int().cpu().tolist()
#for result in results:
    # Detection
    #result.boxes.xyxy   # box with xyxy format, (N, 4)
    #result.boxes.xywh   # box with xywh format, (N, 4)
    #result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    #result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    #result.boxes.conf   # confidence score, (N, 1)
    #result.boxes.cls    # cls, (N, 1)

    # Segmentation
    #result.masks.data      # masks, (N, H, W)
    #result.masks.xy        # x,y segments (pixels), List[segment] * N
    #result.masks.xyn       # x,y segments (normalized), List[segment] * N

    # Classification
    #result.probs     # cls prob, (num_class, )

    # Each result is composed of torch.Tensor by default,
    # in which you can easily use following functionality:
    #result = result.cuda()
    #result = result.cpu()
    #result = result.to("cpu")
    #result = result.numpy()
    
    
    
    
    # Esperar 25 milisegundos antes de mostrar el siguiente frame
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Liberar el objeto de captura y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()



# https://docs.ultralytics.com/modes/predict/#images