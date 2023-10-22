# Week 3

During the third week of the project, two main tasks were tackled: object detection and tracking. To achieve this goal, we initially tried different pretrained models (Faster RCNN, YoloV8, and DETR), and then we fine-tuned the DETR model. For tracking, we experimented with two different methods: BBox overlapping and Kalman filters.

## Available tasks

* **Task 1.1**: Off-the-shelf:
	* ``inferenceModels.py`` : This script can be used to create and evaluate the predictions of Faster RCNN, YoloV8, and DETR pretrained models on video frames. The model name can be defined in the script. The script provides mAP, mIoU, mean inference time (s/frame), and stores a GIF with the prediction bounding boxes.
* **Task 1.2**: Annotations: Roboflow was used (more information in the report slides). Results in different format are in the folder ``annotations_s05_c10/``. To visualize the results ``visualizeAnnotations.py`` script was used.
* **Task 1.3**: Fine-tune the DETR model to our data:
	* ``trainDetr.py`` : This script can be used to fine-tune the DETR model using weights from the COCO dataset. In this script, the dataset folder (in COCO 2017 format) can be defined. This script uses the official training script, so it is important to clone the repository before executing it:
``git clone https://github.com/facebookresearch/detr.git``

In order to split the dataset and parse it to COCO format, we used the ``splitDataset.py`` and ``storeCOCOformat.py`` scripts.

* **Task 1.4**: Cross validation
	* The same ``splitDataset.py`` and ``storeCOCOformat.py`` scripts can be used to save sequential and random partition (4 fold) of the frames (always 25% of the frames for training and the rest for validation). 
	* In order to fine-tune the DETR with different datasets same ``trainDetr.py`` script can be used. 
	* Use `inferenceFineTunedModel.py` script to evaluate the validation set results, such as mAP and mIoU, by defining the fine-tuned weights path.

For the next tracking methods model detections in a MOT format were used. To save the detections in MOT format, whether using fine-tuned or pre-trained models, use the `createModelDetections.py` script.
* **Task 2.1**: Overlapping tracking
	* Use the `task_2_1.py` script to assign an ID to each bounding box detection by using the highest overlapping of bounding boxes through frames method.
* **Task 2.2**: Tracking using Kalman filters
	* Replace the images of the video, detection, ground truth file, and the sequence info file in their respective folders (MOT 15 database format).
	* Use `sort.py` script (original implementation: [https://github.com/abewley/sort](https://github.com/abewley/sort)) to track and obtain the results.
 * **Task 2.3**: Evaluate tracking
	* Clone the [TrackEval](https://github.com/JonathonLuiten/TrackEval) repository by running `git clone https://github.com/JonathonLuiten/TrackEval.git`.
	* Create the `data/` folder with the same format as the MOT15-all challenge to evaluate.
	* Evaluate using the following command:
``python run_mot_challenge.py --GT_FOLDER f:/cv_material/m6/week3/data/gt/mot_challenge/ --TRACKERS_FOLDER f:/cv_material/m6/week3/data/trackers/mot_challenge/ --BENCHMARK 'MOT15' --SPLIT_TO_EVAL 'all' --DO_PREPROC False``

		Make sure the paths of the folders are correct.