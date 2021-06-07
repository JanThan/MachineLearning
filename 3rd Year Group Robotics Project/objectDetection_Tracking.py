# This program performs the actual detection of each of the moving 'pucks'. Default is to call then tensorflow built cnn model to detect all pucks within the video frame. 
# object tracking will then be applied to follow each of the identified bounding boxes. 
# There is the option to stop all pucks moving and manually draw each of these bounding boxes if it is required (as a last resort backup option)

import numpy as np
import os
import tensorflow as tf
import imutils
import time
import cv2
import argparse


from utils import label_map_util
from utils import visualization_utils as vis_util
from imutils.video import VideoStream
from imutils.video import FPS

# ------------------------------ # Definitions # ------------------------------ #
def object_detection(image_np_expanded,detection_limit):
  
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
  # Each box represents a part of the image where a particular object was detected.
  boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  

  # Each score represent how level of confidence for each of the objects.
  # Score is shown on the result image, together with the class label.
  scores = detection_graph.get_tensor_by_name('detection_scores:0')
  classes = detection_graph.get_tensor_by_name('detection_classes:0')
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')

  # Actual detection 

  # drawing all boxes defined by model here
  (boxes, scores, classes, num_detections) = sess.run(
      [boxes, scores, classes, num_detections],
      feed_dict={image_tensor: image_np_expanded})

  # # Visualization of the results # #
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category_index,
      use_normalized_coordinates=True,
      line_thickness=8)
  
  box_temp = ((boxes[0])[0])
  xmin = box_temp[1] * 640
  xmax = box_temp[3] * 640
  ymin = box_temp[0] * 480
  ymax = box_temp[2] * 480
  width = xmax-xmin
  height = ymax - ymin
  
  return (xmin,ymin,width,height)

# ----------------- # ------------------------------------ # ----------------- #




# -------- # Argument preparation, Multi-Tracker Creation # ------------------ #

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="csrt",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

tracker = OPENCV_OBJECT_TRACKERS[args['tracker']]()





# -------------------------- # Model preparation # --------------------------- #
# What model to download.
MODEL_NAME = 'current_PuckModel'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'puck_label.pbtxt') #changed to training, where the label map is

NUM_CLASSES = 1

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)







# ------------------------------ # Main # ------------------------------------ #


print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
tracker = cv2.TrackerCSRT_create()
detection_limit = 5

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      # read current frame from camera
      ret, image_np = cap.read()

      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)

      # object detection happens here,
      # box defined is the 1st element (highest scoring detection) 
      # This will have to change to be the bounding box for the broken PUCK
      xmin, ymin, width, height = object_detection(image_np_expanded,detection_limit)
      
      
#      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      # display output feed
      cv2.putText(image_np,'Detection Algorithm',(30,50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
      cv2.putText(image_np,'click q to enter Tracking Stage with detected ROI',(30,70),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
      cv2.putText(image_np,'or click s to manually select ROI',(30,90),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
      cv2.imshow('Output Feed', image_np)
      # optional delay for debugging purposes
      time.sleep(0)
      
      # button press to exit detection stage and enter tracking stage
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
        manualROI = 0 
        break
        #cv2.destroyAllWindows()
      elif key == ord('s'):
        manualROI = 1
        break
      
    
       
    # now begin to track the defined bounding box
    first_run = 1
    initBB = None
    while True:
      # read current frame from camera
      ret, image_np = cap.read()
      
      # check and run if there is an object to track (bounding box is initialised)
      if initBB is not None:
        # grab the updated bounding box for the current frame (tracking)
        (success, box) = tracker.update(image_np)
        
        # if tracking was successful, draw this bounding box on output feed
        if success:
          (x,y,w,h) = [int(v) for v in box]
          cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),2)
      
      # displaying output feed   
      cv2.putText(image_np,'Tracking Algorithm, click q to exit',(30,50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
      cv2.imshow('Output Feed',image_np)
      
      # this is only run for the first detected bounding box
      if first_run == 1:
        # option to manually select ROI if detection isnt working (backup)
        if manualROI == 1:
          initBB = cv2.selectROI('Output Feed',image_np,fromCenter=False,showCrosshair=True)
        else:
        # initialising bounding box with detected box
          initBB = (xmin,ymin,width,height)
          
        tracker.init(image_np,initBB)
        first_run = 0
#        print(initBB)
      
      # exit this loop under 'q' button press
      key= cv2.waitKey(1) & 0xFF
      if key == ord('q'):
        break

cv2.destroyAllWindows()
