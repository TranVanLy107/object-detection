
# import the necessary packages
from LyTran.centroidtracker import CentroidTracker
from LyTran.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import os
import cv2
import tensorflow as tf
import sys
import glob

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
#from object_detection.utils import ops as utils_ops
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.8,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
args = vars(ap.parse_args())
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
#from utils.speed_and_direction_prediction_module import speed_prediction1
# from object_detection.utils import label_map_util

# from object_detection.utils import visualization_utils as vis_util

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
#VIDEO_NAME = 'FCAM34.avi'

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.

PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
video = os.path.join(CWD_PATH,'video_counting')
for folder in glob.glob(video + '/*.avi'):
    PATH_TO_VIDEO = folder

#ct = CentroidTracker.CentroidTracker()
countedObjectID = []
# Number of classes the object detector can identify
NUM_CLASSES = 5
ROI_POSITION = 200
bottom_position_of_detected_vehicle = [0]
current_frame_number_list = [0]
# Draw line:
def draw_line(image):
    im_height, im_width = image.shape[:2]
    center = (im_width / 2, im_height / 2)
    #cv2.rectangle(image, (20, int(center[1])), (im_width - 150, int(center[1] + 1)), [0,0,255], 2)
    #cv2.line(frame, (0, im_height // 2), (im_width, im_height // 2), (0, 0, 255), 2)
    return center
# Load the label map.
# Label maps map indices to category names, so that when our convolution
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Helper function
def load_image_into_numpy_array(image):
  return np.nsarray(image).astype(np.uint8)

def put_number_vehicles(image, count):
    text = "Number of vehicles: {}".format(count)
    cv2.putText(image, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)

# initialize the video writer (we'll instantiate later if need be)
writer = None
message = 'STOPPING!'
# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
count = 0

# start the frames per second throughput estimator
fps = FPS().start()

while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)
    # resize the frame to have a maximum width of 500 pixels (the
	# less data we have, the faster we can process it), then convert
	# the frame from BGR to RGB for dlib
    frame = imutils.resize(frame, width=600)
   
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	
	# if the frame dimensions are empty, set them
    if W is None or H is None:
	    (H, W) = frame.shape[:2]
		
	# initialize the current status along with our list of bounding
	# box rectangles returned by either (1) our object detector or
	# (2) the correlation trackers
    status = "Waiting"
    rects = []

	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
    if totalFrames % args["skip_frames"] == 0:
	# set the status and initialize our new set of object trackers
        status = "Tracking and Counting"
        trackers = []

    # convert the frame to a blob and pass the blob through the
	# network and obtain the detections
        (h, w) = frame.shape[:2]
        
	# Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})    
            
    #Draw the results of the detection (aka 'visulaize the results')
        image, box_to_color_map = vis_util.visualize_boxes_and_labels_on_image_array(
            #video.get(1),
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh=0.50)
		
        # loop over the detections	    
       
        center_image = draw_line(frame)
        for box, color in box_to_color_map.items():
            ymin, xmin, ymax, xmax = box
            im_height, im_width = frame.shape[:2]
            (startX, endX, startY, endY) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
			
            # construct a dlib rectangle object from the bounding
            # box coordinates and then start the dlib correlation
	        # tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
            tracker.start_track(rgb, rect)
			
			#add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))
			# add the tracker to our list of trackers so we can
			# utilize it during skip frames
            trackers.append(tracker)
            
           
    # otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing throughput
    else:
	    # loop over the trackers
        for tracker in trackers:
		    # set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
	        status = "Tracking"

			# update the tracker and grab the updated position
	        tracker.update(rgb)
	        pos = tracker.get_position()

			# unpack the position object
	        startX = int(pos.left())
	        startY = int(pos.top())
	        endX = int(pos.right())
	        endY = int(pos.bottom())
            

			#add the bounding box coordinates to the rectangles list
	        rects.append((startX, startY, endX, endY))
            
            
	# draw a horizontal line in the center of the frame -- once an
	# object crosses this line we will determine whether they were
	# moving 'up' or 'down'
    #cv2.line(frame, (w //2, 0), (w //2, h), (0, 255, 255), 2)
    

	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
   
    objects = ct.update(rects)
	
    
	# loop over the tracked objects
    for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
        to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
        else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
			
			# check to see if the object has been counted or not
            if not to.counted:
            #if objectID not in countedObjectID:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
                if direction < 0 and abs(center_image[1] - centroid[1]) <= 20:
                    totalUp += 1
                    to.counted = True
                    
				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
                elif direction > 0 and abs(center_image[1] - centroid[1]) <= 20:
                    totalDown += 1
                    to.counted = True
					
			          
		# store the trackable object in our dictionary
        trackableObjects[objectID] = to
        
        #put_number_vehicles(frame, count)
		#draw both the ID of the object and the centroid of the
		#object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
	#construct a tuple of information we will be displaying on the
	#frame
    info = [
        ("Vehicles detected: ", totalUp + totalDown),
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status),
    ]
	
   
	#loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, h - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)	
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
	
	# increment the total number of frames processed thus far and
	# then update the FPS counter
    #totalFrames += 1
    fps.update()
# stop the timer and display FPS information
fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# Clean up
video.release()
cv2.destroyAllWindows()
