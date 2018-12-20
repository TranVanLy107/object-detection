import cv2

# Draw line:
def draw_line(image):
    im_height, im_width = image.shape[:2]
    center = (im_width/2, im_height/2)
    cv2.rectangle(image, (460, int(center[1])), (im_width - 250, int(center[1] + 1)), [0,0,255], 1)
    return center

def put_number_vehi1(image, count):
    text = "Container:{}, ".format(count)
    cv2.putText(image, text, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def put_number_vehi2(image, count):
    text = "Xe con:{}, ".format(count)
    cv2.putText(image, text, (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
	
def put_number_vehi3(image, count):
    text = "Xe khach:{}, ".format(count)
    cv2.putText(image, text, (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def put_number_vehi4(image, count):
    text = "Xe tai: {}, ".format(count)
    cv2.putText(image, text, (600, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
	
def put_objectID_into_object_stop(image, centroid, objectID):
    text = "{}".format(objectID)
    cv2.putText(image, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	
def put_objectID_into_object_slow(image, centroid, objectID):
    text = "{}".format(objectID)
    cv2.putText(image, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
