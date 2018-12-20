def check_object_can_deregister(objects, center_image):
    center_above_y = center_image[1] - 150
    center_below_y = center_image[1] + 300
    will_remove_object = []
    for (objectID, centroid) in objects.items():
        if abs(center_above_y - centroid[1]) <= 10 or abs(center_below_y - centroid[1]) <= 10:
            will_remove_object.append(objectID)
    return will_remove_object

def check_can_count_object(object, center_image, counted_objectID):
    objectID, centroid = object
    if objectID not in counted_objectID:
        if abs(center_image[1] - centroid[1]) <= 20:
            return True
        else:
            return False
            