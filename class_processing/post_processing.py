from class_processing.classes import * 
from second_classification.resnet import NonCommon_processing_Resnet
from second_classification.vit import NonCommon_processing_ViT
from second_classification.person_processing import person_processing
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from second_classification.blip import open_vocabulary_classification_blip

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    x_left = max(x1, x1_p)
    y_top = max(y1, y1_p)
    x_right = min(x2, x2_p)
    y_bottom = min(y2, y2_p)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def get_processed_class(explored_boxes, new_box, threshold=0.8):
    """Check for existing bounding boxes and return the class if found, which is for efficiency."""
    for box, class_label in explored_boxes.items():
        if calculate_iou(box, new_box) > threshold:
            return class_label
    return None

def class_post_processing(probas_sub, probas_obj, probas, keep_queries, sub_bboxes_scaled, obj_bboxes_scaled, indices, im):
    graph_triplet = []
    explored_boxes = {}
    for idx, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
        zip(keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
        vg_sub = CLASSES[probas_sub[idx].argmax()]
        vg_obj = CLASSES[probas_obj[idx].argmax()]
        sub_tuple = (sxmin.item(), symin.item(), sxmax.item(), symax.item())
        obj_tuple = (oxmin.item(), oymin.item(), oxmax.item(), oymax.item())
        
        processed_sub_class = get_processed_class(explored_boxes, sub_tuple)
        if processed_sub_class is None:
            if vg_sub in need_more_classes:
                if vg_sub in PERSON_CLASSES:
                    vg_sub = person_processing(x_min=sxmin, y_min=symin, x_max=sxmax, y_max=symax, image=im, i=idx) + vg_sub
                else:
                    vg_sub = NonCommon_processing_Resnet(x_min=sxmin, y_min=symin, x_max=sxmax, y_max=symax, image=im)
            explored_boxes[sub_tuple] = vg_sub
        else:
            vg_sub = processed_sub_class

        processed_obj_class = get_processed_class(explored_boxes, obj_tuple)
        if processed_obj_class is None:
            if vg_obj in need_more_classes:
                if vg_obj in PERSON_CLASSES:
                    vg_obj = person_processing(x_min=oxmin, y_min=oymin, x_max=oxmax, y_max=oymax, image=im, i=idx) + vg_obj
                else:
                    vg_obj = NonCommon_processing_Resnet(x_min=oxmin, y_min=oymin, x_max=oxmax, y_max=oymax, image=im)
            explored_boxes[obj_tuple] = vg_obj
        else:
            vg_obj = processed_obj_class

        graph_triplet.append([vg_sub, REL_CLASSES[probas[idx].argmax()], vg_obj])
        #print(vg_sub, REL_CLASSES[probas[idx].argmax()], vg_obj)
    return graph_triplet