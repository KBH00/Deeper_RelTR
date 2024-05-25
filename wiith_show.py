# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse
import cv2
from deepface import DeepFace
from class_processing.classes import * 
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from second_classification.resnet import NonCommon_processing_Resnet
from second_classification.person_processing import person_processing
from models import build_model

import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
import torch

def tensor_to_tuple(tensor_box):
    """Convert a tensor or tuple of tensors to a flattened tuple of integers."""
    if isinstance(tensor_box, torch.Tensor):
        # If it's a single tensor, check if it needs to be itemized or converted to a list.
        return tuple(map(int, tensor_box.tolist())) if tensor_box.numel() > 1 else (int(tensor_box.item()),)
    elif isinstance(tensor_box, tuple):
        # Recursively flatten and convert elements, avoid creating nested tuples.
        return tuple(sub_item for item in tensor_box for sub_item in (tensor_to_tuple(item) if isinstance(item, (tuple, torch.Tensor)) else (item,)))
    elif isinstance(tensor_box, float):
        # Directly convert float to tuple.
        return (int(tensor_box),)
    return tensor_box


def parse_boxes_relations(data_dict):
    boxes = {}
    relations = {}
    for key, value in data_dict.items():
        if isinstance(key, tuple) and all(isinstance(item, (torch.Tensor, tuple)) for item in key):
            sub, obj = map(tensor_to_tuple, key)
            relations[(sub, obj)] = value
        else:
            box = tensor_to_tuple(key)
            boxes[box] = value
    return boxes, relations

def create_iou(box1, box2):
    """Calculate the Intersection over Union of two bounding boxes given as flat tuples."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection_area / (box1_area + box2_area - intersection_area)

def run_image_box_selector(data_dict):
    root = tk.Tk()
    root.title("Image Box Selector")

    image = Image.open("demo/3.jpg")
    photo = ImageTk.PhotoImage(image)
    canvas = Canvas(root, width=image.width, height=image.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=photo)

    boxes, relations = parse_boxes_relations(data_dict)

    # Draw predefined boxes
    for box in boxes:
        canvas.create_rectangle(box, outline='green', width=2, tags="predefined")

    rect = None
    start_x, start_y = 0, 0

    def on_button_press(event):
        nonlocal start_x, start_y, rect
        start_x, start_y = event.x, event.y
        rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=2)

    def on_move(event):
        nonlocal rect
        canvas.coords(rect, start_x, start_y, event.x, event.y)

    def on_button_release(event):
        nonlocal rect
        end_x, end_y = event.x, event.y
        canvas.coords(rect, start_x, start_y, end_x, end_y)
        user_box = (start_x, start_y, end_x, end_y)

        # Calculate IoU with all predefined boxes and find the best match
        highest_iou = 0
        best_match = None
        for box in boxes:
            iou = create_iou(user_box, box)
            if iou > highest_iou:
                highest_iou = iou
                best_match = box

        if best_match:
            print("Best match box:", best_match, "with IoU of", highest_iou)
            print("Class:", boxes[best_match])
            # Highlight related boxes
            for (sub, obj), relation in relations.items():
                if best_match in (sub, obj):
                    other_box = sub if best_match == obj else obj
                    canvas.create_rectangle(other_box, outline='blue', width=2)
                    print("Relation:", relation, "with box:", boxes[other_box])

    canvas.bind("<Button-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_button_release)

    root.mainloop()

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
    """Check for existing bounding boxes and return the class if found."""
    for box, class_label in explored_boxes.items():
        if calculate_iou(box, new_box) > threshold:
            return class_label
    return None

def class_post_processing(probas_sub, probas_obj, probas, keep_queries, sub_bboxes_scaled, obj_bboxes_scaled, indices, im):
    graph_triplet = []
    triplet_dict = {}
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
        #print(vg_sub, REL_CLASSES[probas[idx].argmax()], vg_obj)
        triplet_dict[sub_tuple] = vg_sub
        triplet_dict[obj_tuple] = vg_obj
        triplet_dict[((sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax))] = REL_CLASSES[probas[idx].argmax()]
    return triplet_dict

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default='demo/3.jpg',
                        help="Path of the test image")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='C:/Users/kbh/Code/sgg/RelTR/checkpoint0149.pth', help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class") 


    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser


def main(args):

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    # VG classes
    #CLASSES[probas_sub[idx].argmax()]
    #REL_CLASSES[probas[idx].argmax()]
    #CLASSES[probas_obj[idx].argmax()]

    model, _, _ = build_model(args)
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['model'])
    model.eval()

    img_path = args.img_path
    im = Image.open(img_path)
    if im.mode != 'RGB':
        im = im.convert('RGB')

    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.+ confidence
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                            probas_obj.max(-1).values > 0.3))

    # convert boxes from [0; 1] to image scales
    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

    topk = 10
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
    keep_queries = keep_queries[indices]

    # use lists to store the outputs via up-values
    conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
            lambda self, input, output: dec_attn_weights_sub.append(output[1])
        ),
        model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
            lambda self, input, output: dec_attn_weights_obj.append(output[1])
        )
    ]
    with torch.no_grad():
        # propagate through the model
        outputs = model(img)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]
        dec_attn_weights_sub = dec_attn_weights_sub[0]
        dec_attn_weights_obj = dec_attn_weights_obj[0]

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]
        im_w, im_h = im.size

        triplet_dict = class_post_processing(probas_sub=probas_sub, probas_obj=probas_obj, probas=probas, keep_queries=keep_queries,
                              sub_bboxes_scaled=sub_bboxes_scaled, obj_bboxes_scaled=obj_bboxes_scaled, indices=indices, im=im)
        run_image_box_selector(triplet_dict)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
