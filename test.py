import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
import torch

def tensor_to_tuple(tensor_box):
    if isinstance(tensor_box, torch.Tensor):
        return tuple(map(int, tensor_box.tolist()))
    elif isinstance(tensor_box, tuple):
        return tuple(tensor_to_tuple(item) for item in tensor_box)
    return tensor_box

def parse_boxes_relations(data_dict):
    boxes = {}
    relations = {}
    for key, value in data_dict.items():
        if isinstance(key, tuple) and all(isinstance(item, tuple) for item in key):
            # It's a relationship entry
            sub, obj = map(tensor_to_tuple, key)
            relations[(sub, obj)] = value
        else:
            # It's a single box entry
            box = tensor_to_tuple(key)
            boxes[box] = value
    return boxes, relations

def create_iou(box1, box2):
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
    boxes, relations = parse_boxes_relations(data_dict)
    root = tk.Tk()
    root.title("Image Box Selector")

    image = Image.open("path_to_your_image.jpg")
    photo = ImageTk.PhotoImage(image)
    canvas = Canvas(root, width=image.width, height=image.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=photo)

    def on_click(event):
        user_box = (event.x - 50, event.y - 50, event.x + 50, event.y + 50)
        canvas.create_rectangle(user_box, outline='red', width=2)

        highest_iou = 0
        best_match = None
        for box, description in boxes.items():
            iou = create_iou(user_box, box)
            if iou > highest_iou:
                highest_iou = iou
                best_match = box

        if best_match:
            canvas.create_rectangle(best_match, outline='blue', width=2)
            print("Best match:", boxes[best_match])

            # Display related boxes
            for (sub, obj), rel in relations.items():
                if best_match in (sub, obj):
                    other_box = sub if best_match == obj else obj
                    canvas.create_rectangle(other_box, outline='green', width=2)
                    print("Related box:", boxes[other_box], "Relation:", rel)

    canvas.bind("<Button-1>", on_click)
    root.mainloop()

data_dict = {
    (torch.tensor(248.9680), torch.tensor(78.4099), torch.tensor(1208.9292), torch.tensor(1296.0344)): '22 year old happy latino hispanic man',
    (torch.tensor(507.7213), torch.tensor(73.7445), torch.tensor(878.5533), torch.tensor(344.2323)): 'hair',
}

if __name__ == '__main__':
    run_image_box_selector(data_dict)
