import torch
from torchvision import transforms
from torchvision.models import resnet152, ResNet152_Weights


def NonCommon_processing_Resnet(x_min, y_min, x_max, y_max, image):
    #model = models.resnet50(pretrained=True)
    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    model.eval() 
    x_min = x_min.item()
    y_min = y_min.item()
    x_max = x_max.item()
    y_max = y_max.item()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    #x_min, y_min, x_max, y_max, _, _ = box.cpu().numpy()
    cropped_image = image.crop((x_min, y_min, x_max, y_max))

    if cropped_image.mode != 'RGB':
        cropped_image = cropped_image.convert('RGB')

    #image = Image.open(image_path)
    img_tensor = preprocess(cropped_image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
    
    _, predicted_class = outputs.max(1)
    predicted_class = predicted_class.item()

    with open("second_classification/imagenet_classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes[predicted_class]