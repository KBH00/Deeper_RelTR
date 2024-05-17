from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import requests

def NonCommon_processing_ViT(box, original_image):
    model_name = 'google/vit-base-patch16-224'
    model = ViTForImageClassification.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    x_min, y_min, x_max, y_max, _, _ = box.cpu().numpy()
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max))

    if cropped_image.mode != 'RGB':
        cropped_image = cropped_image.convert('RGB')

    # if original_image.startswith('http'):
    #     image = Image.open(requests.get(original_image, stream=True).raw)
    # else:
    #     image = Image.open(original_image)

    inputs = feature_extractor(images=cropped_image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label.get(predicted_class_idx, f"Label index {predicted_class_idx}")

    return predicted_class

# image_path = 'C:/Users/kbh/img/1.jpg'  
# predicted_label = classify_image_with_vit(image_path)
# print(f'Predicted label: {predicted_label}')
