from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image

image = Image.open('path_to_your_document_image').convert('RGB')

processor = AutoImageProcessor.from_pretrained("microsoft/dit-large-finetuned-rvlcdip")
model = AutoModelForImageClassification.from_pretrained("microsoft/dit-large-finetuned-rvlcdip")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 16 RVL-CDIP classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
