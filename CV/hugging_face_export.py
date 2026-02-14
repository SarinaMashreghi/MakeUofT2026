# Use a pipeline as a high-level helper
import os
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForImageClassification


model_id = "dima806/facial_emotions_image_detection"
save_dir = "./models/hugging_face_vit"
image_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
model = AutoModelForImageClassification.from_pretrained(model_id)

pipe = pipeline(
    "image-classification",
    model=model,
    image_processor=image_processor,
)
pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png")

os.makedirs(save_dir, exist_ok=True)
image_processor.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print("Saved model + processor to", save_dir)

# Optional: verify local loading works
local_pipe = pipeline("image-classification", model=save_dir, image_processor=save_dir)
local_pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png")
