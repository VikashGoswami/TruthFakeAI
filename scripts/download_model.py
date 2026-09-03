import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "hamzab/roberta-fake-news-classification"
save_path = "models/"

print(f"Downloading tokenizer and model from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print(f"Saving to {save_path}...")
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print("Download and save complete!")
