from transformers import MarianTokenizer, MarianMTModel
import torch

model_location = "../../Models/opus-mt-sq-en"

# Load the tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_location)

# Load the model and move it to the GPU
model = MarianMTModel.from_pretrained(model_location).to("cuda")

# Define the input text
input_text = "Kjo është një fjali shembull në shqip."

# Encode the input text
encoded_input = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate the translation
output = model.generate(**encoded_input)

# Decode the translation
translated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(translated_text)