from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch

model_location = "../../Models/mt5-small"

# Load the tokenizer
tokenizer = MT5Tokenizer.from_pretrained(model_location)

# Load the model and move it to the GPU
model = MT5ForConditionalGeneration.from_pretrained(model_location).to("cuda")

# Define the input text and target language
input_text = "Kjo është një fjali shembull në shqip."
target_language = "en"  # English

# Encode the input text with the target language
encoded_input = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate the translation
output = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(target_language))

# Decode the translation
translated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(translated_text)