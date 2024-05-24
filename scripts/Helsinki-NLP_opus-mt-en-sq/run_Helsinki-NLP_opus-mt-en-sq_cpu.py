from transformers import MarianTokenizer, MarianMTModel


model_location = "../../Models/opus-mt-en-sq"

# Load the tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_location)

# Load the model
model = MarianMTModel.from_pretrained(model_location)

# Define the input text
input_text = "This is a sample English sentence."

# Encode the input text
encoded_input = tokenizer(input_text, return_tensors="pt")

# Generate the translation
output = model.generate(**encoded_input)

# Decode the translation
translated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(translated_text)