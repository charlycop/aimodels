from transformers import MT5Tokenizer, MT5ForConditionalGeneration

model_location = "../../Models/mt5-small"

# Load the tokenizer
tokenizer = MT5Tokenizer.from_pretrained(model_location)

# Load the model
model = MT5ForConditionalGeneration.from_pretrained(model_location)

# Define the input text and target language
input_text = "Kjo është një fjali shembull në shqip."
target_language = "en"  # English

# Map the target language to the corresponding language ID
language_id_mapping = {
    "en": 0,  # English
    "fr": 1,  # French
    "es": 2,  # Spanish
    "it": 3,  # Italian
    "pt": 4,  # Portuguese
    "de": 5,  # German
    "nl": 6,  # Dutch
    "ru": 7,  # Russian
    "zh": 8,  # Chinese
    "ja": 9,  # Japanese
    "ko": 10,  # Korean
    "ar": 11,  # Arabic
    "hi": 12,  # Hindi
    # ... add more languages if needed
}
target_language_id = language_id_mapping[target_language]

# Encode the input text
encoded_input = tokenizer(input_text, return_tensors="pt")

# Generate the translation with forced_bos_token_id
output = model.generate(
    **encoded_input,
    forced_bos_token_id=target_language_id
)

# Decode the translation
translated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(translated_text)
