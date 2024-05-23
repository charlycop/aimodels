from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Charger le modèle et le tokenizer
model_location = "../../Models/mt5-small"

tokenizer = MT5Tokenizer.from_pretrained(model_location)
model = MT5ForConditionalGeneration.from_pretrained(model_location)

# Texte à traduire
source_language = "en"  # English
target_language = "fr"  # French
source_text = "This is a test."
input_text = f"translate {target_language} {source_text}"

# Tokeniser le texte
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Générer la traduction
outputs = model.generate(input_ids, max_length=100)

# Décoder et afficher la traduction
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
translated_text = translated_text.replace("<extra_id_0>", "").strip()
print("Traduction:", translated_text)




# # Texte à traduire
# source_text = "translate English to French: This is a test."

# # Tokeniser le texte
# input_ids = tokenizer.encode(source_text, return_tensors="pt")

# # Générer la traduction
# outputs = model.generate(input_ids, max_length=100)

# # Décoder et afficher la traduction
# translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("Traduction:", translated_text)