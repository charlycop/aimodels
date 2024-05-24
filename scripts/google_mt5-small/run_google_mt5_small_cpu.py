from transformers import MT5ForConditionalGeneration, MT5Tokenizer

model_location = "../../Models/mt5-small"

# Charger le modèle et le tokenizer
model = MT5ForConditionalGeneration.from_pretrained(model_location)
tokenizer = MT5Tokenizer.from_pretrained(model_location)

# Phrase à traduire
source_text = "This is a sample sentence to translate."

# Préparer l'entrée pour le modèle
input_text = f"translate English to French: {source_text}"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Générer la traduction
output_ids = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)

# Décoder la sortie du modèle
translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(f"Texte source (en): {source_text}")
print(f"Traduction (fr): {translated_text}")