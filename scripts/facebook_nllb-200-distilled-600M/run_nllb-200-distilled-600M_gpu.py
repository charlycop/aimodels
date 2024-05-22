from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

model_location = "../../Models/nllb-200-distilled-600M"
# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_location)
model = AutoModelForSeq2SeqLM.from_pretrained(model_location)

# Move the model to GPU
device = torch.device("cuda")

model.to(device)
model = pipeline('translation', tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang='fra_Latn', max_length = 200, model=model)

text="My name is Charly."

# Generate text with the model
output = model(text, max_length=50, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)[0]['translation_text']

# Print the output
print(f"Input: {text}")
print(f"Output: {output}")
