from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


import torch

# Define the local and Hugging Face model locations
local_model_location = "../../../Models/nllb-200-distilled-600M"
huggingface_model = "facebook/nllb-200-distilled-600M"

actual_model = huggingface_model
device = torch.device("cpu")

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(actual_model)
model = AutoModelForSeq2SeqLM.from_pretrained(actual_model)
model.to(device)
model = pipeline('translation', tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang='fra_Latn', max_length = 200, model=model)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Obtenir les données d'entrée depuis la requête
    input_data = request.json['input']

    # Generate text with the model
    output = model(input_data, max_length=50, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)[0]['translation_text']

    # Convertir la sortie en format JSON
    output_json = output.tolist()

    return jsonify(output_json)

if __name__ == '__main__':
    app.run(host='localhost', port=5000)