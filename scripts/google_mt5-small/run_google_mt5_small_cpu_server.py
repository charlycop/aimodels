from flask import Flask, request, jsonify
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

model_location = "../../Models/mt5-small"

app = Flask(__name__)

# Load the tokenizer and model
tokenizer = MT5Tokenizer.from_pretrained(model_location)
model = MT5ForConditionalGeneration.from_pretrained(model_location)

@app.route('/translate', methods=['POST'])
def translate():
    input_text = request.json['text']
    target_language = request.json['targetLanguage']

    # Encode the input text with the target language
    encoded_input = tokenizer(input_text, return_tensors="pt")

    # Generate the translation
    output = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(target_language))

    # Decode the translation
    translated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({'translatedText': translated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)