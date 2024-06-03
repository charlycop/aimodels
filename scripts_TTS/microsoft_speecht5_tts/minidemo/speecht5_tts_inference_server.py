from flask import Flask, request, jsonify, make_response, send_file
from flask_cors import CORS
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

app = Flask(__name__)
CORS(app)

# forcing CPU computing
device = "cpu"
print(f"Using device: {device}")

# Define the local and Hugging Face model locations
local_model_location = "../../Models/speecht5_tts"
huggingface_model = "microsoft/speecht5_tts"

# Load the model and tokenizer
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)

@app.route('/')
def index():
    response = make_response("Server is alive", 200)
    return response

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    inputs = processor(text=data['input'], return_tensors="pt")
    inputs = inputs.to(device)  # Move inputs to the device

    try:
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        sf.write("speech_cpu.wav", speech.cpu().numpy(), samplerate=16000)
        return send_file("speech_cpu.wav", mimetype="audio/wav", as_attachment=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        return make_response("Error during the inference process", 500)

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
