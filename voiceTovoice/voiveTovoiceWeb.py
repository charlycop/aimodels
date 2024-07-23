import os
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, pipeline, VitsModel
import soundfile as sf
import librosa
from flask_socketio import SocketIO, emit
import subprocess
import sys

# Ensure safetensors and wget are installed
try:
    import safetensors
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors"])
    import safetensors

try:
    subprocess.check_call(['wget', '--version'])
except FileNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wget"])
    import wget

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
OUTPUT_FILE = 'output.wav'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
socketio = SocketIO(app)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the file
            socketio.start_background_task(process_file, filepath)
            
            return redirect(url_for('progress'))
    return '''
    <!doctype html>
    <title>Upload a new File</title>
    <h1>Upload a new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/progress')
def progress():
    return '''
    <!doctype html>
    <title>Processing</title>
    <h1>Processing your file, please wait...</h1>
    <div id="progress">
        <p>Waiting to start...</p>
    </div>
    <script src="//cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.0/socket.io.min.js"></script>
    <script type="text/javascript" charset="utf-8">
        var socket = io();
        socket.on('progress', function(data) {
            document.getElementById('progress').innerHTML = '<p>' + data.message + '</p>';
        });
        socket.on('complete', function(data) {
            window.location.href = '/results';
        });
    </script>
    '''

@app.route('/results')
def results():
    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'results.txt'), 'r') as f:
        results = f.read().split('\n')
    stt_text = results[0]
    ttt_translation = results[1]
    return render_template_string('''
    <!doctype html>
    <title>Results</title>
    <h1>Results</h1>
    <p><strong>STT Result:</strong> {{ stt_text }}</p>
    <p><strong>TTT Result:</strong> {{ ttt_translation }}</p>
    <a href="{{ url_for('download_file', filename=output_file) }}">Download Output Audio</a>
    ''', stt_text=stt_text, ttt_translation=ttt_translation, output_file=OUTPUT_FILE)

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@socketio.on('connect')
def handle_connect():
    emit('progress', {'message': 'Connected'})

def process_file(filepath):
    stt_model_location = "openai/whisper-large-v2"
    ttt_model_location = "facebook/nllb-200-distilled-600M"
    tts_model_location = "facebook/mms-tts-swh"

    # Load the STT model and processor
    stt_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    stt_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(stt_model_location, torch_dtype=stt_torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    stt_model.to(stt_device)
    stt_processor = AutoProcessor.from_pretrained(stt_model_location)
    stt_pipeline = pipeline(
        "automatic-speech-recognition",
        model=stt_model,
        tokenizer=stt_processor.tokenizer,
        feature_extractor=stt_processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=stt_torch_dtype,
        device=stt_device
    )

    # Load the TTT model and tokenizer
    ttt_tokenizer = AutoTokenizer.from_pretrained(ttt_model_location)
    ttt_model = AutoModelForSeq2SeqLM.from_pretrained(ttt_model_location)
    ttt_pipeline = pipeline('translation', model=ttt_model, tokenizer=ttt_tokenizer, src_lang='eng_Latn', tgt_lang='swh_Latn')

    # Load the TTS model and tokenizer
    tts_model = VitsModel.from_pretrained(tts_model_location)
    tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_location)

    # Load the audio file
    wav, sr = librosa.load(filepath, sr=16000, mono=True)
    wav = torch.from_numpy(wav)

    # Speech-to-Text
    socketio.emit('progress', {'message': 'Performing STT...'})
    stt_result = stt_pipeline(wav.numpy())
    stt_text = stt_result["text"]
    socketio.emit('progress', {'message': f'STT completed: {stt_text}'})

    # Text-to-Text Translation
    socketio.emit('progress', {'message': 'Performing TTT...'})
    ttt_translation = ttt_pipeline(stt_text, max_length=200, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)[0]['translation_text']
    socketio.emit('progress', {'message': f'TTT completed: {ttt_translation}'})

    # Save STT and TTT results
    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'results.txt'), 'w') as f:
        f.write(f"{stt_text}\n{ttt_translation}")

    # Text-to-Speech
    socketio.emit('progress', {'message': 'Performing TTS...'})
    inputs = tts_tokenizer(ttt_translation, return_tensors="pt")
    with torch.no_grad():
        output = tts_model(**inputs).waveform

    # Save the output audio to a file
    output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], OUTPUT_FILE)
    sf.write(output_filepath, output.squeeze().cpu().numpy(), samplerate=22050)
    socketio.emit('progress', {'message': f'TTS completed. Output saved to {OUTPUT_FILE}'})

    socketio.emit('complete', {'message': 'Processing complete.'})

if __name__ == '__main__':
    socketio.run(app, debug=True)
