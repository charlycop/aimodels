from TTS.api import TTS

remote_unknown_location = "tts_models/en/ljspeech/tacotron2-DDC"

# Initialize TTS with a model from Hugging Face
tts = TTS(model_name=remote_unknown_location, progress_bar=True, gpu=False)

# Text to be converted to speech
text = "Hello, how are you today?"

# Generate speech and save it to a file
tts.tts_to_file(text=text, file_path="output.wav")