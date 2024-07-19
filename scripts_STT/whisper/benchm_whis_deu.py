import whisper
from bert_score import score
import torch
import os

# Load Whisper model
# model_path= whisper.load_model ("/home/shared/Models/whisper-large-v3", device="cuda" if torch.cuda.is_available() else "cpu")
model = whisper.load_model("large-v3")

# Load the local weights
# model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))

# Path to your German audio file
audio_file = "/home/walter/benchmark/aimodels/scripts_STT/common_voice_de_40622399.mp3"

# Reference transcript (the correct German transcript of the audio)
reference_transcript = "Dieser ist aus Liebe zu ihr bereit, sich für sie töten zu lassen."

# Transcribe the audio
result = model.transcribe(audio_file)
transcription = result["text"]

print(f"Whisper transcription: {transcription}")
print(f"Reference transcript: {reference_transcript}")

# Calculate BERTScore
P, R, F1 = score([transcription], [reference_transcript], lang="de", verbose=True)

print(f"BERTScore Precision: {P.item():.4f}")
print(f"BERTScore Recall: {R.item():.4f}")
print(f"BERTScore F1: {F1.item():.4f}")