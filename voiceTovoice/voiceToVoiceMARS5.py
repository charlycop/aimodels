import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, pipeline
import soundfile as sf
import librosa
import subprocess
import sys
import numpy as np

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

# Helper function to display a menu (mocked here for simplicity)
def display_menu():
    print("Menu:")
    print("1. Use local models")
    print("2. Use Hugging Face models")
    print("3. Delete cache")
    print("4. Exit")
    return input("Enter your choice: ")

# Set model locations
local_ttt_model_location = "../../Models/nllb-200-distilled-600M"
huggingface_ttt_model = "facebook/nllb-200-distilled-600M"

local_stt_model_location = "to be defined"
huggingface_stt_model = "openai/whisper-large-v2"

local_tts_model_location = "to be defined"
huggingface_tts_model = "Camb-ai/mars5-tts"

# Menu loop to choose models
while True:
    user_choice = display_menu()

    if user_choice == "1":
        ttt_model_location = local_ttt_model_location
        stt_model_location = local_stt_model_location
        tts_model_location = local_tts_model_location
    elif user_choice == "2":
        ttt_model_location = huggingface_ttt_model
        stt_model_location = huggingface_stt_model
        tts_model_location = huggingface_tts_model
    elif user_choice == "3":
        # Mocked cache deletion
        print("Cache deleted.")
        continue
    elif user_choice == "4":
        print("Goodbye!")
        break
    else:
        print("Invalid choice. Please try again.")
        continue

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
    ttt_pipeline = pipeline('translation', model=ttt_model, tokenizer=ttt_tokenizer, src_lang='eng_Latn', tgt_lang='fra_Latn')

    # Load the TTS model
    tts_model, tts_config_class = torch.hub.load('Camb-ai/mars5-tts', 'mars5_english', trust_repo=True)

    # Load a sample audio file (replace with your own audio source)
    subprocess.check_call(['wget', '-O', 'example.wav', 'https://github.com/Camb-ai/mars5-tts/raw/master/docs/assets/example_ref.wav'])
    wav, sr = librosa.load('./example.wav', sr=tts_model.sr, mono=True)
    wav = torch.from_numpy(wav)

    # Speech-to-Text
    stt_result = stt_pipeline(wav.numpy())
    stt_text = stt_result["text"]
    print(f"Transcribed text: {stt_text}")

    # Text-to-Text Translation
    ttt_translation = ttt_pipeline(stt_text, max_length=200, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)[0]['translation_text']
    print(f"Translated text: {ttt_translation}")

    # Text-to-Speech
    ref_transcript = "We actually haven't managed to meet demand."
    tts_cfg = tts_config_class(deep_clone=True, rep_penalty_window=100, top_k=100, temperature=0.7, freq_penalty=3)
    _, wav_out = tts_model.tts(ttt_translation, wav, ref_transcript, cfg=tts_cfg)

    # Save the output audio to a file
    sf.write('output.wav', wav_out.numpy(), tts_model.sr)
    print("Voice-to-voice translation completed. Output saved to 'output.wav'.")
