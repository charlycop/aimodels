from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import sys
sys.path.append("../../src_utils/")
import our_utils

# Check if CUDA is available
device = our_utils.find_gpu_type()
print(f"Using device: {device}")

# Define the local and Hugging Face model locations
local_model_location = "../../Models/speecht5_tts"
huggingface_model = "microsoft/speecht5_tts"

# Create a loop that displays the menu and prompts the user for their choice
while True:
    # Display the menu and prompt the user for their choice
    user_choice = our_utils.display_menu()

    # Perform the appropriate action based on the user's choice
    if user_choice == "1":
        # Load the local model and perform the translation
        actual_model = local_model_location
    elif user_choice == "2":
        # Load the Hugging Face model and perform the translation
        actual_model = huggingface_model
    elif user_choice == "3":
        our_utils.delete_cache()
        continue
    elif user_choice == "4":
        # Exit the program
        print("\nGoodbye!")
        exit(1)
    else:
        # Display an error message if the user's choice is invalid
        print("\nInvalid choice. Please try again.")

    processor = SpeechT5Processor.from_pretrained(actual_model)
    model = SpeechT5ForTextToSpeech.from_pretrained(actual_model).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    inputs = processor(text="Charly is so handsome, incredible!", return_tensors="pt")
    inputs = inputs.to(device)  # Move inputs to the device

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sf.write("speech_gpu.wav", speech.cpu().numpy(), samplerate=16000)