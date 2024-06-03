from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf
from scipy.signal import resample
import sys
sys.path.append("../../src_utils/")
import our_utils

# forcing CPU computing
device = "cpu"
print(f"Using device: {device}")

# Define the local and Hugging Face model locations
local_model_location = "../../Models/tomodify"
huggingface_model = "facebook/wav2vec2-large-960h"

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
        break
    else:
        # Display an error message if the user's choice is invalid
        print("\nInvalid choice. Please try again.")




    # Load pre-trained model and processor
    processor = Wav2Vec2Processor.from_pretrained(actual_model)
    model = Wav2Vec2ForCTC.from_pretrained(actual_model)

    # Load audio file
    audio_input, sample_rate = sf.read("../tacotron2_tts/output.wav")

    # Resample the audio to 16000 Hz if it's not already at that rate
    if sample_rate != 16000:
        number_of_samples = round(len(audio_input) * float(16000) / sample_rate)
        audio_input = resample(audio_input, number_of_samples)
        sample_rate = 16000

    # Preprocess the audio file
    input_values = processor(audio_input, return_tensors="pt", sampling_rate=sample_rate).input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Retrieve the predicted ids
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the ids to text
    transcription = processor.batch_decode(predicted_ids)[0]

    # Print the transcription
    print("Transcription:", transcription)
