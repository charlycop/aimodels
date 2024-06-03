
from TTS.api import TTS
import sys
sys.path.append("../../src_utils/")
import our_utils

# forcing CPU computing
device = "cpu"
print(f"Using device: {device}")

# Define the local and Hugging Face model locations
local_model_location = "../../Models/tomodify"
huggingface_model = "tts_models/en/ljspeech/tacotron2-DDC"

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


    # Initialize TTS with a model from Hugging Face
    tts = TTS(model_name=actual_model, progress_bar=True, gpu=False)

    # Text to be converted to speech
    text = "Hello, how are you today?"

    # Generate speech and save it to a file
    tts.tts_to_file(text=text, file_path="output.wav")