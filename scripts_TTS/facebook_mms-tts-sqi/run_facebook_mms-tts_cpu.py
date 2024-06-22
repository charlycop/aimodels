from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
import sys
sys.path.append("../../src_utils/")
import our_utils

# Define the local and Hugging Face model locations
local_model_location = "../../Models/facebook/mms-tts-sqi"
huggingface_model = "facebook/mms-tts-sqi"

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

    # Load the model and tokenizer
    model = VitsModel.from_pretrained(actual_model)
    tokenizer = AutoTokenizer.from_pretrained(actual_model)

    # Define the text to be converted to speech
    text = "Në një fshat të vogël në zemër të Shqipërisë, jetonte një burrë i quajtur Arben, i cili ishte një fermer i përkushtuar dhe i ndershëm, i njohur nga të gjithë për punën e tij të palodhur dhe përkushtimin ndaj familjes së tij. Çdo mëngjes, ai zgjohej herët për të ushqyer lopët, delet dhe pulat, dhe më pas kalonte orë të tëra duke punuar në fusha, ndërsa gjatë verës korrte grurin dhe misrin, dhe në dimër kujdesej për kafshët që të ishin të ngrohta dhe të ushqyera mirë."

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Generate the speech waveform using the model
    with torch.no_grad():
        output = model(**inputs).waveform

    # Save the generated speech waveform to a file
    sf.write("output.wav", output.squeeze().cpu().numpy(), samplerate=22050)
