from transformers import MarianTokenizer, MarianMTModel
import os
import shutil
import sys

sys.path.append("../../src_utils/")
import our_utils

# Define the local and Hugging Face model locations
local_model_location = "../../Models/opus-mt-sq-en"
huggingface_model = "Helsinki-NLP/opus-mt-sq-en"

device = our_utils.find_gpu_type()

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

    tokenizer = MarianTokenizer.from_pretrained(actual_model)
    model = MarianMTModel.from_pretrained(actual_model).to(device)

    # Define the input text
    input_text = "Kjo është një fjali shembull në shqip."

    # Encode the input text
    encoded_input = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate the translation
    output = model.generate(**encoded_input)

    # Decode the translation
    translated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(translated_text)
