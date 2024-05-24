from transformers import MarianTokenizer, MarianMTModel
import os
import shutil
import torch

# Define the local and Hugging Face model locations
local_model_location = "../../Models/opus-mt-en-sq"
huggingface_model = "Helsinki-NLP/opus-mt-en-sq"
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


# Define the menu options
menu_options = {
    1: "Use local model",
    2: "Use Hugging Face model",
    3: "Delete the huggingface cache folder",
    4: "Exit"
}

# Create a loop that displays the menu and prompts the user for their choice
while True:
    # Display the menu options
    print("\nTranslation Menu")
    for option, description in menu_options.items():
        print(f"{option}. {description}")

    # Prompt the user for their choice
    user_choice = input("\nEnter your choice: ")

    # Perform the appropriate action based on the user's choice
    if user_choice == "1":
        # Load the local model and perform the translation
        tokenizer = MarianTokenizer.from_pretrained(local_model_location)
        model = MarianMTModel.from_pretrained(local_model_location).to("cuda")

    elif user_choice == "2":
        # Load the Hugging Face model and perform the translation
        tokenizer = MarianTokenizer.from_pretrained(huggingface_model)
        model = MarianMTModel.from_pretrained(huggingface_model).to("cuda")

    elif user_choice == "3":
        shutil.rmtree(cache_dir)
        continue

    elif user_choice == "4":
        # Exit the program
        print("\nGoodbye!")
        break

    else:
        # Display an error message if the user's choice is invalid
        print("\nInvalid choice. Please try again.")

    # Define the input text
    input_text = "This is a sample English sentence."

    # Encode the input text
    encoded_input = tokenizer(input_text, return_tensors="pt")

    # Generate the translation
    output = model.generate(**encoded_input)

    # Decode the translation
    translated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(translated_text)
