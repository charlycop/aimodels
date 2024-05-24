# translation_menu.py

import os
import shutil

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

# Define a function to display the menu and prompt the user for their choice
def display_menu():
    # Display the menu options
    print("\nTranslation Menu")
    for option, description in menu_options.items():
        print(f"{option}. {description}")

    # Prompt the user for their choice
    user_choice = input("\nEnter your choice: ")

    return user_choice

# Define a function to delete the Hugging Face cache directory
def delete_cache():
    # Delete the cache directory and all of its contents
    shutil.rmtree(cache_dir)

    # Print a message to confirm that the cache was deleted
    print("Hugging Face cache deleted.")
