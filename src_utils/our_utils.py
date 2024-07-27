# our_utils.py

import os
import shutil
import torch

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

# Return the GPU the system is using
def find_gpu_type():
    # Check for ROCm support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else :
        print("No GPU !!")
        exit(1)
    
    print(f"Using device: {device}")
    return device
