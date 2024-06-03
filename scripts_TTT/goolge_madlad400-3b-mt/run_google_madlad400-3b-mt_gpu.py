from transformers import T5ForConditionalGeneration, T5Tokenizer

import sys

sys.path.append("../../src_utils/")
import our_utils

# Define the local and Hugging Face model locations
local_model_location = "../../Models/madlad400-3b-mt"
huggingface_model = "google/madlad400-3b-mt"
gpu_type = our_utils.find_gpu_type()

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

    model = T5ForConditionalGeneration.from_pretrained(actual_model, device_map=gpu_type)
    tokenizer = T5Tokenizer.from_pretrained(actual_model)
    
    # Define the input text
    input_text = "This is a sample English sentence."

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids=input_ids)

    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(translated_text)
