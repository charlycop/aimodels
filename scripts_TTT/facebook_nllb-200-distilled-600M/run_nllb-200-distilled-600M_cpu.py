from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import torch
import sys

sys.path.append("../../src_utils/")
import our_utils

# Define the local and Hugging Face model locations
local_model_location = "../../Models/nllb-200-distilled-600M"
huggingface_model = "facebook/nllb-200-distilled-600M"

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
    tokenizer = AutoTokenizer.from_pretrained(actual_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(actual_model)

    # Move the model to GPU
    device = torch.device("cpu")

    model.to(device)
    model = pipeline('translation', tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang='fra_Latn', max_length = 200, model=model)

    text="My name is Charly."

    # Generate text with the model
    output = model(text, max_length=50, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)[0]['translation_text']

    # Print the output
    print(f"Input: {text}")
    print(f"Output: {output}")
