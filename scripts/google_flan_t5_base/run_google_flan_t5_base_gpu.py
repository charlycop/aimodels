from transformers import T5Tokenizer, T5ForConditionalGeneration 
import sys

sys.path.append("../")
import our_utils

# Define the local and Hugging Face model locations
local_model_location = "../../Models/flan_t5_base"
huggingface_model = "google/flan-t5-base"

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

    tokenizer = T5Tokenizer.from_pretrained(actual_model) 
    model = T5ForConditionalGeneration.from_pretrained(actual_model).to(device)

    input_text = "translate English to German: How old are you?" 
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    outputs = model.generate(input_ids) 

    print(tokenizer.decode(outputs[0])) 