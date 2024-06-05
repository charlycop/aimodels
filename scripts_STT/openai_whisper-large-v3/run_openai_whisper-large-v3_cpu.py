import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import sys
import numpy as np
sys.path.append("../../src_utils/")
import our_utils

# Define the local and Hugging Face model locations
local_model_location = "../../Models/wav2vec2-large-960h"
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

    # Set device and data type
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Model ID for Whisper
    model_id = "openai/whisper-large-v3"

    # Load the model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # Initialize the pipeline with French language
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device
    )

    # Load the Albanian dataset
    dataset = load_dataset("eugenetanjc/speech_accent_albanian_test", split="train")

    # Process the first sample in the dataset
    sample = dataset[0]

    # Ensure the audio data is in the correct format (numpy array)
    audio_data = sample["audio"]

    # Convert the audio data to a numpy array if it's not already
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data)

    # Run the ASR pipeline on the sample
    result = pipe(audio_data)

    # Print the transcription result in French
    print(result["text"])