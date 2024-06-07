import torch
import torch.onnx
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the pre-trained model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
model.eval()

# Define input dimensions
batch_size = 1
max_length = 200  # Maximum sequence length
dummy_input = tokenizer("print('Hello, World!')", return_tensors="pt", padding=True, truncation=True, max_length=max_length).input_ids

# Export the model to ONNX format
torch.onnx.export(model,
                  dummy_input,
                  "nllb-200.onnx",
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input_ids', 'attention_mask'],
                  output_names=['output'],
                  dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                                'output': {0: 'batch_size', 1: 'sequence_length'}})