from transformers import T5Tokenizer, T5ForConditionalGeneration 

model_location = "../../Models/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(model_location) 
model = T5ForConditionalGeneration.from_pretrained(model_location) 

input_text = "translate English to German: How old are you?" 
input_ids = tokenizer(input_text, return_tensors="pt").input_ids 

outputs = model.generate(input_ids) 

print(tokenizer.decode(outputs[0])) 