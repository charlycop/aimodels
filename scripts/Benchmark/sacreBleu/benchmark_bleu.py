import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Define paths to the extracted Europarl files
source_file = '../../../europarl-v7.fr-en.en'
target_file = '../../../europarl-v7.fr-en.fr'

# Read the data
with open(source_file, 'r', encoding='utf-8') as src, open(target_file, 'r', encoding='utf-8') as tgt:
    inputs = src.readlines()
    references = tgt.readlines()

# Limit the data size for testing to avoid memory issues
inputs = inputs[:1000]
references = [[ref.strip()] for ref in references[:1000]]

# Load the model and tokenizer
model_location = "../../Models/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_location)
model = AutoModelForSeq2SeqLM.from_pretrained(model_location)

# Move the model to CPU (you can change this to "cuda" if you have a GPU)
device = torch.device("cpu")
model.to(device)

# Create a translation pipeline
translation_pipeline = pipeline(
    'translation', 
    model=model, 
    tokenizer=tokenizer, 
    src_lang='eng_Latn', 
    tgt_lang='fra_Latn', 
    max_length=200, 
    device=0 if torch.cuda.is_available() else -1
)

# Generate translations using the model
translations = []
for text in inputs:
    result = translation_pipeline(text.strip(), max_length=50, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)[0]['translation_text']
    translations.append(result)

# Print generated translations for inspection
print("Generated Translations:", translations[:5])  # Print first 5 translations for inspection

# Calculate BLEU score
bleu = sacrebleu.corpus_bleu(translations, references)
print(f"BLEU score: {bleu.score}")

# Print input and output for inspection
for i, (inp, trans, ref) in enumerate(zip(inputs[:5], translations[:5], references[:5])):  # Limit to first 5 for readability
    print(f"Input {i+1}: {inp.strip()}")
    print(f"Output {i+1}: {trans}")
    print(f"Reference {i+1}: {ref[0]}")
    print()


# 0-10: Very Poor Quality
# Translations are largely unintelligible or incorrect.
# 10-20: Poor Quality
# Many errors; some correct words or short phrases, but overall structure flawed.
# 20-30: Mediocre Quality
# Somewhat understandable; significant errors in grammar and syntax.
# 30-40: Fair Quality
# Understandable with noticeable errors; main ideas are conveyed.
# 40-50: Good Quality
# Generally accurate; few significant errors, but minor grammatical issues.
# 50-60: Very Good Quality
# Accurate and fluent; minor errors present.
# 60-70: Excellent Quality
# Highly accurate and fluent; very few errors, near professional quality.
# 70-80: Near-Human Quality
# Almost indistinguishable from human translations; rare minor issues.
# 80-90: Human Quality
# Very high accuracy and fluency; minimal and hard-to-spot errors.
# 90-100: Perfect Quality
# Perfect translations exactly matching reference translations; extremely rare.