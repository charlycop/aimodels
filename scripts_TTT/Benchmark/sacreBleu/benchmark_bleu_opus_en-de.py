import sacrebleu
from transformers import MarianTokenizer, MarianMTModel

# Define paths to the extracted Europarl files
target_file = '../wmt17.de-en.de'
source_file = '../wmt17.en-de.en'

# Read the data
with open(source_file, 'r', encoding='utf-8') as src, open(target_file, 'r', encoding='utf-8') as tgt:
    inputs = src.readlines()
    references = tgt.readlines()

# Limit the data size for testing to avoid memory issues
max_lines = 200
inputs = inputs[:200]
references = [[ref.strip()] for ref in references[:max_lines]]

# Define the local and Hugging Face model locations
local_model_location = "../../Models/opus-mt-en-de"
huggingface_model = "Helsinki-NLP/opus-mt-en-de"

actual_model = huggingface_model

tokenizer = MarianTokenizer.from_pretrained(actual_model)
model = MarianMTModel.from_pretrained(actual_model)

# Generate translations using the model
translations = []
cpt=0
for text in inputs:

    # Encode the input text
    encoded_input = tokenizer(text, return_tensors="pt")

    # Generate the translation
    output = model.generate(**encoded_input)
    
    translations.append(tokenizer.decode(output[0], skip_special_tokens=True))

    cpt += 1
    print("Lines processed : " + str(cpt) + "/" + str(max_lines))

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