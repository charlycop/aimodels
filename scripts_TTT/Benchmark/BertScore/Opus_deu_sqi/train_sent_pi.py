import sentencepiece as spm

# Define the parameters
input_files = ['de_clean.txt', 'sq_clean.txt']
model_prefix = 'de-sq'
vocab_size = 32000
character_coverage = 1.0
model_type = 'bpe'

# Create a comma-separated string of input files
input_argument = ','.join(input_files)

# Train the SentencePiece model
spm.SentencePieceTrainer.train(
    input=input_argument,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    character_coverage=character_coverage,
    model_type=model_type,
    input_sentence_size=1000000,  # Adjust based on your data size
    shuffle_input_sentence=True,
    normalization_rule_name='nmt_nfkc_cf'
)

print(f"SentencePiece model trained and saved with prefix: {model_prefix}")