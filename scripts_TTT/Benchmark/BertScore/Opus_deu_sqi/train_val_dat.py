import subprocess
from transformers import MarianMTModel, MarianTokenizer
import os

# Paths and filenames
MARIAN = "/home/shared/Models/marian-dev"  # Update this path
SRC_TRAIN = "de_train.txt"
TGT_TRAIN = "sq_train.txt"
SRC_VAL = "de_val.txt"
TGT_VAL = "sq_val.txt"
VOCAB = "de-sq.model"

# Construct the Marian command
marian_command = [
    f"{MARIAN}/marian",
    "--model", "model.npz",
    "--type", "transformer",
    "--train-sets", SRC_TRAIN, TGT_TRAIN,
    "--max-length", "150",
    "--vocabs", VOCAB, VOCAB,
    "--mini-batch-fit", "-w", "5000", "--maxi-batch", "1000",
    "--early-stopping", "10",
    "--valid-freq", "5000", "--save-freq", "5000", "--disp-freq", "500",
    "--valid-metrics", "cross-entropy", "perplexity", "translation",
    "--valid-sets", SRC_VAL, TGT_VAL,
    "--valid-script-path", "validate.py",
    "--log", "model.log", "--valid-log", "model.valid.log",
    "--enc-depth", "6", "--dec-depth", "6",
    "--transformer-heads", "8",
    "--transformer-postprocess-emb", "d",
    "--transformer-postprocess", "dan",
    "--transformer-dropout", "0.1",
    "--label-smoothing", "0.1",
    "--learn-rate", "0.0003", "--lr-warmup", "16000", "--lr-decay-inv-sqrt", "16000", "--lr-report",
    "--optimizer-params", "0.9", "0.98", "1e-09", "--clip-norm", "5",
    "--tied-embeddings-all",
    "--devices", "0", "1", "2", "3", "--sync-sgd", "--seed", "1111",
    "--exponential-smoothing"
]

# Run the Marian command
subprocess.run(marian_command)

print("Training completed.")