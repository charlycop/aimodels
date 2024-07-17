#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

MARIAN=/home/shared/Models/marian-dev/build  # Adjust this path as needed

$MARIAN/marian-decoder \
    -m model.npz \
    -v de-sq.model de-sq.model \
    --devices 0 \
    --mini-batch 8 --maxi-batch 25 --maxi-batch-sort src \
    --beam-size 6 --normalize 0.6 \
    --max-length-factor 2.0 --max-length 100 \
    --log marian.log --log-level debug\
    < de_val.txt > sq_val.out

# Check if output and reference have the same number of lines
if [ $(wc -l < sq_val.out) -ne $(wc -l < sq_val.txt) ]; then
    echo "Error: sq_val.out and sq_val.txt have different numbers of lines."
    exit 1
fi

# Use Python to run sacrebleu
python3 -m sacrebleu sq_val.txt < sq_val.out

# Add BERTScore evaluation
if python3 -c "import bert_score" 2>/dev/null; then
    python3 -m bert_score -r sq_val.txt -c sq_val.out --lang sq
else
    echo "Error: bert-score is not installed or not found in the current Python environment."
    echo "Try installing it with: pip install bert-score"
    exit 1
fi