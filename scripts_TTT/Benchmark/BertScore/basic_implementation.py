import torch
from bert_score import score

candidate = ['The quick brown dog jumps over the lazy fox.']
reference = ['The quick brown fox jumps over the lazy dog.']
P, R, F1 = score(candidate, reference, lang='en')
