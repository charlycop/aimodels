from bert_score import BERTScorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import wandb
import psutil
import numpy as np
from tqdm import tqdm
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

wandb.login()

def get_resource_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    cpu_percent = process.cpu_percent()
    return {
        'memory_usage_mb': memory_info.rss / 1024 / 1024,
        'cpu_percent': cpu_percent
    }

source_file = "/home/shared/Models/Reference Albanian German/NLLB opus/NeuLab-TedTalks.de-sq.de"

target_file = '/home/shared/Models/Reference Albanian German/NLLB opus/NeuLab-TedTalks.de-sq.sq'

with open(source_file, 'r', encoding='utf-8') as src, open(target_file, 'r', encoding='utf-8') as tgt:
    inputs = src.readlines()
    references = tgt.readlines()

inputs = inputs[:1000]
references = [[ref.strip()] for ref in references[:1000]]

local_model_location = "/home/shared/Models/nllb-200-3.3B/"
huggingface_model = "facebook/nllb-200-3.3B"



model_location = huggingface_model


def batch_bert_score(inputs, references, model_name, batch_size=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = AutoModel.from_pretrained(model_name).to(device)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    
    def custom_tokenizer(texts):
        return tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Initialize BERTScorer with custom model and tokenizer
    scorer = BERTScorer(lang="de", model_type=model_name, num_layers=1, 
                        batch_size=batch_size, device=device)
    all_p = []
    all_r = []
    all_f1 = []
    
    wandb.init(project="bertscore-nllb200", name="evaluation")
    
    for i in tqdm(range(0, len(references), batch_size)):
        batch_refs = references[i:i+batch_size]
        batch_cands = inputs[i:i+batch_size]
        
        P, R, F1 = scorer.score(batch_cands, batch_refs)
        
        all_p.extend(P.tolist())
        all_r.extend(R.tolist())
        all_f1.extend(F1.tolist())
        
        resource_usage = get_resource_usage()
        
        wandb.log({
            'batch': i // batch_size,
            'avg_precision': P.mean().item(),
            'avg_recall': R.mean().item(),
            'avg_f1': F1.mean().item(),
            **resource_usage
        })
    
    return np.mean(all_p), np.mean(all_r), np.mean(all_f1)


model_name = "facebook/nllb-200-3.3B"
avg_p, avg_r, avg_f1 = batch_bert_score(inputs, references, model_name)

print(f"Average Precision: {avg_p:.4f}")
print(f"Average Recall: {avg_r:.4f}")
print(f"Average F1: {avg_f1:.4f}")

wandb.finish()


# def batch_bert_score(references, candidates, model_name, batch_size=32):
    # device = torch.device("cpu")
    # model.to(device)
    
    # tokenizer = AutoTokenizer.from_pretrained(local_model_location)
    # model = AutoModelForSeq2SeqLM.from_pretrained(local_model_location)


# translation_pipeline = pipeline(
#     'translation', 
#     model=model, 
#     tokenizer=tokenizer, 
#     src_lang='ger_Latn', 
#     tgt_lang='eng_Latn', 
#     max_length=200, 
#     device=0 if torch.cuda.is_available() else -1
# )

# # Generate translations using the model
# translations = []
# for text in inputs:
#     result = translation_pipeline(text.strip(), max_length=200, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)[0]['translation_text']
#     translations.append(result)



# # Calculate BERTScore
# P, R, F1 = score(cands, refs, model_type=model, tokenizer=tokenizer, num_layers=None, verbose=False, idf=False, batch_size=3, nthreads=4, all_layers=False)

# # Print the BERTScore
# print(f"Precision: {P.item():.4f}, Recall: {R.item():.4f}, F1: {F1.item():.4f}")
