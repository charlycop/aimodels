import random

src_file = '/home/shared/Models/Reference Albanian German/Bookshop/EUbookshop.de-sq.de'
tgt_file = '/home/shared/Models/Reference Albanian German/Bookshop/EUbookshop.de-sq.sq'

def split_data(src_file, tgt_file, train_ratio=0.9):
    with open(src_file, 'r', encoding='utf-8') as src, open(tgt_file, 'r', encoding='utf-8') as tgt:
        data = list(zip(src.readlines(), tgt.readlines()))
    
    random.shuffle(data)
    split = int(len(data) * train_ratio)
    
    train_data = data[:split]
    val_data = data[split:]
    
    return train_data, val_data

# Split the data
train_data, val_data = split_data('/home/walter/benchmark/aimodels/scripts_TTT/Benchmark/BertScore/opus-mt-src-tgt/de_clean.txt', '/home/walter/benchmark/aimodels/scripts_TTT/Benchmark/BertScore/opus-mt-src-tgt/sq_clean.txt')

# Write training data
with open('de_train.txt', 'w', encoding='utf-8') as de_train, open('sq_train.txt', 'w', encoding='utf-8') as sq_train:
    for de, sq in train_data:
        de_train.write(de)
        sq_train.write(sq)

# Write validation data
with open('de_val.txt', 'w', encoding='utf-8') as de_val, open('sq_val.txt', 'w', encoding='utf-8') as sq_val:
    for de, sq in val_data:
        de_val.write(de)
        sq_val.write(sq)

print("Data splitting completed.")

