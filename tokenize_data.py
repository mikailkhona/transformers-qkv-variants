
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
import pdb
import numpy as np

# Load dataset: Minipile
# dataset = load_dataset("JeanKaddour/minipile")
dataset = load_dataset("open-web-math/open-web-math")
pdb.set_trace()
splits = ['train', 'validation', 'test'] 

# len = 500
val_data = dataset['validation']

# len = 1M
train_data = dataset['train']

print(f"Number of train sentences: {len(dataset['train'])}")
print(f"Example sentence:{dataset['train'][0]['text']}")

# Tokenize data
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

def encode(sentence):
    return tokenizer(sentence['text'], truncation=True, padding="max_length")

#tokenize val data

# Dataset({
#     features: ['text', 'input_ids', 'attention_mask'],
#     num_rows: 500
# })

val_data_tokenized = val_data.map(encode, batched=True)
val_data_tokenized.set_format(type="torch", columns=["input_ids"])
np.save('open-web-math/tokenized_eval_data.npy',val_data_tokenized['input_ids'].numpy())

# tokenize train data
train_data_tokenized = train_data.map(encode, batched=True)
train_data_tokenized.set_format(type="torch", columns=["input_ids"])
np.save('open-web-math/tokenized_train_data.npy',train_data_tokenized['input_ids'].numpy())
# Create dataloader
# dataloader = torch.utils.data.DataLoader(val_data_tokenized, batch_size=32, shuffle=False)
# for sentence in dataloader:
#     print(sentence['input_ids'])
