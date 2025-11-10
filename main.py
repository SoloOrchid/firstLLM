import os
import urllib.request

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

import re

#here we create a simple tokanizer
result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
result = [item.strip() for item in result if item.strip()]
preprocessed = result

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

#add unknown and end of line to the list of tokens in case we run into unknown words
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

#convert tokens into token IDs
vocab = {token:integer for integer,token in enumerate(all_tokens)}


"""
this class takes a vocab and saves the vocab as string to int then invert the map.
the encode takes any text and uses regex to break down the text to tokens then create the token IDs,
this makes it so that for each word in the vocab we give it an ID.

"""
# class SimpleTokenizerV2:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i: s for s, i in vocab.items()}
#
#     def encode(self, text):
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#
#         preprocessed = [
#             item.strip() for item in preprocessed if item.strip()
#         ]
#         preprocessed = [
#             item if item in self.str_to_int
#             else "<|unk|>" for item in preprocessed
#         ]
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
#
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])
#         # Replace spaces before the specified punctuations
#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
#         return text
#
# tokenizer = SimpleTokenizerV2(vocab)
#
# text = """"Hello. It's the last he painted, you know,"
#            Mrs. Gisburn said with pardonable pride. is this-- a test?"""
# ids = tokenizer.decode(tokenizer.encode(text))
# print(ids)

# byte pair encoding
import tiktoken

# tokenizer = tiktoken.get_encoding("gpt2")
#
# enc_text = tokenizer.encode(raw_text)
# enc_sample = enc_text[:50]
#
#
# context_size = 4
# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]
#
# print(f"x: {x}")
# print(f"y:      {y}")


#user pytorch to predict the next word of the prompt
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        #tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        #use a sliding window to to chunck the book into overlapping sequensez of max length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataLoader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

    #init the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    #create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    #create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )

    return dataloader

dataloader = create_dataLoader_v1(raw_text, batch_size=1, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
#second batch
print(next(data_iter))


# create token embeddings
input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3
# torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)

#encoding word position
vocab_size = 50257
output_dim = 257 #this needs to go up a lot more for any serious llm

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataLoader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("token IDs:\n", inputs)
print("\nInput Shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
print(token_embeddings[0,0])

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

#coding attention mechanism

