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
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizerV2(vocab)

text = """"Hello. It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride. is this-- a test?"""
ids = tokenizer.decode(tokenizer.encode(text))
print(ids)