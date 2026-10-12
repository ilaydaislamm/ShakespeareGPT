import os
from typing import List

import torch
import urllib.request

SHAKESPEARE_URL = "https://raw.githubusercontent.com/atilsamancioglu/ShakespeareInput/refs/heads/main/input.txt "
DATA_PATH = "data/shakespeare"


def download_file(url):
    if os.path.exists(DATA_PATH):
        print("File already exists")
        return

    os.makedirs("data",exist_ok = True)
    urllib.request.urlretrieve(url, DATA_PATH)


class CharacterTokenizer:
    def __init__(self,text):
        self.character = sorted(list(set(text)))

        self.vocabulary_size = len(self.character)

        self.char_to_id = {}
        for id,char in enumerate(self.character):
            self.char_to_id[char] = id

        self.id_to_char = {}
        for id,char in enumerate(self.character):
            self.id_to_char[id] = char


    def encode(self, text : str) -> List[int]:
        ids = []
        for char in text:
            ids.append(self.char_to_id[char])

        return ids

    def decode(self,ids : List[int]) -> str:
        text = []

        if isinstance(ids,torch.Tensor):
            ids = ids.tolist()

        for id in ids:
            text.append(self.id_to_char[id])

        return "".join(text)


def load_data(data_url : str,
              file_path : str,
              train_split: float = 0.9):
    download_file(data_url)
    with open(file_path,"r") as file:
        text = file.read()

    tokenizer = CharacterTokenizer(text)

    all_ids = tokenizer.encode(text)
    data = torch.tensor(all_ids,dtype=torch.long)

    train_split = int(len(data) * train_split)

    train_data = data[:train_split]
    test_data = data[train_split:]

    return train_data,test_data,tokenizer

def get_batch(data : torch.Tensor,
              block_size : int = 256,
              batch_size : int = 15):

    max_start = len(data) - block_size - 1

    positions = torch.randint(0,max_start,(batch_size,))

    X_list = []
    y_list = []

    for pos in positions:

        X_list.append(data[pos:pos+block_size])
        y_list.append(data[pos + 1: pos + block_size + 1])

    X = torch.stack(X_list)
    y = torch.stack(y_list)

    return X,y




if __name__ == "__main__":
    train_data, test_data, tokenizer = load_data(SHAKESPEARE_URL,DATA_PATH)




