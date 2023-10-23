import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model.model import LSTMTagger
import pandas as pd

def embedded(sentence, word2vec):
    split_list = re.findall(r"[\w']+|[.,!?;]", sentence.lower())

    output = []
    for word in split_list:
        output.append(
            word2vec.loc[word2vec[0] == word, 1:].iloc[0].tolist()
        )
        
    return torch.Tensor(output)


if __name__ == "__main__":
    torch.manual_seed(1)

    word2vec = pd.read_csv("glove.6B.50d.txt", delim_whitespace=True, header=None, engine='python', encoding='utf-8')

    print("Starting embedded...")