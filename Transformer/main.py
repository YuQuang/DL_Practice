import tensorflow_datasets as tfds
from tqdm import tqdm
import torch
import math

def loadWmtDataset():
    config = tfds.translate.wmt.WmtConfig(
            version="0.0.1",
            language_pair=("zh","en"),
            subsets={
                tfds.Split.TRAIN: [
                    "wikititles_v1",
                    # "commoncrawl",
                    # "newscommentary_v14",
                    # "uncorpus_v1",
                ],
                tfds.Split.VALIDATION: [
                    "newstest2018",
                ],
            },
        )
    return tfds.load(
        name="wmt_translate", 
        split="train[:1%]", 
        as_supervised=True, 
        builder_kwargs={ "config": config }
    )

def loadVocab(text_dataset, vocab_file="./en_vocab.txt", max_subword_length=20):
    try:
        subword_encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(vocab_file)
        print(f"載入已建立的字典： {vocab_file}")
    except:
        print("沒有已建立的字典，從頭建立。")
        subword_encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            text_dataset, 
            target_vocab_size=2**13,
            max_subword_length=max_subword_length
        )
        subword_encoder.save_to_file(vocab_file)
    return subword_encoder

def addBOSandEOS(sentence, vocab):
    return [vocab.vocab_size] + sentence + [vocab.vocab_size+1]

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

if __name__ == '__main__':
    # Load Dataset & Vocab
    train_dataset = loadWmtDataset()
    subword_encoder_en = loadVocab(
        ( en.numpy() for _, en in tqdm(train_dataset)),
        "./en_vocab.txt"
    )
    subword_encoder_zh = loadVocab(
        ( zh.numpy() for zh, _ in tqdm(train_dataset)),
        "./zh_vocab.txt",
        max_subword_length=1
    )
    
    MAX_LENGTH = 40
    for index, (zh, en) in enumerate(train_dataset):
        zh = addBOSandEOS(
                subword_encoder_zh.encode( zh.numpy().decode() ),
                subword_encoder_zh
            )
        en = addBOSandEOS(
                subword_encoder_en.encode( en.numpy().decode() ),
                subword_encoder_en
            )
        en += [0] * (MAX_LENGTH - len(en))
        zh += [0] * (MAX_LENGTH - len(zh))

        if len(zh) > 40 or len(en) > 40: continue
        

        # Word Embedding, +2 is for BOS & EOS
        en_embeds = torch.nn.Embedding(
                subword_encoder_en.vocab_size+2,
                5
            )
        zh_embeds = torch.nn.Embedding(
                subword_encoder_zh.vocab_size+2,
                5
            )
        
        emb_en = en_embeds(torch.tensor([en], dtype=torch.int64 ))
        emb_zh = zh_embeds(torch.tensor([zh], dtype=torch.int64 ))

        emb_inp = torch.cat((emb_en, emb_zh))
        v = torch.randint(low=0, high=2, size=emb_inp.shape, dtype=torch.float32)
        attention = scaled_dot_product_attention(emb_inp, emb_inp, v)
        print(attention.shape)
        break
        
