import torch
from transformers import AlbertTokenizer, AlbertForTokenClassification, BertTokenizer
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.util import ngrams

class Bigram:
    def __init__(self, txt, tokenizer):
        self.df = pd.DataFrame(data=txt)
        self.len = len(self.df)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # text = self.df.iloc[idx, :].values[0]
        # encoded = self.tokenizer.encode_plus(text, add_special_tokens=True,\
        #     max_length=32, pad_to_max_length=True, return_tensor='pt')
        # print(encoded['input_ids'])
        # return encoded['input_ids']
        text = self.df.iloc[idx, :].values[0]
        print('getitem text:', text)
        word_pieces = ['[CLS]']
        tokens = self.tokenizer.tokenize(text)
        print('tokens:', tokens)
        word_pieces += tokens + ['[SEP]']
        print('word_piece:', word_pieces)
        len_token = len(word_pieces)
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        segment_tensor = torch.tensor([0]*len_token, dtype=torch.long)
        return(tokens_tensor, segment_tensor)

    def __len__(self):
        return self.len

def print_tensor(trainset):
    print(trainset.df.shape)
    sample_idx = 0
    text = trainset.df.iloc[sample_idx].values[0]
    print(text)
    # tokens_tensor = trainset[sample_idx]
    # tokens = trainset.tokenizer.convert_ids_to_tokens(tokens_tensor)
    # combined_text = ' '.join(tokens)
    tokens_tensor, segments_tensor = trainset[sample_idx]
    tokens = trainset.tokenizer.convert_ids_to_tokens(tokens_tensor.tolist())
    combined_text = ' '.join(tokens)
    print(f"""[原始文本]
    句子1: {text}

    -------------------
    [Dataset 回傳的 tensors]
    tokens_tensor  ：{tokens_tensor}

    --------------------

    [還原 tokens_tensors]
    {combined_text}
    """)

def ngram(context, n):
     return [context[i:i+n] for i in range(0, len(context)-1)]

def main():
    # txt = 'life is a box of chocolate'
    # pretrained_model = 'albert-base-v2'
    # tokenizer = AlbertTokenizer.from_pretrained(pretrained_model)
    txt = '原來竹子會結果？一般草本植物每年都會開花結果，但是竹子卻不同，從五十年到一百二十年不等，視不同品種的竹類而有所差異。由於所有竹類的植物，都不是靠開花結果來繁殖的。而大都是由同一棵竹的根部長出新筍繁殖分枝出來，食用的竹筍，就是竹子的根部分株所生出來的新芽。'
    pretrained_model = 'voidful/albert_chinese_base'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    trainset = Bigram([txt], tokenizer=tokenizer)
    # 原始文本
    print_tensor(trainset)
# 1.加底線, e.g: life_is is_a a_box box_of of_chocolate chocolate
    # token = nltk.word_tokenize(txt)
    # bigram_list = list(ngrams(token, 2))
    bigram_list = list(ngram(txt, 2))
    print(bigram_list)
    bigram_txt = ''
    for bigram in bigram_list:
        bigram = '_'.join(bigram)
        bigram_txt += bigram + ' '
    print(bigram_txt)
    trainset = Bigram([bigram_txt.strip()], tokenizer=tokenizer)
    print_tensor(trainset)
# 2.合併成一個字, e.g: lifeis isa abox boxof ofchocolate chocolate
    # token = nltk.word_tokenize(txt)
    # bigram_list = list(ngrams(token, 2))
    bigram_list = list(ngram(txt, 2))
    bigram_txt = ''
    for bigram in bigram_list:
        bigram = ''.join(bigram)
        bigram_txt += bigram + ' '
    # print(bigram_txt)
    trainset = Bigram([bigram_txt.strip()], tokenizer=tokenizer)
    print_tensor(trainset)

if __name__ == '__main__':
    # nltk.download('punkt')
    main()