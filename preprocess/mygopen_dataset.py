import json
import torch
from transformers import BertTokenizer, AlbertForSequenceClassification
import pandas as pd

class MygopenDataset:
    def __init__(self, txt, tokenizer):
        self.df = pd.DataFrame.from_dict(txt)
        self.len = len(self.df)
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        text = str(self.df.iloc[idx, :].values)[1:-1]
        # print('getitem text: ', text)
        word_pieces = ['[CLS]']
        tokens = self.tokenizer.tokenize(text)[:510]
        word_pieces += tokens + ['[SEP]']
        len_token = len(word_pieces)
        # 將整個token序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        segment_tensor = torch.tensor([0]*len_token, dtype = torch.long)
        return(tokens_tensor, segment_tensor)
    
    def __len__(self):
        return self.len

def main():
    # test
    test = '原來竹子會結果？一般草本植物每年都會開花結果，但是竹子卻不同，從五十年到一百二十年不等，\
        視不同品種的竹類而有所差異。由於所有竹類的植物，都不是靠開花結果來繁殖的。而大都是由同一棵竹\
        的根部長出新筍繁殖分枝出來，食用的竹筍，就是竹子的根部分株所生出來的新芽。'

    pretrained_model = "ll_ncl_ps_albert_base"
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    trainset = MygopenDataset(splitContext(test), tokenizer=tokenizer)

    # 原始文本
    print(trainset.df.shape)
    sample_idx = 0
    print(trainset.df.iloc[sample_idx].values)
    text = trainset.df.iloc[0].values
    # 利用原始文本取出轉換後的id tensor
    tokens_tensor, segments_tensor = trainset[sample_idx]
    # 將 token_tensor 還原成文本
    tokens = trainset.tokenizer.convert_ids_to_tokens(tokens_tensor.tolist())
    combined_text = ''.join(tokens)
    print(f"""[原始文本]
    句子 1：{text}

    --------------------

    [Dataset 回傳的 tensors]
    tokens_tensor  ：{tokens_tensor}

    segments_tensor：{segments_tensor}

    --------------------

    [還原 tokens_tensors]
    {combined_text}
    """)

if __name__ == '__main__':
    main()
    
