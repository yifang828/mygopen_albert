import torch
from transformers import BertTokenizer, AlbertForSequenceClassification
import pandas as pd

class Mygopen:
    def __init__(self, mode, tokenizer):
        assert mode in ['train', 'test']
        self.mode = mode
        if self.mode == 'train':
            self.df = pd.read_excel('./data/mygopen_baseline_data.xlsx', engine='openpyxl')
        elif self.mode == 'test':
            self.df = pd.read_excel('./data/TEST_mygopen_baseline_data.xlsx', engine='openpyxl')
        # required_label
        
        self.len = len(self.df)
        self.label_map = {0:0, 1:1}
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        label, text = self.df.iloc[idx, :].values
        label_id = self.label_map[label]
        label_tensor = torch.tensor(label_id)
        word_pieces = ['[CLS]']
        tokens = self.tokenizer.tokenize(text)
        word_pieces += tokens + ['[SEP]']
        len_token = len(word_pieces)

        # 將整個token序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        segment_tensor = torch.tensor([0]*len_token, dtype = torch.long)
        return(tokens_tensor, segment_tensor, label_tensor)
    
    def __len__(self):
        return self.len

def main():
    pretrained_model = "voidful/albert_chinese_base"
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    trainset = Mygopen('train', tokenizer=tokenizer)
    testset = Mygopen('test', tokenizer=tokenizer)
    # print('testset label list: ', testset.df.iloc[:,0].values)
    # 原始文本
    print(trainset.df.shape)
    sample_idx = 0
    print(trainset.df.iloc[sample_idx].values)
    label, text = trainset.df.iloc[0].values
    
    # 利用原始文本取出轉換後的id tensor
    tokens_tensor, segments_tensor, label_tensor = trainset[sample_idx]
    # 將 token_tensor 還原成文本
    tokens = trainset.tokenizer.convert_ids_to_tokens(tokens_tensor.tolist())
    combined_text = ''.join(tokens)
    print(f"""[原始文本]
    句子 1：{text}
    分類  ：{label}

    --------------------

    [Dataset 回傳的 tensors]
    tokens_tensor  ：{tokens_tensor}

    segments_tensor：{segments_tensor}

    label_tensor   ：{label_tensor}

    --------------------

    [還原 tokens_tensors]
    {combined_text}
    """)

if __name__ == '__main__':
    main()

