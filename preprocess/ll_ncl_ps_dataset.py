from lt_udn_dataset import LtUdnDataset
from transformers import BertTokenizer, AlbertForSequenceClassification
import torch

class LlNclPs(LtUdnDataset):
    def __init__(self, mode, tokenizer):
        assert mode in ['train', 'test']
        self.mode = mode
        if self.mode == 'train':
            self.df = super().get_train_dataset()
        elif self.mode == 'test':
            self.df = super().get_test_dataset()
        # required_label
        self.df = self.df[self.df['label']!='X']
        self.df = self.df[self.df['label'].isin(['LL', 'NCL', 'PS'])]
        self.df = self.df.drop(['keyWord'], axis=1)
        
        self.len = len(self.df)
        self.label_map = {'LL':0, 'NCL':1, 'PS':2}
        self.tokenizer = tokenizer
    
    def get_required_label_dataset(self):
        self.df = self.df[self.df['label']!='X']
        self.df = self.df[self.df['label'].isin(['LL', 'NCL', 'PS'])]
        self.df = self.df.drop(['keyWord'], axis=1)
        # print(self.df)

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
    trainset = LlNclPs('train', tokenizer=tokenizer)
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

