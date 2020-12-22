from preprocess.lt_udn_dataset import LtUdnDataset
from transformers import BertTokenizer, AlbertForSequenceClassification
import torch

class LlNclOthersPsX(LtUdnDataset):
    def __init__(self, mode, tokenizer):
        assert mode in ['train', 'test']
        self.mode = mode
        if self.mode == 'train':
            self.df = super().get_train_dataset()
            self.df.loc[(self.df['label'] !='X') & (self.df['label'] !='LL') & (self.df['label'] !='NCL') & (self.df['label'] !='PS')\
                & (self.df['label'] !='BWF') & (self.df['label'] !='LLL')] = 'OTHERS'
            train_label_ncl = self.df[self.df['label']=='NCL']
            train_label_ncl = train_label_ncl.sample(n=60)
            required_ncl_idx = train_label_ncl.index

            train_label_ps = self.df[self.df['label']=='PS']
            train_label_ps = train_label_ps.sample(n=24)
            required_ps_idx = train_label_ps.index

            train_label_x = self.df[self.df['label']=='X']
            train_label_x = train_label_x.sample(n=404)
            required_x_idx = train_label_x.index

            self.df = self.df[(self.df['label']=='LL') | (self.df['label']=='OTHERS') | (self.df.index.isin(required_ncl_idx)) | \
                (self.df.index.isin(required_ps_idx)) | (self.df.index.isin(required_x_idx))]
            self.df = self.df.drop(['keyWord'], axis=1)
            
        elif self.mode == 'test':
            self.df = super().get_test_dataset()
            self.df.loc[(self.df['label'] !='X') & (self.df['label'] !='LL') & (self.df['label'] !='NCL') & (self.df['label'] !='PS')\
                & (self.df['label'] !='BWF')] = 'OTHERS'
            test_label_ll = self.df[self.df['label']=='LL']
            test_label_ll = test_label_ll.sample(n=98)
            required_test_ll_idx = test_label_ll.index

            test_label_x = self.df[self.df['label']=='X']
            test_label_x = test_label_x.sample(frac=0.25)
            self.df = self.df[(self.df.index.isin(required_test_ll_idx)) | (self.df['label']=='OTHERS')  | (self.df['label']=='NCL') | \
                (self.df['label']=='PS') | (self.df.index.isin(test_label_x.index))]
            self.df = self.df.drop(['keyWord'], axis=1)
        # required_label
        
        self.len = len(self.df)
        self.label_map = {'LL':0, 'NCL':1, 'OTHERS':2, 'PS':3, 'X':4}
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        label, text = self.df.iloc[idx, :].values
        # print('getitem: ', text)
        # print('getitem text type: ', type(text))
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
    trainset = LlNclPsX('train', tokenizer=tokenizer)
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