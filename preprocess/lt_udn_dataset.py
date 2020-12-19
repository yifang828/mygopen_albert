from torch.utils.data import Dataset
import pandas as pd

class LtUdnDataset(Dataset):
    
    def get_train_dataset(self):
        file_path = './data/UDN_train.tsv'
        train_df_udn = pd.read_csv(file_path, sep='\\t', encoding='utf-8', engine='python')
        train_df_udn.columns = ['label', 'keyWord', 'text']
        file_path_lt = './data/LT_train.tsv'
        train_df_lt = pd.read_csv(file_path_lt, sep='\\t', encoding='utf-8', engine='python')
        train_df_lt.columns = ['label', 'keyWord', 'text']
        train_df = pd.concat([train_df_udn,train_df_lt],axis=0, ignore_index=True)
        return train_df

    def get_test_dataset(self):
        file_path = './data/UDN_test.tsv'
        test_df_udn = pd.read_csv(file_path, sep='\\t', encoding='utf-8', engine='python')
        test_df_udn.columns = ['label', 'keyWord', 'text']
        file_path_lt = './data/LT_test.tsv'
        test_df_lt = pd.read_csv(file_path_lt, sep='\\t', encoding='utf-8', engine='python')
        test_df_lt.columns = ['label', 'keyWord', 'text']
        test_df = pd.concat([test_df_udn,test_df_lt],axis=0, ignore_index=True)
        return test_df
