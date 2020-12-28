import re
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, AlbertModel, AlbertForSequenceClassification
from preprocess.mygopen_dataset import MygopenDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# PRETRAINED_MODEL_NAME = 'll_ncl_ps_albert_base'
# PRETRAINED_MODEL_NAME = 'll_ncl_ps_x_albert_base'
# PRETRAINED_MODEL_NAME = 'll_ncl_others_ps_albert_base'
PRETRAINED_MODEL_NAME = 'll_ncl_others_ps_x_albert_base'
# NUM_LABELS = 3
# NUM_LABELS = 4
NUM_LABELS = 5
tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_base')
model = AlbertModel.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
sequence_model = AlbertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

cls_list = []
tokens_avg_list = []
results = []

# train_df = pd.read_excel('./data/mygopen_train.xlsx', engine='openpyxl')
train_df = pd.read_excel('./data/mygopen_test.xlsx', engine='openpyxl')
train_txt = train_df['context']

# split文章的每個句子 by ',' '。','!','?'
def splitContext(str):
    tokensList = [s.span() for s in re.finditer(r'[,，。！!?？\s]+', str)]
    lastIdx = 0
    splitedList = []
    for token in tokensList:
        splitedList.append(str[lastIdx : token[1]])
        lastIdx =  token[1]
    return splitedList

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    # attention masks將token_tensors裏頭不為zero padding的位置設為1讓BERT只關注這些位置的tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    return tokens_tensors, segments_tensors, masks_tensors

def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0

    with torch.no_grad():
        # 遍尋整個資料集
        for data in dataloader:
            # 將所有tensor 移到GPU上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda") for t in data if t is not None]
            # 前3個tensors分別為tokens, segments, masks，建議再將這些tensors丟入model時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors)
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)

            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            # 當前batch記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions

count = 0
for txt in train_txt:
    # print(txt)
    count += 1
    print('count: ', count)
    print('len: ', len(txt))
    inputs = tokenizer(txt)
    tokens_tensor = inputs['input_ids']
    input_ids = torch.tensor(tokens_tensor).unsqueeze(0) #batch size 1

    output = model(input_ids)
    tokens_avg = output[0].mean(1)[0].detach().numpy()
    cls_output = output[1][0].detach().numpy()
    # print('token_avg: ', tokens_avg)
    tokens_avg_list.append(tokens_avg)
    # print('cls_output: ', cls_output)
    cls_list.append(cls_output)

    ####預測每句label####
    split_txt = splitContext(txt)
    target = MygopenDataset(split_txt, tokenizer=tokenizer)
    targetloader =  DataLoader(target, batch_size=1, collate_fn=create_mini_batch)
    # 對文章的每個句子(split by ',' '。','!','?')預測
    result = np.array([0, 0, 0, 0, 0])
    # label_map = {'LL':0, 'NCL':1, 'PS':2}
    # label_map = {'LL':0, 'NCL':1, 'OTHERS':2, 'PS':3, 'X':4}
    
    # 用分類器來預測測試集
    predictions = get_predictions(sequence_model, targetloader)
    # 將預測的label id 轉回 labe文字
    if predictions is not None:
        # print(predictions.tolist())
        for p in predictions.tolist():
            result[p]+=1 
    print(result)
    results.append(result)

train_df['cls'] = cls_list
train_df['token_avg'] = tokens_avg_list
train_df['predict_feature'] = results
# train_df.to_excel('./data/cls_token_avg/train_ll_ncl_others_ps_x.xlsx', engine='xlsxwriter', index=False)
train_df.to_excel('./data/cls_token_avg/test_ll_ncl_others_ps_x.xlsx', engine='xlsxwriter', index=False)


'''
inputs = tokenizer(testString)
tokens_tensor = inputs['input_ids']
print('tokens_tensor: ',tokens_tensor)
input_ids = torch.tensor(tokens_tensor).unsqueeze(0) #batch size 1
print('input_ids: ',input_ids)

output = model(input_ids)
print('output: ', output)
print('last hidden state: ', output[0])# last hidden state: tensor([[[-0.2404,...],[-0.3037,...]]])
print('token avg: ', output[0].mean(1))
print('token avg: ', output[0].mean(1).size())

output_np = output[0].mean(1)[0].detach().numpy()
print('token avg: ', output[0].mean(1)[0].size())
print(output_np)
# output_pd = pd.DataFrame(output_np, columns = ['tokens_avg'])
# print('last hidden state size: ', output[0].size())# last hidden state size: torch.Size([1, 60, 768])
# print('pooler output: ', output[1])# pooler output: tensor([[ 0.6845,...]]) 
# print('pooler output size: ', output[1].size()) # pooler output size:  torch.Size([1, 768])
print( output[1][0].detach().numpy()) # output[1][0] = tensor([ 0.6845,  0.9633,...])
# print('output[1][0]: ', output[1][0].size()) # size = [768]
'''