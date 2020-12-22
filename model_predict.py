import time
import json
import re
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, AlbertForSequenceClassification
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
BATCH_SIZE = 64
# test
# STR = '原來竹子會結果？一般草本植物每年都會開花結果，但是竹子卻不同，從五十年到一百二十年不等，\
#     視不同品種的竹類而有所差異。由於所有竹類的植物，都不是靠開花結果來繁殖的。而大都是由同一棵竹\
#     的根部長出新筍繁殖分枝出來，食用的竹筍，就是竹子的根部分株所生出來的新芽。'

# split文章的每個句子 by ',' '。','!','?'
def splitContext(str):
    tokensList = [s.span() for s in re.finditer(r'[,，。！!?？\s]+', str)]
    lastIdx = 0
    splitedList = []
    for token in tokensList:
        splitedList.append(str[lastIdx : token[1]])
        lastIdx =  token[1]
    return splitedList
# print(splitContext(str))

def getContext():
    # with open('data/transfer_learning/rumor.json', encoding='utf-8') as jf:
    with open('data/transfer_learning/truth.json', encoding='utf-8') as jf:
        datas = json.load(jf)
        result = {}
        count = 0
        for data in datas:
            for k, v in data.items():
                if k == "source" and v !="" and len(v)<512:
                    result[count] = v
                    count +=1
                # if k == "truth" and v !="" and len(v)<512:
                #     result[count] = v
                #     count +=1
        return result
context = getContext()
for k, v in context.items():
    context[k] = splitContext(v)
print("context size: ", len(context))

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    # attention masks將token_tensors裏頭不為zero padding的位置設為1讓BERT只關注這些位置的tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    return tokens_tensors, segments_tensors, masks_tensors

# 初始化一個每次回傳64個訓練樣本的DataLoader
# 利用collate_fn將list of samples 合併成一個 mini-batch是關鍵

def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0

    with torch.no_grad():
        # 遍尋整個資料集
        for data in dataloader:
            # 將所有tensor 移到GPU上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
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

########預測########
results = []
tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_base')
model = AlbertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

init_time = time.time()
for keys, values in context.items():
    print('keys: ', keys)
    target = MygopenDataset(values, tokenizer=tokenizer)
    targetloader =  DataLoader(target, collate_fn=create_mini_batch)
    # 對文章的每個句子(split by ',' '。','!','?')預測
    result = np.array([0, 0, 0, 0, 0])
    label_map = {'LL':0, 'NCL':1, 'OTHERS':2, 'PS':3, 'X':4}
    # 用分類器來預測測試集
    predictions = get_predictions(model, targetloader)
    # 將預測的label id 轉回 labe文字
    if predictions is None:
        continue
    print(predictions.tolist())
    for p in predictions.tolist():
        result[p]+=1 
    print(result)
    results.append(result)

print('results size= ', len(results))
cost_time = time.time() - init_time
print(f"""cost time: {cost_time}""")

arr = np.array(results)
df = pd.DataFrame({'LL':arr[:,0], 'NCL':arr[:,1], 'OTHERS':arr[:,2], 'PS':arr[:,3], 'X':arr[:,4]})
df.to_excel('./data/transfer_learning/truth_ll_ncl_others_ps_x.xlsx', index=False)