from transformers import BertTokenizer, AlbertForSequenceClassification
import torch
from preprocess.ll_ncl_ps_dataset import LlNclPs
from preprocess.ll_ncl_ps_x_dataset import LlNclPsX
from preprocess.ll_ncl_others_ps_dataset import LlNclOthersPs
from preprocess.ll_ncl_others_ps_x_dataset import LlNclOthersPsX
from preprocess.mygopen import Mygopen
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import time

PRETRAINED_MODEL_NAME = 'voidful/albert_chinese_base'
# NUM_LABELS = 3 # LlNclPs
# NUM_LABELS = 4 # LlNclPsX # LlNclOtheresPs
# NUM_LABELS = 5
NUM_LABELS = 2

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
model = AlbertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    # 測試集有labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    # attention masks將token_tensors裏頭不為zero padding的位置設為1讓BERT只關注這些位置的tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

# 初始化一個每次回傳64個訓練樣本的DataLoader
# 利用collate_fn將list of samples 合併成一個 mini-batch是關鍵
BATCH_SIZE = 64
# trainset = LlNclPs('train', tokenizer=tokenizer)
# trainset = LlNclPsX('train', tokenizer=tokenizer)
# trainset = LlNclOthersPs('train', tokenizer=tokenizer)
# trainset = LlNclOthersPsX('train', tokenizer=tokenizer)
trainset = Mygopen('train', tokenizer=tokenizer)
trainloader =  DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

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

# 讓模型跑在GPU上並取得訓練集的分類準確率
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
model = model.to(device)
_, acc = get_predictions(model, trainloader, compute_acc=True)
print("classification acc:", acc)

# 計算整個分類模型的參數量,線性分類器的參數量
# def get_learnable_params(module):
#     return [p for p in module.parameters() if p.requires_grad]
# model_params = get_learnable_params(model)
# clf_params = get_learnable_params(model.classifier)
# print(f"""
# 整個分類模型的參數量：{sum(p.numel() for p in model_params)}
# 線性分類器的參數量：{sum(p.numel() for p in clf_params)}
# """)
# 計時開始
start_time = time.time()
# 訓練模型
model.train()
# 使用Adam Optim更新整個分類模型的參數
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

EPOCH = 25
for epoch in range(EPOCH):
    running_loss = 0.0
    for data in trainloader:
        tokens_tensors, segments_tensors, masks_tensors, \
        labels = [t.to(device) for t in data]
        # 將參數梯度歸零
        optimizer.zero_grad()
        # forward pass
        outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors,\
            attention_mask=masks_tensors, labels=labels)
        loss = outputs[0]
        # backward
        loss.backward()
        optimizer.step()
        # 紀錄當前batch loss
        running_loss += loss.item()
    #計算分類準確率
    _, acc = get_predictions(model, trainloader, compute_acc=True)
    print('[epoch %d] loss: %.3f, acc: %.3f' %(epoch+1, running_loss, acc))

end_time = time.time()
print('execute_time: ', str(end_time-start_time))
# model.save_pretrained('ll_ncl_ps_albert_base')
# model.save_pretrained('ll_ncl_ps_x_albert_base')
# model.save_pretrained('ll_ncl_others_ps_albert_base')
# model.save_pretrained('ll_ncl_others_ps_x_albert_base')
model.save_pretrained('mygopen_albert_base')

### testset
# testset = LlNclPs('test', tokenizer=tokenizer)
# testset = LlNclPsX('test', tokenizer=tokenizer)
# testset = LlNclOthersPs('test', tokenizer=tokenizer)
# testset = LlNclOthersPsX('test', tokenizer=tokenizer)
testset = Mygopen('test', tokenizer=tokenizer)
y_test_origin = testset.df.iloc[:,0].values
testloader =  DataLoader(testset, batch_size=32, collate_fn=create_mini_batch)
# model = AlbertForSequenceClassification.from_pretrained('ll_ncl_ps_albert_base')
# model = AlbertForSequenceClassification.from_pretrained('ll_ncl_ps_x_albert_base')
# model = AlbertForSequenceClassification.from_pretrained('ll_ncl_others_ps_albert_base')
# model = AlbertForSequenceClassification.from_pretrained('ll_ncl_others_ps_x_albert_base')
model = AlbertForSequenceClassification.from_pretrained('mygopen_albert_base')
prediction, test_acc = get_predictions(model, testloader, compute_acc=True)

print("======================================")
print('test accuracy: ',test_acc)
# f1_score
print('f1 macro: ',f1_score(y_test_origin, prediction.tolist(), average='macro'))
print('f1 micro: ',f1_score(y_test_origin, prediction.tolist(), average='micro'))
print('f1 weighted: ',f1_score(y_test_origin, prediction.tolist(), average='weighted'))
# precision_score
print('precision macro: ', precision_score(y_test_origin, prediction.tolist(), average='macro'))
print('precision micro: ', precision_score(y_test_origin, prediction.tolist(), average='micro'))
print('precision weighted: ', precision_score(y_test_origin, prediction.tolist(), average='weighted'))
# recall_score
print('recall macro: ', recall_score(y_test_origin, prediction.tolist(), average='macro'))
print('recall micro: ', recall_score(y_test_origin, prediction.tolist(), average='micro'))
print('recall weighted: ', recall_score(y_test_origin, prediction.tolist(), average='weighted'))