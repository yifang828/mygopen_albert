import torch
import numpy as np
from transformers import BertTokenizer, AlbertModel

testString = '原來竹子會結果？一般草本植物每年都會開花結果，但是竹子卻不同，從五十年到一百二十年不等，\
    視不同品種的竹類而有所差異。'
print(len(testString))
PRETRAINED_MODEL_NAME = 'll_ncl_ps_albert_base'
NUM_LABELS = 3

tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_base')
model = AlbertModel.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

inputs = tokenizer(testString)
tokens_tensor = inputs['input_ids']
print('tokens_tensor: ',tokens_tensor)
input_ids = torch.tensor(tokens_tensor).unsqueeze(0) #batch size 1
print('input_ids: ',input_ids)

output = model(input_ids)
print('output: ', output)
print('last hidden state: ', output[0])
print('last hidden state size: ', output[0].size())
print('pooler output: ', output[1])
print('pooler output size: ', output[1].size())
print( output[1][0])
print('output[1][0]: ', output[1][0].size())