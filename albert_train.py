from transformers import *
import torch

PRETRAINED_MODEL_NAME = 'voidful/albert_chinese_base'
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
model = AlbertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME)

vocab = tokenizer.vocab
print('字典大小:', len(vocab))

