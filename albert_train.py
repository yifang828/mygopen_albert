from transformers import BertTokenizer, AlbertForSequenceClassification
import torch
from preprocess.ll_ncl_ps_dataset import LlNclPs

PRETRAINED_MODEL_NAME = 'voidful/albert_chinese_base'
NUM_LABELS = 3

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
model = AlbertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

vocab = tokenizer.vocab
print('字典大小:', len(vocab))

