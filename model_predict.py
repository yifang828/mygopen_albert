import time
import pandas as pd
import numpy as np
from keras.models import load_model
import json
from load_predict_data import context

# model = load_model('./classify_into_3labels_others.h5')
# results = []
# bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=512)

PRETRAINED_MODEL_NAME = 'voidful/albert_chinese_base'
NUM_LABELS = 3
results = []
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
model = AlbertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

init_time = time.time()
# 對文章的每個句子(split by ',' '。','!','?')預測
for keys, values in context.items():
    result = np.array([0, 0, 0, 0])
    print('keys: ', keys)
    for v in values:
        # 將句子轉向量
        vec = bert_model.encode([v])["encodes"][0]
        x_train = np.array([vec])
        # 執行預測
        predicted = model.predict(x_train)
        idx = np.argmax(predicted, axis=1)[0]
        result[idx] +=1
    results.append(result)

print('results size= ', len(results))
cost_time = time.time() - init_time
print("Average cost time: %s." % (cost_time/len(texts)))

arr = np.array(results)
df = pd.DataFrame({'LL':arr[:,0], 'NCL':arr[:,1], 'OTHERS':arr[:,2], 'PS':arr[:,3]})
df.to_excel('./data/transfer_learning/rumor_3labels_others.xlsx')