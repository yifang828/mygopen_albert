# mygopen_albert

## Experi3_BERT_CLS different activate function 2021/01/11
bigram and trigram

## Experi2_BERT_CLS different activate function 2021/01/03
原先在model_mlp.py中使用relu當activate function
根據bert原始碼 https://github.com/google-research/bert/blob/master/modeling.py
pooler是使用tanh
sequence_output是使用gelu
所以實驗
1.使用tanh當activation
結論:沒有差很多
## Experi2_BERT_CLS
1. 取BERT最後一層CLS與Label feature聯集，跑mlp
2. 取一篇文章last_hidden_state平均與Lable feature聯集，跑mlp

### gen_mygopen_dataset.py
mygopen爬回的資料./data/transfer/rumor.json中每篇的truth,source和./data/transfer/truth.json中source
如果資料長度大於510則取資料前511個字(BERT本身限制512還要扣除[CLS])匯集起來寫到mygopen_train.xlsx, mygopen_test.xlsx

### mygopen.py
讀取gen_mygopen_dataset.py產生的mygopen_train.xlsx, mygopen_test.xlsx轉換成BERT接受格式(每一個句子前標註CLS)

### mygopen_dataset.py
做mygopen遷移式學習用，model_predict.py用 *_albert_base模型轉換mygopen rumor.json, truth.json 成對應的 ll, nls, ps, x 

### albert_train.py
做albert fine-tune

### gen_cls_and_token_avg.py
測試從BERT模型中取出last_hidden_state(batch_size, seq_len, hidden_size), pooler_output(batch_size, hidden_size)

### model_predict.py
Do mygopen transfer learning with fine-tuned albert model

### model_mlp.py
mlp neural network

### model_lstm.py
lstm neural network
