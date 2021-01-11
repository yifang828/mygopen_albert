from keras.utils import np_utils
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

train_pd = pd.read_excel('./data/cls_token_avg/train_ngram_ll_ncl_ps.xlsx', engine='openpyxl')
test_pd = pd.read_excel('./data/cls_token_avg/test_ngram_ll_ncl_ps.xlsx', engine='openpyxl')

y_train_origin = train_pd['label']
y_train = np_utils.to_categorical(y_train_origin)
x_train = train_pd.drop(['label'], axis=1)

train_combine = []
# for c, pred in zip(x_train['bigram_cls'], x_train['bigram_predict_feature']):
# for c, pred in zip(x_train['bigram_token_avg'], x_train['bigram_predict_feature']):
# for c, pred in zip(x_train['trigram_cls'], x_train['trigram_predict_feature']):
for c, pred in zip(x_train['trigram_token_avg'], x_train['trigram_predict_feature']):
    token_np = np.fromstring(c[1:-1], dtype=np.float, sep=' ')
    pred_np = np.fromstring(pred[1:-1], dtype=np.float, sep=' ')
    train_combine.append(np.concatenate((token_np, pred_np)))
    # train_combine.append(token_np)

y_test_origin = test_pd['label']
y_test = np_utils.to_categorical(y_test_origin)
x_test = test_pd.drop(['label'], axis=1)

test_combine = []
# for c, pred in zip(x_test['bigram_cls'], x_test['bigram_predict_feature']):
# for c, pred in zip(x_test['bigram_token_avg'], x_test['bigram_predict_feature']):
# for c, pred in zip(x_test['trigram_cls'], x_test['trigram_predict_feature']):
for c, pred in zip(x_test['trigram_token_avg'], x_test['trigram_predict_feature']):
    token_np = np.fromstring(c[1:-1], dtype=np.float, sep=' ')
    pred_np = np.fromstring(pred[1:-1], dtype=np.float, sep=' ')
    test_combine.append(np.concatenate((token_np, pred_np)))
    # test_combine.append(token_np)

# train_np = np.array(train_combine)
# test_np = np.array(test_combine)
train_np =  MinMaxScaler().fit_transform(np.array(train_combine))
test_np =  MinMaxScaler().fit_transform(np.array(test_combine))
# '''
# 建立模型
callback = EarlyStopping(monitor='loss', patience=3)
model = Sequential()
model.add(Dense(units=256, input_dim=771, kernel_initializer=initializers.TruncatedNormal(stddev=0.02), activation='relu'))
model.add(Dense(units=2, kernel_initializer=initializers.TruncatedNormal(stddev=0.02), activation='softmax'))
print(model.summary())

# 訓練模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
history = model.fit(x=train_np, y=y_train, validation_split=0.2, batch_size=4, epochs=50, verbose=2, callbacks=[callback])
model.save('./model/ngram/token_trigram_ll_ncl_ps.h5')

def show_train_history(history, train, validation):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(history, 'loss', 'val_loss')

# 評估準確率
print("======================================")
print('accuracy: ', model.evaluate(test_np, y_test)[1])
# 進行預測
prediction = model.predict_classes(test_np)
# confusion matrix
print(pd.crosstab(y_test_origin, prediction, rownames=['label'], colnames=['predict']))
# f1_score
# print('f1 macro: ',f1_score(y_test_origin, prediction, average='macro'))
# print('f1 micro: ',f1_score(y_test_origin, prediction, average='micro'))
print('f1 weighted: ',f1_score(y_test_origin, prediction, average='weighted'))
# precision_score
# print('precision macro: ', precision_score(y_test_origin, prediction, average='macro'))
# print('precision micro: ', precision_score(y_test_origin, prediction, average='micro'))
print('precision weighted: ', precision_score(y_test_origin, prediction, average='weighted'))
# recall_score
# print('recall macro: ', recall_score(y_test_origin, prediction, average='macro'))
# print('recall micro: ', recall_score(y_test_origin, prediction, average='micro'))
print('recall weighted: ', recall_score(y_test_origin, prediction, average='weighted'))
# '''