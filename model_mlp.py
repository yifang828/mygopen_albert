from keras.utils import np_utils
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

train_rumor = pd.read_excel('./data/transfer_learning/rumor_ll_ncl_others_ps_x.xlsx', engine='openpyxl')
train_truth = pd.read_excel('./data/transfer_learning/truth_ll_ncl_others_ps_x.xlsx', engine='openpyxl')
# train_rumor_truth = pd.read_excel('./data/transfer_learning/truth_rumor_ll_ncl_others_ps_x.xlsx', engine='openpyxl')

test_rumor = pd.read_excel('./data/transfer_learning/TEST_rumor_ll_ncl_others_ps_x.xlsx', engine='openpyxl')
test_truth = pd.read_excel('./data/transfer_learning/TEST_truth_ll_ncl_others_ps_x.xlsx', engine='openpyxl')
# test_rumor_truth = pd.read_excel('./data/transfer_learning/TEST_truth_rumor_ll_ncl_others_ps_x.xlsx', engine='openpyxl')

# 加一欄label 0:謠言, 1:truth
train_rumor['label'] = 0
train_truth['label'] = 1
# train_rumor_truth['label'] = 1

test_rumor['label'] = 0
test_truth['label'] = 1
# test_rumor_truth['label'] = 1

# 合併rumor and truth
# df_train = pd.concat([train_rumor, train_truth, train_rumor_truth], axis = 0, ignore_index= True)
df_train = pd.concat([train_rumor, train_truth], axis = 0, ignore_index= True)

# 最大最小正規化
df_train = shuffle(df_train)
y_train_origin = df_train['label']
y_train = np_utils.to_categorical(y_train_origin)
x_train = df_train.drop(['label'], axis=1)
x_train = MinMaxScaler().fit_transform(x_train)

# 畫圖觀察值的分布
train_num = y_train.shape[0]
print(df_train.head())
print(df_train['LL'].head())
sns.regplot(x=df_train['LL'], y=y_train_origin)
plt.show()

# df_test = pd.concat([test_rumor, test_truth, test_rumor_truth], axis = 0, ignore_index= True)
df_test = pd.concat([test_rumor, test_truth], axis = 0, ignore_index= True)
df_test = shuffle(df_test)
y_test_origin = df_test['label']
y_test = np_utils.to_categorical(y_test_origin)
x_test = df_test.drop(['label'], axis=1)
x_test = MinMaxScaler().fit_transform(x_test)
# 建立模型
callback = EarlyStopping(monitor='loss', patience=3)
model = Sequential()
model.add(Dense(units=256, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=2, kernel_initializer='normal', activation='softmax'))
print(model.summary())

# 訓練模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
history = model.fit(x=x_train, y=y_train, validation_split=0.2, batch_size=4, epochs=50, verbose=2, callbacks=[callback])
model.save('./model/mygopen_ll_ncl_others_ps_x_small.h5')

def show_train_history(history, train, validation):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(history, 'loss', 'val_loss')
# show_train_history(history, 'acc', 'val_acc')

# 評估準確率
print("======================================")
print('accuracy: ', model.evaluate(x_test, y_test)[1])
# 進行預測
prediction = model.predict_classes(x_test)
# confusion matrix
print(pd.crosstab(y_test_origin, prediction, rownames=['label'], colnames=['predict']))
# f1_score
print('f1 macro: ',f1_score(y_test_origin, prediction, average='macro'))
print('f1 micro: ',f1_score(y_test_origin, prediction, average='micro'))
print('f1 weighted: ',f1_score(y_test_origin, prediction, average='weighted'))
# precision_score
print('precision macro: ', precision_score(y_test_origin, prediction, average='macro'))
print('precision micro: ', precision_score(y_test_origin, prediction, average='micro'))
print('precision weighted: ', precision_score(y_test_origin, prediction, average='weighted'))
# recall_score
print('recall macro: ', recall_score(y_test_origin, prediction, average='macro'))
print('recall micro: ', recall_score(y_test_origin, prediction, average='micro'))
print('recall weighted: ', recall_score(y_test_origin, prediction, average='weighted'))