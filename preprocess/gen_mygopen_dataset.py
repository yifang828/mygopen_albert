import torch
import pandas as pd
import json
import xlsxwriter

def getContextFromPath(path, mode):
    result = {}
    with open(path, encoding='utf-8') as jf:
        datas = json.load(jf)
        labellist = []
        contextlist = []
        count = 0
        for data in datas:
            print('================start :',count)
            for k, v in data.items():
                if mode=='rumor' and  k == "truth" and v !="":
                    labellist.append(1)
                    contextlist.append(v[:510])
                elif mode=='rumor' and k == "source" and v !="":
                    labellist.append(0)
                    contextlist.append(v[:510])
                elif mode=='truth' and k == "source" and v !="":
                    labellist.append(1)
                    contextlist.append(v[:510])
            count +=1
        result['label'] = labellist
        print('label size: ', len(labellist))
        result['context'] = contextlist
        print('context size: ', len(contextlist))
    return result

RUMOR_PATH='./data/transfer_learning/rumor.json'
TRUTH_PATH='./data/transfer_learning/truth.json'
rumor_context = getContextFromPath(RUMOR_PATH, 'rumor')
truth_context = getContextFromPath(TRUTH_PATH, 'truth')

all_context = {'label': rumor_context['label']+truth_context['label'], \
    'context': rumor_context['context']+truth_context['context']}
df = pd.DataFrame.from_dict(all_context)
train_df = df.sample(frac=0.8)
print('train label size: ',train_df.groupby('label').size())
test_df = df.drop(train_df.index)
print('test label size: ',test_df.groupby('label').size())
train_df.to_excel('./data/mygopen_train.xlsx', engine='xlsxwriter', index=False)
test_df.to_excel('./data/mygopen_test.xlsx', engine='xlsxwriter', index=False)