import json
import re

# test
# str = '原來竹子會結果？一般草本植物每年都會開花結果，但是竹子卻不同，從五十年到一百二十年不等，\
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
    with open('data/transfer_learning/rumor.json', encoding='utf-8') as jf:
        datas = json.load(jf)
        result = {}
        count = 0
        for data in datas:
            for k, v in data.items():
                if k == "source":
                    result[count] = v
                    count +=1
                # if k == "truth":
                #     result[count] = v
                #     count +=1
        return result

context = getContext()
for k, v in context.items():
    context[k] = splitContext(v)
# print("context size: ", len(context))
# print(context)