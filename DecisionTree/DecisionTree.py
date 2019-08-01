import numpy as np
import math
import json
from collections import Counter
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],         #数据集
                [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']#分类属性
    return dataSet, labels                #返回数据集和分类属性
#data:分类的列表
def calcShannonEnt(data):
    cot=Counter(list(data))
    total=len(data)
    H=0
    for v in cot.values():
        H+=v/total*math.log(v/total,2)
    return -H

def chooseBestFeatureToSplit_index(dataSet):
    #计算信息增益，选择最优特征
    #1.划分特征
    #2.每列划分出不同类别
    #3.每个类别对应的Y特征数量
    row_len,column_len=dataSet.shape
    label_col=dataSet[:,-1]
    Hd=calcShannonEnt(label_col)
    gs=[]#用于存储对应特征的增益
    for col in range(0,column_len-1):
        features=dataSet[:,col]
#         if is_Continuous_values(features):
#             #是连续数值型
#             pass
        cot=Counter(features)
        g=0
        for cate,v in cot.items():
            #该特征下为cate的所有行
            cate_rows=np.where(features==cate)
            #某类在对应Y的香农熵
            Hi=calcShannonEnt(label_col[cate_rows])
            g+=v/row_len*Hi
        Gain=Hd-g
        gs.append(Gain)
    #ID3算法
    max_Ent=max(gs)
    index=gs.index(max_Ent)#返回最大信息增益的下标
    #C4.5算法
    gs_array=np.array(gs)
    gs_mean=np.mean(gs_array)
    row_list=np.where(gs_array>gs_mean)
    gs_RTmean_array=gs_array[row_list]
    max_ix=index
    max_Gain_radio=0
    for num in list(gs_RTmean_array):
        ix=gs.index(num)
        IV=calcShannonEnt(dataSet[:,ix])
        Gain_radio=num/IV
        if Gain_radio>max_Gain_radio:
            max_ix=ix
            max_Gain_radio=Gain_radio
    index=max_ix
    return index
def createTree(dataSet,labels,cates_list):
    row_len,column_len=dataSet.shape
    data_cates=dataSet[:,-1]
    cates=Counter(data_cates).most_common()
    if len(cates)==1:
        #如果所有训练集都是一个分类,则设置为当前类别
        return cates[0][0]
    best_feature_index=chooseBestFeatureToSplit_index(dataSet)
    best_label=labels[best_feature_index]
    best_feature_array=dataSet[:,best_feature_index]
    #切割datsSet，labels
    split_dataSets=np.delete(dataSet,best_feature_index,axis=1)
    split_labels=labels.copy()
    split_labels.pop(best_feature_index)
    split_cates_list=cates_list.copy()
    split_cates_list.pop(best_feature_index)
    #
    tree={best_label:{}}
    for best_feature_cate in cates_list[best_feature_index]:
        cate_row_indexs=np.where(best_feature_array==best_feature_cate)#对应类别的行标
        split_dataSet=split_dataSets[list(cate_row_indexs[0]),:]
        if split_dataSet.size==0:
            #该分类的数据集为空，则设置为其父结点的最大分类
            tree[best_label][best_feature_cate]=cates[0][0]
        else:
            tree[best_label][best_feature_cate]=createTree(split_dataSet,split_labels,split_cates_list)
    return tree

#得到每一个特征的分类:[['a','b'],['0','1'],['0','1','2']]
def Elem2cates(dataSet):
    row,col=dataSet.shape
    cate_list=[]
    for j in range(col):
        cates=[]
        for cate in set(dataSet[:,j]):
            cates.append(cate)
        cate_list.append(cates)
    return cate_list

#存入树
def storeTree(tree,filename):
    with open(filename,'w') as f:
        f.write(json.dumps(tree))

#读取树
def loadTree(filename):
    with open(filename,'r') as f:
        tree=f.read()
    return json.loads(tree)

#转换数据，并创建和存储树
def Data2Tree(data,label,filename='tree.txt'):
    data=np.array(data)
    cates_list=Elem2cates(data)
    tree=createTree(data,label,cates_list)
    storeTree(tree,filename)
    print('成功生成并存储决策树：',tree)


def classify(tree, label, testvec):
    # isinstance (a,int)
    while isinstance(tree, dict):
        # 取标签名
        items = list(tree.items())[0]
        feature_index = label.index(items[0])
        value = str(testvec[feature_index])
        tree = items[1][value]
    if tree == 'yes':
        print('放贷')
    elif tree == 'no':
        print('不放贷')

data,label=createDataSet()
Data2Tree(data,label)
featLabels = ['有自己的房子', '有工作']
vec = [0, 1]
tree = loadTree('tree.txt')
classify(tree, featLabels, vec)
