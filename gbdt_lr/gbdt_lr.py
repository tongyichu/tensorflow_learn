#coding=utf-8
"""
usage:训练GBDT树并使用其讲数据转换成新的特征向量，用于训练Logistic Regression
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from gbdt_reg import GBDT
from sklearn.linear_model import LogisticRegression
import numpy as np

class Config(object):
    learningRate=0.1
    maxTreeLength=5
    maxLeafCount=30
    maxTreeNum=50

def generate_data():
    #通过make_classification函数自动生成一批数据，每个X是20维的向量，可以通过n_features=xx来调整维度
    X, y = make_classification(n_samples=1000)
    #生成训练/测试数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    #对于训练数据，前一半作为训练GBDT，后一半用来训练LR
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)
    return X_train, X_train_lr, y_train, y_train_lr,X_test, y_test




def main():
    X_train, X_train_lr, y_train, y_train_lr,X_test, y_test=generate_data()
    config=Config()
    gbdt=GBDT(config=config)
    gbdt.buildGbdt(X_train,y_train)
    trainDataFeatures=gbdt.generateFeatures(X_train_lr)
    testDataFeatures=gbdt.generateFeatures(X_test)
    print(len(trainDataFeatures[0]))
    lrModel = LogisticRegression()
    lrModel.fit(trainDataFeatures,y_train_lr)
    #test model
    testLabel = lrModel.predict(testDataFeatures)
    accuracy = np.sum((np.array(testLabel)==np.array(y_test)))*1.0/len(y_test)
    print ("the accuracy is % f"%accuracy)

if __name__=='__main__':
    main()






