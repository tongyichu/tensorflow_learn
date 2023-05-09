## 目录介绍
该目录下的文件拷贝自https://github.com/garfieldsun/recsys/tree/master

并在该代码基础上做了少量修改，使其可以在tensorflow2上运行。

## 文件介绍
1. fm_numpy是ipython文件，单步执行，具体看fm是如何进行预测的
2. fm_tensorflow_movielens文件使用movielens100k的数据进行预测，使用了最基本的one-hot编码
3. fm文件是用自己的数据集就行实验，因为自己的数据集特征太多，直接使用one-hot编码运行时会过载，因此对特征建立字典进行提取

## 运行环境
1. Python 3.9.2
2. tensorflow 2.12.0
3. MaxOs Big Sur 11.2.3