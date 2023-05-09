"""
采用Auto MPG数据集，通过两层全连接神经网络来实现一个回归模型。
参考文章 https://subaochen.github.io/tensorflow/2019/07/24/basic_regression/

    Auto MPG数据集是UCI Machine Learning Repository的476个数据集之一，包含了398条记录，结构如下：
        mpg: 百公里油耗
        cylinders: 气缸数
        displacement: 排气量
        horsepower: 马力
        weight: 车重
        acceleration: 加速能力
        model year: 年份
        origin: 原产地
        car name: 车型名称

这个模型最终的效果是：当输入cylinders、displacement、horsepower、weight、acceleration、model year、origin时预测出mpg值
"""
import tensorflow as tf
import pandas as pd
import seaborn as sns
import plt_util
from tensorflow import keras
from tensorflow.keras import layers

#Step_1 获取数据
dataset_path = 'auto-mpg.data'

#使用pandas读取数据
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                         na_values='?', comment='\t',
                         sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
print('原始数据')
print(dataset.tail())

#Step_2 数据清洗
dataset = dataset.dropna() #删除所有包含NaN的行
origin = dataset.pop('Origin') #获取Origin那一列数据,将该列数据扩展为3列：USA Europe Japan
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print('Origin扩展')
print(dataset.tail())

#Step_3 划分训练集和测试集
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
#观察训练集中几对列的联合分布,并保存对比图片
# fig = sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
# fig.savefig('/Users/habo/Desktop/test.png',dpi=400)

#Step_4 统计数据
train_stats = train_dataset.describe()
print('统计数据\n')
print(train_stats)
train_stats.pop("MPG")
train_stats = train_stats.transpose() #转置是为了方便后面”标准化“函数的处理
print('统计数据转置\n')
print(train_stats)

#Step_5 取出标签
'''
    标签（label）是训练模型的“指导”或者“准星”，
    根据label训练模型的目的就是在预测时，能够尽量逼近label。
    这里直接将train_dataset/test_dataset中的MPG列pop出来即可
'''
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

#对数据进行标准化处理的函数定义
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

#Step_6 标准化数据
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
print('数据标椎化\n')
print(normed_train_data.tail())

#Step_7 定义模型
'''
    这里使用了高阶的keras API构建模型，在这个简单的模型中只包含了两个全连接隐藏层和一个全连接输出层，
    其中输入层的shape为(len(train_dataset.keys()),)，第一个全连接层的输出shape为(None, 64)，
    第二个全连接层的输出为(None, 64)。其中，None的大小由BATCH_SIZE决定。
'''
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                 optimizer=optimizer,
                 metrics=['mae', 'mse'])
    return model
#构建模型
model = build_model()
#打印模型的简单描述
model.summary()


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

#Step_8 训练模型
'''
    训练轮次(epochs)为1000,训练和验证的准确率记录在history中了,方便通过图形化显示准确率的变动情况。
    fit函数的参数validation_split表示将训练数据自动拿出指定比例的数据作为验证数据集,验证数据集是不参与训练的。
'''
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(
  normed_train_data, train_labels,
  epochs=1000, validation_split = 0.2, verbose=0,
  callbacks=[early_stop, PrintDot()])

#训练结束后，通过图形化展示history对象中存储的训练过程
plt_util.plot_history(history)
###########################################################
#到此处，模型已经训练结束，以下代码是对模型的测试和应用
#用测试数据对模型进行测试
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("\nTesting set Mean Abs Error: {:5.2f} MPG".format(mae))

#用测试数据对模型进行预测
'''
    用测试数据集进行预测，看一下预测的准确率如何？预测的方法是使用测试数据集调用模型的predict方法，
    将预测结果和测试数据集的label集合对比，即可以看出预测的效果了。
    这里采取的方法是，将测试集的label和预测值构造为一个（x,y）点显示在二维坐标上，
    于是所有的(x,y)如果集中在45度角的直线附近的话，证明预测的准确率比较高。
'''
test_predictions = model.predict(normed_test_data).flatten()
plt_util.plot_predict(test_labels,test_predictions)