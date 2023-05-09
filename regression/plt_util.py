import matplotlib.pyplot as plt
import pandas as pd


def plot_predict(test_labels,test_predictions):
  plt.scatter(test_labels, test_predictions)
  plt.xlabel('True Values [MPG]')
  plt.ylabel('Predictions [MPG]')
  #plt.axis('equal')
  plt.axis('square') # 正方形，即ｘ轴和ｙ轴比例相同
  plt.xlim([0,plt.xlim()[1]])
  plt.ylim([0,plt.ylim()[1]])
  _ = plt.plot([0, 100], [0, 100]) # 参考的对角线
  plt.show()

  plt.plot(test_predictions,label='test prediction')
  plt.plot(test_labels.values,label='test true')
  plt.legend()
  plt.show()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

  # loss
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Loss [MPG]')
  plt.plot(hist['epoch'], hist['loss'],label='Train Loss')
  plt.plot(hist['epoch'], hist['val_loss'], label='Val Loss')
  plt.legend()
  plt.show()