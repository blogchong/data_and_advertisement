import warnings
warnings.filterwarnings("ignore")
import pandas as pd

file = './data/train_subset_1000000.csv'
df = pd.read_csv(file)
##截取部分当成训练集

df_click_1 =  df[df["click"] == 1].iloc[:10000,:]
df_click_0 =  df[df["click"] == 0].iloc[:52400,:]
#然后合并回去（行合并），作为新的df_train
df_train=pd.concat([df_click_1, df_click_0])

##截取部分当成预测集
df_click_1 =  df[df["click"] == 1].iloc[10001:20000,:]
df_click_0 =  df[df["click"] == 0].iloc[52401:100000,:]
#然后合并回去（行合并），作为新的df_train
df_predict=pd.concat([df_click_1, df_click_0])

#保存数据
df_train.to_csv('./data/df_train.csv')
df_predict.to_csv('./data/df_predict.csv')
