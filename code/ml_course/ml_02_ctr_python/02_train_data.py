#!/user/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
##导入XGB相关的库
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.externals import joblib
import time

##接下来对特征进行处理，先将类别特征进行编码
#针对类型类的特征，先进行编码，编码之前构建字典
def label_encode(field,df):
    dic = []
    df_field = df[field]
    list_field = df_field.tolist()

    #构建field字典
    for i in list_field:
        if i not in dic:
            dic.append(i)

    label_field = preprocessing.LabelEncoder()
    label_field.fit(dic)

    df_field_enconde_tmp = label_field.transform(df_field)
    df_field_enconde = pd.DataFrame(df_field_enconde_tmp, index=df.index, columns=[(field+'_enconde')])
    return df_field_enconde

#数据准备
def encode_data():
    # 读入我们切割好的训练文件
    file = './data/df_train.csv'
    df = pd.read_csv(file)

    print(f'--All data:{df.id.count()}')
    y_1_nums = df[df["click"] == 1].id.count()
    y_0_nums = df[df["click"] == 0].id.count()
    print(f'--1 data:{y_1_nums}')
    print(f'--0 data:{y_0_nums}')
    print(f'--0 VS 1 => {round(y_0_nums / y_1_nums, 2)}:1')

    # 特征编码
    df_site_id_enconde = label_encode('site_id', df)
    df_site_domain_enconde = label_encode('site_domain', df)
    df_site_category_enconde = label_encode('site_category', df)
    df_app_id_enconde = label_encode('app_id', df)
    df_app_domain_enconde = label_encode('app_domain', df)
    df_app_category_enconde = label_encode('app_category', df)
    df_device_id_enconde = label_encode('device_id', df)
    df_device_ip_enconde = label_encode('device_ip', df)
    df_device_model_enconde = label_encode('device_model', df)

    #特征拼接
    df_input = pd.concat([df[['click','banner_pos','device_type','device_conn_type'
                              ,'C1','C14','C15','C16','C17','C18','C19','C20','C21']]
                          ,df_site_id_enconde
                          ,df_site_domain_enconde
                          ,df_site_category_enconde
                          ,df_app_id_enconde
                          ,df_app_domain_enconde
                          ,df_app_category_enconde
                          ,df_device_id_enconde
                          ,df_device_ip_enconde
                          ,df_device_model_enconde], axis=1)

    return df_input

#效果输出函数
def func_print_score(x_data, y_data, data_type, model_x):
    y_pred = model_x.predict(x_data)
    print(f'==============({data_type})===================')
    confusion = metrics.confusion_matrix(y_data, y_pred)
    print(confusion)
    print('------------------------')
    auc = metrics.roc_auc_score(y_data, y_pred)
    print(f'AUC: {auc}')
    print('------------------------')
    accuracy = metrics.accuracy_score(y_data, y_pred)
    print(f'Accuracy: {accuracy}')
    print('------------------------')
    report = metrics.classification_report(y_data, y_pred)
    print(report)
    print('=============================================')

#训练模型
def train_data(df_input):
    #对数据进行分割，分割为训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(df_input.iloc[:,1:],df_input["click"],test_size=0.3, random_state=123)

    begin_time = time.time()
    print(f'Begin Time : {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(begin_time))}')

    ##受限于机器的资源，这里就不做gridsearch调参了，直接凑合着来(按最小资源消耗来设置参数)
    model = XGBClassifier(learning_rate=0.1
                         ,n_estimators=10
                         ,max_depth=3
                         ,objective='binary:logistic'
                         )

    model.fit(x_train, y_train, eval_metric="auc")
    end_time = time.time()
    print(f'End Time : {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))}')
	return model

if __name__=='__main__':
    df_input=encode_data()
    model=train_data(df_input)

    #保存模型
    joblib.dump(model, './model/xgb_model.pkl')
