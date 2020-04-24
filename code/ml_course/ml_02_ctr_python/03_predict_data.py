#!/user/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn import metrics
from xgboost import XGBClassifier

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
    # 读入我们切割好的预测文件
    file = './data/df_predict.csv'
    df = pd.read_csv(file)

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

    #特征拼接，注意，这里在实际中是没有click数据的
    df_input = pd.concat([df[['id','click','banner_pos','device_type','device_conn_type'
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

if __name__=='__main__':
    df_input=encode_data()
    #加载模型
    model = joblib.load('./model/xgb_model.pkl')
    #预测数据
    #这里我们先把id和原来的click摘出来，然后一会儿predict出来之后，用来对比一下预测结果
    df_id_click=df_input.iloc[:,:2]
    df_predict = df_input.iloc[:, 2:]
    y_predict = model.predict(df_predict)
    #将序列转换成df，原始是这样的array([0, 0, 0, ..., 0, 0, 0])，并字段命名为p_click
    df_y_predict = pd.DataFrame(y_predict, columns=["p_click"], index=df_id_click.index)
    #我们先把预测数据保存下来,在实际生产过程中，这个结果就可以拿去用了
    df_output=pd.concat([df_id_click.iloc[:,:1], df_y_predict], axis=1)
    df_output.to_csv('./predict/predict.csv')

    #为了验证我们的预测结果，我们看下多少个完全预测对了
    #合并新旧数据，click为真实的click，p_predict为预测的click
    df_new_id = pd.concat([df_id_click, df_y_predict], axis=1)
    df_new_id.to_csv('./predict/predict_pk.csv')
    #我们适当打印一下差异
    print(f'ALL ID: {df_id_click.id.count()}')
    print(f'Disaccord ID: {df_new_id[df_new_id["click"] != df_new_id["p_click"]].id.count()}')


