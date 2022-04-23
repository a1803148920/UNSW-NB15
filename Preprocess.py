import pandas as pd
import numpy as np

from sklearn import preprocessing

# 读取资料集
dataframe1 = pd.read_csv("UNSW-NB15_1.csv", header=None)
dataframe2 = pd.read_csv("UNSW-NB15_2.csv", header=None)
dataframe3 = pd.read_csv("UNSW-NB15_3.csv", header=None)
dataframe4 = pd.read_csv("UNSW-NB15_4.csv", header=None)

# 合并成一个资料集
dataframe = pd.concat([dataframe1, dataframe2, dataframe3, dataframe4])
print(dataframe.head())

"""取出资料特征"""

feature_info = pd.read_csv("NUSW-NB15_features.csv", encoding="ISO-8859-1", header=None).values
features = feature_info[1:, 1]  # 特征名称
feature_types = np.array([item.lower() for item in feature_info[1:, 2]])  # 特征的数据类型

# 按特征的数据类型分组 (输出为索引)
nominal_cols = np.where(feature_types == "nominal")[0]  # 名词
integer_cols = np.where(feature_types == "integer")[0]  # 整数
binary_cols = np.where(feature_types == "binary")[0]  # 二进制
float_cols = np.where(feature_types == "float")[0]  # 浮点数

# 将不同数据类型的特征分成组
nominal_feature = features[nominal_cols]
integer_feature = features[integer_cols]
binary_feature = features[binary_cols]
float_feature = features[float_cols]

"""处理资料集"""

# 每一行都转换为同一种数值类型，无效解析设置为NaN
dataframe[integer_cols] = dataframe[integer_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)
dataframe[binary_cols] = dataframe[binary_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)
dataframe[float_cols] = dataframe[float_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)

# 去除空值
# 第48列(攻击类别的名称): 1、把NaN替换为"normal"; 2、把"backdoors"替换成"backdoor"
dataframe.loc[:, 47] = dataframe.loc[:, 47].replace(np.nan, 'normal', regex=True).apply(
    lambda x: x.strip().lower())
dataframe.loc[:, 47] = dataframe.loc[:, 47].replace('backdoors', 'backdoor', regex=True).apply(
    lambda x: x.strip().lower())
# 数字列: 把NaN替换为0
dataframe.loc[:, integer_cols] = dataframe.loc[:, integer_cols].replace(np.nan, 0, regex=True)
dataframe.loc[:, binary_cols] = dataframe.loc[:, binary_cols].replace(np.nan, 0, regex=True)
dataframe.loc[:, float_cols] = dataframe.loc[:, float_cols].replace(np.nan, 0, regex=True)
# 名词列: 删除字符串前后空白，并将其改成小写
dataframe.loc[:, nominal_cols] = dataframe.loc[:, nominal_cols].applymap(lambda x: x.strip().lower())

# 修改列名
dataframe.columns = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
                     'sloss', 'dloss', 'service', 'sload', 'dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
                     'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt',
                     'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
                     'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
                     'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']

# 删除"源IP地址"和"目标IP地址"两列 (官方给的训练集中没有这两列)
dataframe = dataframe.drop(['srcip', 'dstip'], 1)

"""One-Hot 编码"""
onehot_proto = pd.get_dummies(dataframe['proto'])  # 135 col
onehot_proto = pd.get_dummies(dataframe['proto'], prefix='proto')  # 栏位名称加上前缀"proto"

onehot_state = pd.get_dummies(dataframe['state'])  # 16 col
onehot_state = pd.get_dummies(dataframe['state'], prefix='state')

onehot_service = pd.get_dummies(dataframe['service'])  # 13 col
onehot_service = pd.get_dummies(dataframe['service'], prefix='service')

dataframe = dataframe.drop(['proto', 'state', 'service'], 1)
dataframe = pd.concat([dataframe, onehot_proto, onehot_state, onehot_service], axis=1)  # 表格合并

"""取出X和Y"""
Y = dataframe['attack_cat']
Y_label = dataframe['label']
X = dataframe.drop(['attack_cat', 'label'], 1)
X = X.astype(np.float32)

"""资料分群"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y_label, test_size=0.1, random_state=0)

# 训练集做MinMax处理
minmax_scaler = preprocessing.MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X_train)

feature_names = [column for column in X_train]  # 取列名
X_minmax = pd.DataFrame(X_minmax)
X_minmax.columns = feature_names

# 保存资料集
# X_train.to_csv('Dataset/X_train.csv', index=False)
# X_minmax.to_csv('Dataset/X_minmax.csv', index=False)
# X_test.to_csv('Dataset/X_test.csv', index=False)

# y_train.to_csv('Dataset/y_train.csv', index=False)
# y_test.to_csv('Dataset/y_test.csv', index=False)
# Y_label.to_csv('Dataset/Y_label.csv', index=False)
