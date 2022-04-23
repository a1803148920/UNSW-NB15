import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# 读取资料集
dataframe1 = pd.read_csv("UNSW-NB15_1.csv", header=None)
dataframe2 = pd.read_csv("UNSW-NB15_2.csv", header=None)
dataframe3 = pd.read_csv("UNSW-NB15_3.csv", header=None)
dataframe4 = pd.read_csv("UNSW-NB15_4.csv", header=None)

# 合并成一个资料集
dataframe = pd.concat([dataframe1, dataframe2, dataframe3, dataframe4])

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
                     'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
                     'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt',
                     'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
                     'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
                     'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']

# 删除"源IP地址"和"目标IP地址"两列 (官方给的训练集中没有这两列)
dataframe = dataframe.drop(['srcip', 'dstip'], 1)
print(dataframe.head())

# 拆分训练集、验证集和测试集
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), '训练集数')
print(len(val), '验证集数')
print(len(test), '测试集数', '\n')


def df_to_dataset(df, shuffle=True, batch_size=32):
    """将pandas的df转换成dataset"""

    labels = df.pop('label')  # 取出target列
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))  # 转成元组，给出特征数据和特征标签

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))  # 全部数据随机排序

    ds = ds.batch(batch_size)  # 按照顺序每次取出32行数据，最后一次输出可能小于batch

    return ds


train_ds = df_to_dataset(train, shuffle=True)
val_ds = df_to_dataset(val, shuffle=False)
test_ds = df_to_dataset(test, shuffle=False)

"""feature_column特征处理"""
feature_columns = []

# 数值列
for header in ['sport', 'dsport', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'sload', 'dload',
               'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
               'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt',
               'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
               'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
               'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']:
    feature_columns.append(feature_column.numeric_column(header))

# 分类列
# proto One-Hot处理
proto = feature_column.categorical_column_with_vocabulary_list('proto',
                                                               ['udp', 'arp', 'tcp', 'ospf', 'icmp', 'igmp', 'sctp',
                                                                'udt', 'sep', 'sun-nd', 'swipe', 'mobile', 'pim', 'rtp',
                                                                'ipnip', 'ip', 'ggp', 'st2', 'egp', 'cbt', 'emcon',
                                                                'nvp', 'igp', 'xnet', 'argus', 'bbn-rcc', 'chaos',
                                                                'pup', 'hmp', 'mux', 'dcn', 'prm', 'trunk-1',
                                                                'xns-idp', 'trunk-2', 'leaf-1', 'leaf-2', 'irtp', 'rdp',
                                                                'iso-tp4', 'netblt', 'mfe-nsp', 'merit-inp', '3pc',
                                                                'xtp', 'idpr', 'tp++', 'ddp', 'idpr-cmtp', 'ipv6', 'il',
                                                                'idrp', 'ipv6-frag', 'sdrp', 'ipv6-route', 'gre',
                                                                'rsvp', 'mhrp', 'bna', 'esp', 'i-nlsp', 'narp',
                                                                'ipv6-no', 'tlsp', 'skip', 'ipv6-opts', 'any', 'cftp',
                                                                'sat-expak', 'kryptolan', 'rvd', 'ippc', 'sat-mon',
                                                                'ipcv', 'visa', 'cpnx', 'cphb', 'wsn', 'pvp',
                                                                'br-sat-mon', 'wb-mon', 'wb-expak', 'iso-ip',
                                                                'secure-vmtp', 'vmtp', 'vines', 'ttp', 'nsfnet-igp',
                                                                'dgp', 'tcf', 'eigrp', 'sprite-rpc', 'larp', 'mtp',
                                                                'ax.25', 'ipip', 'micp', 'aes-sp3-d', 'encap',
                                                                'etherip'])
proto_one_hot = feature_column.indicator_column(proto)
feature_columns.append(proto_one_hot)

# 嵌入列
# proto Embedding处理
proto_embedding = feature_column.embedding_column(proto, dimension=4)
feature_columns.append(proto_embedding)

# 分类列
# state One-Hot处理
state = feature_column.categorical_column_with_vocabulary_list('state',
                                                               ['con', 'int', 'fin', 'urh', 'req', 'eco', 'rst', 'clo',
                                                                'txd', 'urn', 'no', 'acc', 'par', 'mas', 'tst', 'ecr'])
state_one_hot = feature_column.indicator_column(state)
feature_columns.append(state_one_hot)

# 嵌入列
# state Embedding处理
state_embedding = feature_column.embedding_column(state, dimension=4)
feature_columns.append(state_embedding)

# 分类列
# service One-Hot处理
service = feature_column.categorical_column_with_vocabulary_list('service',
                                                                 ['dns', '-', 'http', 'smtp', 'ftp-data', 'ftp', 'ssh',
                                                                  'pop3', 'snmp', 'ssl', 'irc', 'radius', 'dhcp'])
service_one_hot = feature_column.indicator_column(service)
feature_columns.append(service_one_hot)

# 嵌入列
# service Embedding处理
service_embedding = feature_column.embedding_column(service, dimension=4)
feature_columns.append(service_embedding)

# 分类列
# attack_cat One-Hot处理
attack_cat = feature_column.categorical_column_with_vocabulary_list('attack_cat',
                                                                    ['normal', 'exploits', 'reconnaissance', 'dos',
                                                                     'generic', 'shellcode', 'fuzzers', 'worms',
                                                                     'backdoor', 'analysis'])
attack_cat_one_hot = feature_column.indicator_column(attack_cat)
feature_columns.append(attack_cat_one_hot)

# 嵌入列
# attack_cat Embedding处理
attack_cat_embedding = feature_column.embedding_column(attack_cat, dimension=4)
feature_columns.append(attack_cat_embedding)

"""创建、编译和训练模型"""
model = tf.keras.Sequential([
    layers.DenseFeatures(feature_columns),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=300
)

# 测试集
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
