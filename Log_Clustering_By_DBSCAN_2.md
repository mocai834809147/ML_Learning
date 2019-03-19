
# 研究目标

Oracle Management Cloud借助强大的机器学习的能力，帮助客户从多个层面智能运维IT系统。OMC包含了IMCS、APMCS、LACS、ITACS等在内的七个子模块，分别从基础架构、应用系统、日志分析等多个方面全方位监控IT系统的健康情况。本文将借助sklearn包中的DBSCAN算法，实现LACS中的日志聚类功能。

## 数据准备

从Linux的system log中选取其中一个日志文件，进行聚类分析，并和LACS的聚类结果进行比较。


```python
# 导入数据处理所需数据包:pandas,numpy
import pandas as pd
import numpy as np
import time as time

# 读取日志数据文件，并将数据存在dataframe中
log_data = pd.read_csv("message.csv", dtype=str)
# 查看log_data前5行数据
log_data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>Entity</th>
      <th>Entity Type</th>
      <th>Log Source</th>
      <th>Host Name (Server)</th>
      <th>Message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-05-17T08:59:09.000Z</td>
      <td>lakala_linux_sysylog_logs</td>
      <td>Host (Linux)</td>
      <td>Linux Syslog Logs</td>
      <td>localhost</td>
      <td>platform microcode: firmware: requesting intel...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-05-17T08:59:09.000Z</td>
      <td>lakala_linux_sysylog_logs</td>
      <td>Host (Linux)</td>
      <td>Linux Syslog Logs</td>
      <td>localhost</td>
      <td>microcode: CPU24 sig=0x306f2, pf=0x1, revision...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-05-17T08:59:09.000Z</td>
      <td>lakala_linux_sysylog_logs</td>
      <td>Host (Linux)</td>
      <td>Linux Syslog Logs</td>
      <td>localhost</td>
      <td>platform microcode: firmware: requesting intel...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-05-17T08:59:09.000Z</td>
      <td>lakala_linux_sysylog_logs</td>
      <td>Host (Linux)</td>
      <td>Linux Syslog Logs</td>
      <td>localhost</td>
      <td>microcode: CPU25 sig=0x306f2, pf=0x1, revision...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-05-17T08:59:09.000Z</td>
      <td>lakala_linux_sysylog_logs</td>
      <td>Host (Linux)</td>
      <td>Linux Syslog Logs</td>
      <td>localhost</td>
      <td>platform microcode: firmware: requesting intel...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 查看数据表的定义信息
log_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2172 entries, 0 to 2171
    Data columns (total 6 columns):
    Time                  2172 non-null object
    Entity                2172 non-null object
    Entity Type           2172 non-null object
    Log Source            2172 non-null object
    Host Name (Server)    2172 non-null object
    Message               2172 non-null object
    dtypes: object(6)
    memory usage: 101.9+ KB


从上面的数据可以看出，事件ID的数据类型有问题，应为一个整型的值，或者是字符串；另外消息体的列名缺失。因此我们通过操作dataframe进行数据修正。


```python
# 对数据类型进行修正（事件ID）
log_data.dropna(axis=0, how='any', inplace=True)
# 查看数据表头
log_data.head()

# 取“信息”列进行聚类分析
message = log_data["Message"]
print(message.head())
```

    0    platform microcode: firmware: requesting intel...
    1    microcode: CPU24 sig=0x306f2, pf=0x1, revision...
    2    platform microcode: firmware: requesting intel...
    3    microcode: CPU25 sig=0x306f2, pf=0x1, revision...
    4    platform microcode: firmware: requesting intel...
    Name: Message, dtype: object


## 特征工程

1. 由统计信息可知，总共有6列、2172条日志记录；
2. 每一列的值均不含空值，因此无需做空值处理；
3. 由于是纯文本处理，因此需对数据特征化进行处理；
4. 得到的特征矩阵维数过大，需进行降维处理；

### 缺失值处理

无空值，因此无需进行缺失值处理

### 特征抽象

对message进行特征提取，并将结果保存在一个二维向量中


```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# step1: 声明一个向量化工具vectorizer
# stop_words = frozenset([" ", ":","for","and","a","an","the"])
vectoerizer = CountVectorizer(min_df=1, max_df=1.0,stop_words='english')
# vectoerizer = CountVectorizer(min_df=1, max_df=1.0,stop_words=stop_words, token_pattern='\\b\\w+\\b')
# step2: 根据语料集统计词集（fit）
vectoerizer.fit(message.values.astype('U'))
# step3: 打印语料集的词集信息
bag_of_words = vectoerizer.get_feature_names()
print(len(bag_of_words))
# step4: 将语料集转化为词集向量
X = vectoerizer.transform(message.values.astype('U'))
# step5: 声明一个TF-IDF转化器（TfidfTransformer）
tfidf_transformer = TfidfTransformer()
# step6: 根据语料集的词集向量计算TF-IDF（fit）
tfidf_transformer.fit(X.toarray())
# step7: 将语料集的词集向量表示转换为TF-IDF向量表示
tfidf_message = tfidf_transformer.transform(X)
print(tfidf_message.toarray())

```

    1811
    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]


### 特征降维

我们发现关键词共1830个，也就是说对应在特征向量中的维度为1830，必然导致维度灾难，因此我们通过PCA进行降维分析：


```python
from sklearn.decomposition import PCA


pca = PCA(n_components=100)
new_tfidf_message = pca.fit_transform(tfidf_message.toarray())
print(new_tfidf_message)
```

    [[-2.53633052e-01  9.12013732e-01  1.32202461e-01 ... -1.27950092e-03
       2.34781341e-03  3.82812494e-04]
     [-1.93285509e-01  1.45678366e-01 -9.93260061e-02 ...  1.34261679e-03
       8.39243122e-04  1.60183185e-04]
     [-2.53633052e-01  9.12013732e-01  1.32202461e-01 ... -1.27950092e-03
       2.34781341e-03  3.82812494e-04]
     ...
     [-1.93285509e-01  1.45678366e-01 -9.93260061e-02 ...  1.34241128e-03
       8.39178074e-04  1.60194618e-04]
     [-7.86340223e-02 -3.40070182e-02 -2.65198190e-02 ... -9.80687862e-02
      -2.03342151e-02 -2.64121570e-02]
     [-9.14477825e-02 -4.23476209e-02 -3.50776496e-02 ...  9.10152694e-04
      -1.01178356e-03 -1.23340880e-02]]


## 模型训练

使用DBSCAN模型进行数据拟合，并从分类数据获取以下信息：所有聚类，异常值（噪音数据）


```python
from sklearn.cluster import DBSCAN
from sklearn import metrics


# 通过DBSCAN算法拟合当前获取的数据集，返回拟合结果
db = DBSCAN(eps=0.23, min_samples=2).fit(new_tfidf_message)
# db = DBSCAN(eps=1.1, min_samples=2).fit(tfidf_message)

# 将结果中所有标签全部置为 FALSE
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# 将非噪音的数据集中所有数据的索引对应的标签，全部置为 TRUE
core_samples_mask[db.core_sample_indices_] = True
# 将所有的标签赋值给 labels 变量
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
# set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
# list() 函数用于将元组转换为列表。
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) # 类聚的数量
n_noise_ = list(labels).count(-1) # 噪音数量

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))
```

    Estimated number of clusters: 428
    Estimated number of noise points: 9


查看模型聚类的结果，随机抽取10个类别，查看所有元素


```python
cluster_result = pd.DataFrame(message)
cluster_result['类聚id'] = labels
cluster_result.info()
cluster1 = cluster_result[cluster_result['类聚id'] == 1] 
print(cluster1)

# # 将分析结果保存至文件
cluster_result.to_csv("message_cluster_result.csv")
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2172 entries, 0 to 2171
    Data columns (total 2 columns):
    Message    2172 non-null object
    类聚id       2172 non-null int64
    dtypes: int64(1), object(1)
    memory usage: 50.9+ KB
                                                    Message  类聚id
    1     microcode: CPU24 sig=0x306f2, pf=0x1, revision...     1
    3     microcode: CPU25 sig=0x306f2, pf=0x1, revision...     1
    5     microcode: CPU26 sig=0x306f2, pf=0x1, revision...     1
    7     microcode: CPU27 sig=0x306f2, pf=0x1, revision...     1
    9     microcode: CPU28 sig=0x306f2, pf=0x1, revision...     1
    12    microcode: CPU29 sig=0x306f2, pf=0x1, revision...     1
    14    microcode: CPU30 sig=0x306f2, pf=0x1, revision...     1
    16    microcode: CPU31 sig=0x306f2, pf=0x1, revision...     1
    1142  microcode: CPU0 sig=0x306f2, pf=0x1, revision=...     1
    1145  microcode: CPU1 sig=0x306f2, pf=0x1, revision=...     1
    1147  microcode: CPU2 sig=0x306f2, pf=0x1, revision=...     1
    1149  microcode: CPU3 sig=0x306f2, pf=0x1, revision=...     1
    1151  microcode: CPU4 sig=0x306f2, pf=0x1, revision=...     1
    1153  microcode: CPU5 sig=0x306f2, pf=0x1, revision=...     1
    1156  microcode: CPU6 sig=0x306f2, pf=0x1, revision=...     1
    1158  microcode: CPU7 sig=0x306f2, pf=0x1, revision=...     1
    1160  microcode: CPU8 sig=0x306f2, pf=0x1, revision=...     1
    1162  microcode: CPU9 sig=0x306f2, pf=0x1, revision=...     1
    1164  microcode: CPU10 sig=0x306f2, pf=0x1, revision...     1
    1167  microcode: CPU11 sig=0x306f2, pf=0x1, revision...     1
    1169  microcode: CPU12 sig=0x306f2, pf=0x1, revision...     1
    1171  microcode: CPU13 sig=0x306f2, pf=0x1, revision...     1
    1173  microcode: CPU14 sig=0x306f2, pf=0x1, revision...     1
    1175  microcode: CPU15 sig=0x306f2, pf=0x1, revision...     1
    1178  microcode: CPU16 sig=0x306f2, pf=0x1, revision...     1
    1180  microcode: CPU17 sig=0x306f2, pf=0x1, revision...     1
    1182  microcode: CPU18 sig=0x306f2, pf=0x1, revision...     1
    1184  microcode: CPU19 sig=0x306f2, pf=0x1, revision...     1
    1186  microcode: CPU20 sig=0x306f2, pf=0x1, revision...     1
    1189  microcode: CPU21 sig=0x306f2, pf=0x1, revision...     1
    ...                                                 ...   ...
    1200  microcode: CPU26 sig=0x306f2, pf=0x1, revision...     1
    1202  microcode: CPU27 sig=0x306f2, pf=0x1, revision...     1
    1204  microcode: CPU28 sig=0x306f2, pf=0x1, revision...     1
    1206  microcode: CPU29 sig=0x306f2, pf=0x1, revision...     1
    1208  microcode: CPU30 sig=0x306f2, pf=0x1, revision...     1
    1211  microcode: CPU31 sig=0x306f2, pf=0x1, revision...     1
    2119  microcode: CPU0 sig=0x306f2, pf=0x1, revision=...     1
    2121  microcode: CPU1 sig=0x306f2, pf=0x1, revision=...     1
    2123  microcode: CPU2 sig=0x306f2, pf=0x1, revision=...     1
    2125  microcode: CPU3 sig=0x306f2, pf=0x1, revision=...     1
    2128  microcode: CPU4 sig=0x306f2, pf=0x1, revision=...     1
    2130  microcode: CPU5 sig=0x306f2, pf=0x1, revision=...     1
    2132  microcode: CPU6 sig=0x306f2, pf=0x1, revision=...     1
    2134  microcode: CPU7 sig=0x306f2, pf=0x1, revision=...     1
    2136  microcode: CPU8 sig=0x306f2, pf=0x1, revision=...     1
    2139  microcode: CPU9 sig=0x306f2, pf=0x1, revision=...     1
    2141  microcode: CPU10 sig=0x306f2, pf=0x1, revision...     1
    2143  microcode: CPU11 sig=0x306f2, pf=0x1, revision...     1
    2145  microcode: CPU12 sig=0x306f2, pf=0x1, revision...     1
    2147  microcode: CPU13 sig=0x306f2, pf=0x1, revision...     1
    2150  microcode: CPU14 sig=0x306f2, pf=0x1, revision...     1
    2152  microcode: CPU15 sig=0x306f2, pf=0x1, revision...     1
    2154  microcode: CPU16 sig=0x306f2, pf=0x1, revision...     1
    2156  microcode: CPU17 sig=0x306f2, pf=0x1, revision...     1
    2158  microcode: CPU18 sig=0x306f2, pf=0x1, revision...     1
    2161  microcode: CPU19 sig=0x306f2, pf=0x1, revision...     1
    2163  microcode: CPU20 sig=0x306f2, pf=0x1, revision...     1
    2165  microcode: CPU21 sig=0x306f2, pf=0x1, revision...     1
    2167  microcode: CPU22 sig=0x306f2, pf=0x1, revision...     1
    2169  microcode: CPU23 sig=0x306f2, pf=0x1, revision...     1
    
    [64 rows x 2 columns]


## LACS结果对比

将日志文件通过ODU的方式上传至OMC，并在日志分析界面中通过聚类操作，获取聚类结果，如下：
1. 总类聚个数：353
2. 异常值（只出现一次）：4

和上文的分析结果来对比，上文中共分428个类聚，其中异常值9个。
整体的分析结果比较靠近，在比较两组分析结果后可以看出在文本一致的情况下，聚类分析的结果比较好，但是如果消息内容本身就比较杂乱，那么聚类的结果将会比较差。

## 模型评价与调优

从结果看，我们在抽取特征的时候还存在如下问题：
1. 仅从“Message”列进行聚类，导致关键聚类信息缺失，因此接下来的优化思路是：
    * 将除message和time以外的特征进行独热处理
    * 对Message进行分词并按照频率转化为特征矩阵
    * 对特征矩阵进行降维，解决维度灾难
    * 将所有特征合并，然后进行模型训练
2. 降维分析的过程，需要进行多次尝试，找到最佳的维度，可以考虑更加优化的降维方法和参数；
3. 对中文文本的处理不够灵活，聚类结果也比较差，需要进行优化；
4. 尝试使用多种算法进行训练，找出最佳聚类算法和参数，这个过程可以参考如下思路：
    * 循环eps和min_samples参数，计算聚类结果（或者用其他评估方法的结果）；
    * 将结果绘制matplot曲线，查看eps和min_sample变化的过程中对结果的影响。
