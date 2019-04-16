## Chinese Intent Classify 2018-10

#### 1.preprocess

prepare() 将按类文件保存的数据汇总、去重，去除停用词，统一替换地区、时间

等特殊词，word_replace() 替换同音、同义词，打乱后划分训练、测试集

#### 2.explore

统计词汇、长度、类别的频率，条形图可视化，计算 sent / word_per_sent 指标

#### 3.represent

embed() 通过 word_inds 与 word_vecs 建立词索引到词向量的映射

label2ind() 建立标签索引、sorted() 保持一致，align() 将序列填充为相同长度

#### 4.build

train 80% / dev 20% 划分，通过 adnn、crnn、rcnn 构建分类模型

#### 5.classify

predict() 输入单句、清洗后进行预测，输出所有类别的概率，plot_att() 可视化

#### 6.eval

计算总体的正确率、f1 值，各类的精确率、召回率，显示错判样本