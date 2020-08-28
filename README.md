# 任务

使用 `BertForSequenceClassification`，在自己的训练集上训练情感分类模型



# 数据集

数据集来源于 [https://github.com/bojone/bert4keras/tree/master/examples/datasets](https://github.com/bojone/bert4keras/tree/master/examples/datasets)

是一个中文的情感二分类数据集。



# 词汇表

词汇表 `vocab.txt` 来自于哈工大的中文预训练语言模型 `BERT-wwm, Chinese`。

地址：[https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD](https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD)



# 代码

`BertPractice.py`：使用 `BertForSequenceClassification`，训练情感分类模型



# 文章

代码讲解，请参考我的博客文章：[Bert 文本分类实战](https://blog.zhangxiann.com/202008222159/)