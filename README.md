# 开源预训练语言模型合集

这是由追一科技有限公司推出的一个预训练模型合集，主要发布自研的预训练语言模型，推动自然语言处理技术的进步。预训练语言模型通过在大规模文本上进行预训练，可以作为下游自然语言处理任务的模型参数或者模型输入以提高模型的整体性能。

## 模型概览

以下是我们目前公开发布的模型概览：

| 名称           | 数据来源     | 训练数据大小 | 词表大小 | 模型大小 | 下载地址 |
| :----------:  | :---------: | :---------:| :------: | :------: | :------: |
| RoBERTa Tiny  | 百科,新闻 等  |     35G    | 21128    | 27MB | [百度网盘](https://pan.baidu.com/s/1AfRKIBMIoxzXbfqWE4aDsw)(mrcv) |
| RoBERTa Small | 百科,新闻 等  |     35G    | 21128  | 48MB  | [百度网盘](https://pan.baidu.com/s/15-loby3PEwBtMLc-On6Vzg)(j2ns) |
| SimBERT Base  | [百度知道](http://zhidao.baidu.com/) | 2200万相似句组 | 21128  | 344MB  | [百度网盘](https://pan.baidu.com/s/1uGfQmX1Kxcv_cXTVsvxTsQ)(6xhq) |
| RoBERTa<sup>+</sup> Tiny  | 百科,新闻 等  |     35G    | 21128    | 35MB | [百度网盘](https://pan.baidu.com/s/1JjqwVhnpIGtjecXBsdv2BQ)(bbgq) |
| RoBERTa<sup>+</sup> Small | 百科,新闻 等  |     35G    | 21128  | 67MB  | [百度网盘](https://pan.baidu.com/s/1L_15sYXZcVmlxb9QqgAJ-Q)(88wp) |


## 评估结果

这里给出部分数据集上模型的评测结果。

(注：以下实验结果均为重复跑三次后的平均值。预测阶段，两个small模型速度完全一致，两个tiny模型速度也完全一致。)

### 文本情感分类

任务来源：https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip

评测脚本：<a href="https://github.com/ZhuiyiTechnology/pretrained-models/blob/master/examples/task_sentiment.py">task_sentiment.py</a>

评测指标：accuracy

| 模型           | 验证集（valid） | 训练速度    | 第一个epoch结束时的指标值 |  测试集（test） |
| :----------:  | :------------: | :---------:| :--------------------: | :------------: |
| RoBERTa Small |     94.89%     |  38s/epoch |         90.48%         |     94.81%     |
| ALBERT Small  |     94.57%     |  33s/epoch |         91.02%         |     94.52%     |
| RoBERTa Tiny  |     94.46%     |  23s/epoch |         90.83%         |     94.00%     |
| ALBERT Tiny   |     94.14%     |  20s/epoch |         90.18%         |     93.78%     |

### IFLYTEK' 长文本分类

任务来源：https://github.com/CLUEbenchmark/CLUE

评测脚本：<a href="https://github.com/ZhuiyiTechnology/pretrained-models/blob/master/examples/task_iflytek.py">task_iflytek.py</a>

评测指标：accuracy

| 模型           | 验证集（dev） | 训练速度    | 第一个epoch结束时的指标值 |
| :----------:  | :---------: | :---------:| :--------------------: |
| RoBERTa Small |   57.66%    |  27s/epoch |         52.60%         |
| ALBERT Small  |   57.14%    |  24s/epoch |         48.21%         |
| RoBERTa Tiny  |   57.43%    |  16s/epoch |         49.76%         |
| ALBERT Tiny   |   56.42%    |  14s/epoch |         43.84%         |

### LIC2019-IE 信息抽取任务

任务来源：http://lic2019.ccf.org.cn/kg

评测脚本：<a href="https://github.com/ZhuiyiTechnology/pretrained-models/blob/master/examples/task_lic2019_ie.py">task_lic2019_ie.py</a>

评测指标：F1

| 模型           | 验证集（dev） | 训练速度    | 第一个epoch结束时的指标值 |
| :----------:  | :---------: | :---------:| :--------------------: |
| RoBERTa Small |   78.09%    |  375s/epoch |         63.85%        |
| ALBERT Small  |   77.69%    |  335s/epoch |         46.58%        |
| RoBERTa Tiny  |   76.65%    |  235s/epoch |         46.12%        |
| ALBERT Tiny   |   75.94%    |  215s/epoch |         31.66%        |

### CIPS-SogouQA 阅读理解式问答

任务来源：http://task.www.sogou.com/cips-sogou_qa/

评测脚本：<a href="https://github.com/ZhuiyiTechnology/pretrained-models/blob/master/examples/task_cips_sogou_qa.py">task_cips_sogou_qa.py</a>

评测指标：(EM + F1) / 2

| 模型           | 验证集（dev） | 训练速度    | 第一个epoch结束时的指标值 |
| :----------:  | :---------: | :---------:| :--------------------: |
| RoBERTa Small |   70.35%    |  607s/epoch |         61.07%        |
| ALBERT Small  |   66.66%    |  582s/epoch |         50.93%        |
| RoBERTa Tiny  |   67.85%    |  455s/epoch |         49.78%        |
| ALBERT Tiny   |   63.41%    |  443s/epoch |         37.47%        |

（注：此处是直接使用UniLM的Seq2Seq方案来做阅读理解，主要测试模型用做文本生成时的能力。但要说明的是，Seq2Seq并非做阅读理解的标准方案。）

## 模型详情

此处对每个模型进行较为详细的介绍

### RoBERTa Tiny

- <strong>【配置】</strong> 4层模型，hidden size为312，对Embedding层做了低秩分解(312->128->312)，可以用<a href="https://github.com/bojone/bert4keras/tree/master/examples">bert4keras</a>加载使用。

- <strong>【训练】</strong> 使用<a href="https://github.com/bojone/bert4keras/tree/master/pretraining">bert4keras</a>在TPU v3-8上训练，使用带梯度累积的LAMB优化器，批大小为800，累积4步更新，相当于以批大小3200训练了125k步（前3125步为warmup）。

- <strong>【备注】</strong> 速度跟<a href="https://github.com/brightmart/albert_zh">albert tiny</a>一致，普通分类性能也基本一致，但由于roberta模型并没有参数共享这个约束，所以在生成式任务等复杂任务上效果优于albert tiny。

### RoBERTa Small

- <strong>【配置】</strong> 6层模型，hidden size为384，对Embedding层做了低秩分解(384->128->384)，可以用<a href="https://github.com/bojone/bert4keras/tree/master/examples">bert4keras</a>加载使用。

- <strong>【训练】</strong> 使用<a href="https://github.com/bojone/bert4keras/tree/master/pretraining">bert4keras</a>在TPU v3-8上训练，使用带梯度累积的LAMB优化器，批大小为800，累积4步更新，相当于以批大小3200训练了125k步（前3125步为warmup）。

- <strong>【备注】</strong> 速度跟<a href="https://github.com/brightmart/albert_zh">albert small</a>一致，普通分类性能也基本一致，但由于roberta模型并没有参数共享这个约束，所以在生成式任务等复杂任务上效果优于albert small。

### SimBERT Base

- <strong>【配置】</strong> 跟bert base一致，12层模型，hidden size为768。

- <strong>【训练】</strong> 使用<a href="https://github.com/bojone/bert4keras/tree/master/pretraining">bert4keras</a>基于<a href="https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip">chinese_L-12_H-768_A-12</a>进行继续训练，训练任务为“度量学习+UniLM”，以Adam优化器（学习率2e-6，批大小128）的Adam优化器在单个TITAN RTX上训练了117万步。

- <strong>【备注】</strong> 简单来说，这就是一个finetune过的bert base模型，但是[CLS]对应的输出具有句向量的意义，可以用于检索任务，理论上短文本效果会更好，在跟<a href="https://kexue.fm/archives/5743">这里</a>一样的验证集上得到了0.96的top1准确率；此外还具有一对多生成能力。详见例子<a href="examples/simbert_base.py">simbert_base.py</a>。

## 如何引用

Bibtex：

```tex
@techreport{zhuiyipretrainedmodels,
  title={Open Language Pre-trained Model Zoo - ZhuiyiAI},
  author={Jianlin Su},
  year={2020},
  url = "https://github.com/ZhuiyiTechnology/pretrained-models",
}
```

## 致谢信息
本项目部分受到[**谷歌TensorFlow Research Cloud**](https://www.tensorflow.org/tfrc)计划资助，在此特别致谢。

## 联系我们

邮箱：ai@wezhuiyi.com

## 相关链接

追一科技：https://zhuiyi.ai
