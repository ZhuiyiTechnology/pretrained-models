# 开源预训练语言模型合集

这是由追一科技有限公司推出的一个预训练模型合集，主要发布自研的预训练语言模型，推动自然语言处理技术的进步。预训练语言模型通过在大规模文本上进行预训练，可以作为下游自然语言处理任务的模型参数或者模型输入以提高模型的整体性能。

## 模型概览

以下是我们目前公开发布的模型概览：

| 名称           | 数据来源     | 训练数据大小 | 词表大小 | 模型大小 | 下载地址 |
| ------------  | ----------- | -----------| -------- | -------- | -------- |
| RoBERTa Tiny  | 百科,新闻 等  |     35G    | 21128    | 27MB | [百度网盘](https://pan.baidu.com/s/1BWhzP8K9rHi2uWtOUQoX5Q)(beum) |
| RoBERTa Small | 百科,新闻 等  |     35G    | 21128  | 48MB  | [百度网盘](https://pan.baidu.com/s/1AqoD49xkeAO4KHsBrmFFOA)(hjqc) |
| SimBERT Base  | [百度知道](http://zhidao.baidu.com/) | 2200万相似句组 | 21128  | 344MB  | [百度网盘](https://pan.baidu.com/s/1uGfQmX1Kxcv_cXTVsvxTsQ)(6xhq) |

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

- <strong>【备注】</strong> 简单来说，这就是一个finetune过的bert base模型，但是[CLS]对应的输出具有句向量的意义，可以用于检索任务，理论上短文本效果会更好，在跟<a href="https://kexue.fm/archives/5743">这里</a>一样的验证集上得到了0.96的top1准确率；此外还具有一对多生成能力。详见例子<a href="https://github.com/ZhuiyiAI/pretrained-models/blob/master/examples/simbert_base.py">simbert_base.py</a>。

## 如何引用

Bibtex：

```tex
@techreport{zhuiyipretrainedmodels,
  title={Open Language Pre-trained Model Zoo - ZhuiyiAI},
  author={Jianlin Su},
  year={2020},
  url = "https://github.com/ZhuiyiAI/pretrained-models",
}
```

## 联系我们

邮箱：ai@wezhuiyi.com

## 相关链接

追一科技：https://zhuiyi.ai
