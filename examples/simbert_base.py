#! -*- coding: utf-8 -*-
# SimBERT base 基本例子
# 测试环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.7.7

import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
from bert4keras.snippets import uniout
from keras.layers import *

maxlen = 32

# bert配置
config_path = '/root/kg/bert/chinese_simbert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_simbert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate(
            [segment_ids, np.ones_like(output_ids)], 1)
        return seq2seq.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=1, topk=5):
        token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
        output_ids = self.random_sample([token_ids, segment_ids], n, topk)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(start_id=None,
                                       end_id=tokenizer._token_end_id,
                                       maxlen=maxlen)


def gen_synonyms(text, n=100, k=20):
    """"含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    """
    r = synonyms_generator.generate(text, n)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]


"""
gen_synonyms(u'微信和支付宝哪个好？')

[
    u'微信和支付宝，哪个好?',
    u'微信和支付宝哪个好',
    u'支付宝和微信哪个好',
    u'支付宝和微信哪个好啊',
    u'微信和支付宝那个好用？',
    u'微信和支付宝哪个好用',
    u'支付宝和微信那个更好',
    u'支付宝和微信哪个好用',
    u'微信和支付宝用起来哪个好？',
    u'微信和支付宝选哪个好',
    u'微信好还是支付宝比较用',
    u'微信与支付宝哪个',
    u'支付宝和微信哪个好用一点？',
    u'支付宝好还是微信',
    u'微信支付宝究竟哪个好',
    u'支付宝和微信哪个实用性更好',
    u'好，支付宝和微信哪个更安全？',
    u'微信支付宝哪个好用？有什么区别',
    u'微信和支付宝有什么区别？谁比较好用',
    u'支付宝和微信哪个好玩'
]
 """
