# 用户提供核心30问的合并审阅

日期：2026-10-02。最终结论：30/30项用户提供问题已关联到23道规范题；此前15项映射保留。独立审阅的新TFM-020—023事实、公式与来源范围通过，工具审阅发现的三处守卫问题均已修复并复核，无剩余待修项。

来源边界：本轮依据用户贴出的文本合并，`merged_from_user_text`不等于读取知乎正文。SUP-C001仍为blocked，用户文本的origin、30个编号及`original_page_verified=false`另存request；网页标题、作者和版本未直接核验，不据此声称是某公司真题。

## 范围与独立性

只读审阅`tools/supplement.py`、`tools/test_supplement.py`及build/validate/run_checks接入，核对request与source catalog、30项实际映射和`data/transformer_supplement.json`四道新题。另对core的规范答案做需求对应检查；core十二题的完整事实审阅由另一同伴完成，不冒称本报告独立重审了全部core答案。

审阅者本轮参与编写DST-003/EVA-002并添加工程/评测关联标记；这些题的独立内容审核由根任务负责。本报告对其记录映射和来源约束，不以作者自查替代独立同伴审阅。当前审阅阶段只写本报告，未改数据或工具。

## 工具发现与修复复核

### G-01 [P2，已修] request结构在使用前缺少类型检查

初版内存复现：`text_provenance=None`抛AttributeError，`article.items=None`抛TypeError；原bad metadata测试只覆盖规范题的marker字段。修复后request/article要求对象，items要求含字符串key和非空question的对象列表，provenance要求对象。相同复现现在得到明确错误列表，未抛异常。

### G-02 [P2，已修] 30项数量不能证明保留全部原编号

初版将Z01换成Z99并同步题库marker后，仍有30项且校验通过。用户文本状态现要求编号集合恰为Z01—Z30并检查唯一性；原复现被`core30 keys must be exactly Z01 through Z30`拒绝。

### G-03 [P2，已修] request不得自行升级原网页访问状态

初版保持source catalog中SUP-C001为blocked，仅改request的access=full/status=merged即可通过。修复后`validate_supplement`核对source_id对应记录、URL与access；原复现被catalog不一致错误拒绝。blocked直接合并仍被禁止，用户文本合并另要求明确origin和原页未核验。

独立执行`python -X utf8 -B -m unittest discover -s tools -p test_supplement.py -v`：8组全部通过。再次做上述四个内存变体（包括两种G-01情况），都返回预期错误；实际request与题库返回空错误列表。没有运行会重写章节/索引的build或run_checks，最终总装检查由根任务运行。

## 新四题事实与数学

重新读取[原始Transformer论文](https://arxiv.org/pdf/1706.03762)第4节及5.4节、[PyTorch 2.14 Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html)、[2.14 SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)，以及[Jain & Wallace论文](https://aclanthology.org/N19-1357.pdf)和[Wiegreffe & Pinter论文](https://aclanthology.org/D19-1002.pdf)的研究范围、实验与讨论正文。没有运行大模型训练或论文实验。

- TFM-020：inverted dropout的均值/方差、训练与eval区别正确；原论文residual dropout在相加前、embedding+PE上及base配置0.1有直接正文依据。明确SDPA由调用者设置dropout_p，未把nn.Dropout默认0.5或现代所有插入位置倒填到2017论文。独立两点Bernoulli计算验证p=0.3、x=2时均值2、方差1.714285714，与公式一致。
- TFM-021：A是输入相关激活，W_Q/W_K等为参数；softmax行Jacobian、D及Q/K/W_Q梯度形状正确。用标准库对2×2 causal attention的W_Q梯度做中心差分，最大误差约2.50e−11；被mask位置梯度为零。解释性争论限定于所读论文的NLP模型/任务，未把BiRNN/LSTM结论泛化到全部LLM。
- TFM-022：O(1)对应允许位置间的通信路径，不是运行时间；稠密attention的O(n²d)、常规显式矩阵O(hn²)及含投影/FFN总计算正确。区分全局、causal、局部窗口的可达性与逐token生成；短路径未被当成长上下文质量保证。
- TFM-023：标准softmax权重非负；有限允许logit在精确实数计算中严格正，mask为零；负logit和负V/输出例子正确。公式明确dropout前和至少一个允许位置。没有断言所有同名论文都不存在，而是纠正原始Transformer中不成立的组件称呼。

NEWTFM-P01—04均是实际可读的primary资料，notes/scope限定具体API、论文模型及相关正文；未虚构精确出版日或性能复现。四题保留editorial=true，无社区证据边；用户marker体现本轮需求映射，不伪造知乎原页证据。

## 30项需求对应

| 用户编号 | 规范题号 | 对应检查 |
|---|---|---|
| Z01、Z02、Z11、Z27 | TFM-001 | 注意力公式、缩放假设与softmax |
| Z03、Z12 | TFM-002 | 多头投影、reshape、拼接及维度 |
| Z04、Z13 | TFM-019 | 位置公式与顺序感知 |
| Z05、Z15 | TFM-010 | 原始FFN公式、作用及现代变体 |
| Z06、Z16 | TFM-005 | LN统计轴与BN取舍 |
| Z07、Z08、Z18 | PRE-014 | global norm clipping与Noam/warmup |
| Z09、Z17 | TFM-012 | 残差及梯度路径 |
| Z10、Z21 | TFM-014 | 词表概率及参数共享边界 |
| Z14 | TFM-016 | Encoder-Decoder cross-attention |
| Z19 | TFM-020 | Dropout定义及原始插入位置 |
| Z20 | TFM-021 | 动态权重、参数梯度和解释性 |
| Z22 | PRE-001 | 概率乘积、teacher forcing及生成 |
| Z23 | TFM-022 | 长距离通信路径与能力限制 |
| Z24 | TFM-023 | Negative Attention术语纠错 |
| Z25 | TFM-004 | Seq2seq架构 |
| Z26、Z28 | TFM-003 | 掩码、padding与可变长度 |
| Z29 | DST-003、DST-013、DST-015 | 显存分解、重计算及offload |
| Z30 | EVA-002、EVA-005、EVA-006、EVA-015 | 文本/人工/事实/回归评测 |

逐项核对question与merge_note，没有把重复编号删除成需求缺失，也没有重复复制完整答案。必须纠正的原答案摘录包括：结构相同误称跨层/跨头共享；LN误称减少协方差偏移；动态A误称独立参数；小权重误称Negative Attention；mask把概率与score混同；BN的绝对不适用表述；显存技巧混淆优化对象；BLEU/ROUGE单独泛化为生成质量。Dropout的泛化收益亦应作为正则化目的/经验作用，而非保证。request未保存逐字全文的其他原答案仅核对题目和合并说明，不声称核验了未见文本。

## 最终只读检查

229题、273来源；`common.validate`与`validate_supplement(questions,sources)`均返回空错误列表。Z01—Z30全部映射，共23道规范题，原15个U marker仍齐全。扫描全库blocked来源与技术reference_ids或community_evidence的交集，二者均为零。

本轮完成的是30/30用户提供题目的合并与纠错；知乎原页仍未直接核验。先前0/30的审阅记录是收到用户文本前的历史状态，本报告记录收到文本后的最新状态。
