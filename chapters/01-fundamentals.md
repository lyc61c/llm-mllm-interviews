# NLP、数学与深度学习基础

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 目录

- [分词与文本表示](#topic-1)
  - [BAS-001 · 中文分词有哪些难点？jieba 的 DAG、动态规划和 HMM 如何配合？](#bas-001)
  - [BAS-004 · one-hot、静态词向量与上下文 embedding 的区别是什么？](#bas-004)
  - [BAS-005 · Word2Vec 的 CBOW 和 Skip-gram 如何训练，有哪些关键细节？](#bas-005)
  - [BAS-006 · 负采样与层次 Softmax 怎样加速 Word2Vec，是否等价于原始 Softmax？](#bas-006)
  - [BAS-007 · N-gram 与神经语言模型怎样估计序列概率，平滑解决什么问题？](#bas-007)
  - [BAS-015 · TF-IDF 与 TextRank 如何提取关键词，各有哪些局限？](#bas-015)
  - [PRE-002 · BPE、WordPiece、Unigram 与 SentencePiece 分别是什么？](#pre-002)
  - [PRE-003 · 词表越大越好吗，扩词表有哪些代价？](#pre-003)
- [序列任务与经典 NLP](#topic-2)
  - [BAS-002 · POS、NER 如何做序列标注？HMM、独立分类与 CRF 有什么区别？](#bas-002)
  - [BAS-003 · 成分句法与依存句法分别预测什么，怎样评价解析结果？](#bas-003)
  - [BAS-008 · CNN、RNN 与 Transformer 在序列建模、长依赖和并行性上怎样比较？](#bas-008)
  - [BAS-009 · LSTM 与 GRU 怎样缓解普通 RNN 的梯度问题，有何结构区别？](#bas-009)
- [数学、概率与统计](#topic-3)
  - [TFM-013 · 交叉熵、KL 散度与 perplexity 的关系和实现注意点是什么？](#tfm-013)
  - [TFM-017 · 熵衡量什么？离散熵、条件熵与模型输出熵怎样区分？](#tfm-017)
  - [TFM-018 · 矩阵的秩与特征值如何计算，和奇异值有什么关系？](#tfm-018)
- [深度学习与优化基础](#topic-4)
  - [BAS-010 · Sigmoid、Tanh、ReLU、GELU 与 SiLU 的公式和梯度特点是什么？](#bas-010)
  - [BAS-011 · SGD、Momentum、AdaGrad、RMSProp 与 Adam 的更新规则怎样理解？](#bas-011)
  - [BAS-012 · 反向传播怎样使用链式法则？线性层、MSE 与 Softmax 交叉熵如何求梯度？](#bas-012)
  - [BAS-013 · 过拟合、欠拟合与数据泄露怎样区分，L1/L2、早停和 Dropout 分别做什么？](#bas-013)
  - [BAS-014 · Xavier、He 初始化与 embedding 乘 √d_model 的目的是什么？](#bas-014)
  - [BAS-016 · 分类为什么通常用交叉熵而非 MSE？MSE 在数学上可行吗？](#bas-016)

<a id="topic-1"></a>
## 分词与文本表示

<a id="bas-001"></a>
### BAS-001 · 中文分词有哪些难点？jieba 的 DAG、动态规划和 HMM 如何配合？

**L1** · 腾讯

#### 答案

中文分词难在一句话没有空格，既要判断词的边界，又要处理歧义和词典里没有的新词。比如“研究生命起源”里，“研究生”虽然是词，却不一定应该在这里连起来。jieba 的精确模式先查前缀词典，把每个位置可能组成的词都列出来，形成一张只能向后走的有向无环图，也就是 DAG。

接着它用动态规划比较整句话的切分得分，而不是遇到长词就立即选它。公式中的 D(i) 表示从第 i 个字到句尾的最佳得分，j 是候选词的结束位置，词频估计的概率取对数后就能逐词相加。已经求好的后半句得分可以重复使用，因此不必穷举所有完整切分。

对于词典没覆盖好的片段，HMM，也就是隐马尔可夫模型，会根据字符推断词首、词中、词尾或单字词这四种状态，再用 Viterbi 算法找最可能的状态序列。领域新词可以通过用户词典和词频调整改善；搜索模式还会拆开长词以增加检索命中。最后要按业务评价词界和检索效果，因为中文“词”的边界，与大模型 tokenizer 为计算而划分的 token 边界并不是一回事。

```math
D(i)=\max_{j\in\mathrm{DAG}(i)}\{\log P(x_{i:j})+D(j+1)\},\qquad D(n+1)=0
```

#### 易错点

- jieba 比较的是整句切分概率，不能简化成优先选最长词。
- 中文词界与大模型 token 边界的目的不同，不能直接替换。

#### 追问

- 增加一个领域词并调整词频，为什么可能改变邻近位置的切分？
- 词典方法和序列标注方法，分别如何识别词典中没有的新词？

<a id="bas-004"></a>
### BAS-004 · one-hot、静态词向量与上下文 embedding 的区别是什么？

**L1**

#### 答案

one-hot 只表示“这个词是谁”，静态词向量进一步表示“这个词通常怎样使用”，上下文 embedding 则表示“这个词在这句话里是什么意思”。one-hot 是词表长度的向量，只有对应词的位置为 1；不同词之间距离一样，无法表达“猫”和“狗”比“猫”和“数据库”更相近。

静态词向量把词 ID 映射到一张可训练的密集向量表，通常通过共现统计或上下文预测学出使用习惯相近的表示。公式里的 E 是向量表，查表相当于用 one-hot 选出其中一行。同一个词形通常只对应一个向量，所以“吃苹果”和“苹果发布新品”里的“苹果”难以直接区分。上下文表示 h 则由网络读取整段可见文本后计算，因此可以随句子改变。

ELMo 用左右两个语言模型的特征，BERT 在每层融合左右上下文，GPT 的表示依赖当前位置能看到的前缀。这些都是 token 表示，还不自动等于适合检索的句向量；句级检索往往需要专门训练、把多个位置汇聚成一个向量，并正确排除 padding。余弦相似度看两个向量方向有多接近，分母是各自长度，但高相似度只说明所选空间中的相近，不能单独证明两个句子的事实完全一致。

```math
\begin{aligned}e(w)&=E^\top\mathrm{onehot}(w)\\h_t&=f_\theta(x_{1:T})_t\\\mathrm{cos}(u,v)&=\frac{u^\top v}{\lVert u\rVert\lVert v\rVert}\end{aligned}
```

#### 易错点

- 词表向量、上下文 token 表示和检索句向量不是同一种输出。
- 词向量加减类比是特定数据与空间的现象，不保证任何词都成立。

#### 追问

- 同一个词形只有一行静态向量时，为什么难区分多个词义？
- 对 token 向量做平均时，为什么必须排除 padding？

<a id="bas-005"></a>
### BAS-005 · Word2Vec 的 CBOW 和 Skip-gram 如何训练，有哪些关键细节？

**L1**

#### 答案

CBOW 用周围的词预测中间的词，Skip-gram 用中间的词预测周围的词。比如“今天喝了一杯咖啡”，CBOW 可以汇总“今天、喝、一杯”等上下文去预测“咖啡”；Skip-gram 则从中心词生成多个“中心词—上下文词”训练对。两者都通过预测任务学出词向量，而不需要人工标注词义。

模型通常有输入、输出两张向量表。输入向量 v 表示当前读入的词，输出向量 u 表示待预测的词，它们的点积决定匹配得分；完整 softmax 再把所有候选词的得分转成概率。公式中的 c 是中心词、o 是上下文词，V 是词表。CBOW 常对上下文向量求平均，因此计算便宜，但这种汇总通常不保留词序；Skip-gram 让一个中心词贡献多个训练对，常能较好利用稀有词的有限出现，实际效果仍取决于语料和训练设置。

大词表下会用负采样或层次 softmax 减少输出计算。高频词还常被随机丢弃一部分，避免“的、是”等词支配训练；随机窗口使距离更近的词更常进入训练。训练后要说明使用输入表、输出表还是组合表示。继续训练时，词表和频率统计也要更新，并保留有代表性的旧数据，因为新语料会改变已有词的坐标关系，不能理解成只补几个新词而旧空间完全不动。

```math
\begin{aligned}p(o\mid c)&=\frac{\exp(u_o^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}\\\mathcal L_{\rm SG}&=-\sum_{(c,o)}\log p(o\mid c)\\\bar v_C&=\frac1{|C|}\sum_{w\in C}v_w\quad(\mathrm{CBOW})\end{aligned}
```

#### 易错点

- Skip-gram 是中心词预测上下文，不能与 CBOW 的方向说反。
- 窗口共现反映语料中的使用关系，不是全局语义真值。

#### 追问

- 输入词与被预测词为什么使用两张向量表？
- 随机缩小窗口为什么让近距离上下文获得更大训练权重？

<a id="bas-006"></a>
### BAS-006 · 负采样与层次 Softmax 怎样加速 Word2Vec，是否等价于原始 Softmax？

**L2**

#### 答案

负采样和层次 softmax 都减少了 Word2Vec 每一步要更新的输出节点，但它们与完整 softmax 的关系不同：层次 softmax 重新参数化一个归一化词分布，负采样则把训练任务改成区分真实词对和随机词对。完整 softmax 要给词表里的所有词打分，词表很大时成本高。

层次 softmax 把词放在二叉树叶子上，预测某个词就依次判断路径上的左右分支，再把分支概率相乘。公式里的 b 表示某一步走左还是右，u 是该内部节点的向量。所有叶子路径构成完整选择，因此词概率仍能加起来为 1。Huffman 树让高频词有更短路径，典型成本从遍历词表变成约对数级的路径计算。

负采样把真实中心词与上下文词作为正例，再从噪声分布抽 K 个词作为负例，只计算这些词对的 sigmoid 二分类损失。公式中的 f 是词频，3/4 次方是原始方案常用的噪声分布设置。它适合高效学习表示，却不是原 softmax 梯度的精确无偏替代，也不能直接当作完整条件词概率。随机抽到的负词可能仍与中心词相关，因此“负”是训练采样身份，不是绝对的语义判断；K、采样分布和树结构都影响结果。

```math
\begin{aligned}\mathcal L_{\rm NS}&=-\log\sigma(u_o^\top v_c)-\sum_{k=1}^{K}\log\sigma(-u_{n_k}^\top v_c)\\n_k&\sim P_n(w)\propto f(w)^{3/4}\\p_{\rm HS}(w\mid c)&=\prod_{j\in\mathrm{path}(w)}\sigma(b_j u_j^\top v_c),\quad b_j\in\{-1,1\}\end{aligned}
```

#### 易错点

- 负采样改变训练目标，不是原始多分类 softmax 的精确等价实现。
- 随机负例可能仍与中心词相关，不能把它们都视为确定无关。

#### 追问

- 二叉树的各叶子路径概率，为什么能组成和为 1 的词分布？
- Skip-gram 负采样与移位 PMI 矩阵分解有什么联系？

<a id="bas-007"></a>
### BAS-007 · N-gram 与神经语言模型怎样估计序列概率，平滑解决什么问题？

**L1**

#### 答案

语言模型估计的是一段文本出现的概率；N-gram 用最近的几个词来估计，神经语言模型则用网络表示上下文。把整句概率写成每一步条件概率的乘积，是概率链式法则，本身没有近似。N-gram 的近似发生在后面：假设当前词只依赖前 N−1 个词，而不是完整历史。

例如二元模型根据“喝”之后各种词的出现次数，估计下一词是“水”或“咖啡”的概率。公式中 C 是计数，h 是历史上下文，V 是词表。如果某组合训练中没出现，直接用计数得到的概率就是零，包含它的整句概率也会变零。平滑就是从已见事件挪一些概率给未见事件；加法平滑在分子加 α，也必须在分母加 α|V|，才能保持总概率为 1。插值、回退和 Kneser-Ney 会更细致地利用较短上下文和词在不同上下文中的出现情况。

神经模型用共享词向量和网络在上下文之间泛化，RNN 逐步更新状态，Transformer 聚合可见位置，因此通常更擅长复杂依赖，但仍受数据和上下文长度限制。N-gram 便宜、易解释，适合做基线。比较困惑度必须统一 tokenizer、测试数据和计分范围。NLP 是处理人类语言的领域，LLM 是其中一类模型；传统专用模型与大模型都可以做分类、抽取等任务，应根据效果与成本选，而不是用参数量划一道绝对分界。

```math
\begin{aligned}p(x_{1:T})&=\prod_{t=1}^T p(x_t\mid x_{\lt t})\\p_{N\text{-gram}}(x_t\mid x_{\lt t})&\approx\frac{C(x_{t-N+1:t})}{C(x_{t-N+1:t-1})}\\p_{\rm add\text{-}\alpha}(w\mid h)&=\frac{C(h,w)+\alpha}{C(h)+\alpha |V|}\end{aligned}
```

#### 易错点

- 概率链式法则是精确关系，N-gram 截短历史才是额外近似。
- 平滑给未见事件增加概率时，也必须调整分母保持归一化。

#### 追问

- Kneser-Ney 为什么关注一个词出现在多少种不同上下文？
- 未知词处理和 tokenizer 不同时，为什么困惑度不能直接比较？

<a id="bas-015"></a>
### BAS-015 · TF-IDF 与 TextRank 如何提取关键词，各有哪些局限？

**L2**

#### 答案

TF-IDF 用“在当前文档常见、在其他文档少见”衡量词的显著性，TextRank 用“与哪些重要词相连”衡量显著性。比如一篇讲数据库的文章里，“数据库”出现多次，而“今天”在大量文章中都常见，前者通常更值得作为关键词。两种方法都不需要逐篇人工标注关键词。

TF 是文档内词频，可以用计数、归一化计数或对数变换；IDF 是逆文档频率，公式中的 N 是背景文档总数、df 是包含该词的文档数。词出现的文档越少，IDF 越大，加 1 是平滑处理的一种约定。TextRank 则先把词变成图上的节点，把局部窗口内共现变成边，再反复把邻居的重要性沿边传过来。公式里的 a 是阻尼系数，w 是边权，邻居的贡献按它的总边权分摊；同样思路也可用于句子图做摘要。

TF-IDF 忽略词序和同义关系，依赖背景语料；TextRank 依赖窗口和候选词规则，连接很多的常见词可能占优势，孤立但关键的实体也可能低分。jieba 提供这两种工具，但分词、领域词典、停用词仍要调好。评价应看人工关键词覆盖、冗余和下游检索收益；高分不是事实重要性的概率，IDF 也应在合适的训练语料上拟合，避免使用测试数据造成泄露。

```math
\begin{aligned}\mathrm{TFIDF}(w,d)&=\mathrm{tf}(w,d)\log\frac{N+1}{\mathrm{df}(w)+1}\\S(v)&=(1-a)+a\sum_{u\to v}\frac{w_{uv}}{\sum_k w_{uk}}S(u)\end{aligned}
```

#### 易错点

- TF-IDF 与图中心性得分都不是语义正确性的概率。
- 用测试数据拟合 IDF 也可能泄露，统计语料必须明确。

#### 追问

- 扩大共现窗口会怎样改变 TextRank 的图与关键词排序？
- 关键词工具的词界与模型 tokenizer 的 token 边界有什么不同目的？

<a id="pre-002"></a>
### PRE-002 · BPE、WordPiece、Unigram 与 SentencePiece 分别是什么？

**L1** · 腾讯

#### 答案

BPE、WordPiece 和 Unigram 是把文本切成可复用子词单元的方法，SentencePiece 则是能训练 BPE、Unigram 等模型的工具。子词介于整词和单字符之间，让常见片段得到紧凑表示，也使未见过的词能由较小单元组成。例如一个少见英文词可以拆成常见词根与后缀，而不必整个变成未知词。

BPE 从小单元开始，反复合并出现频繁的相邻对，编码时根据学到的合并优先级处理文本。WordPiece 编码常根据最终词表做最长前缀匹配，后续片段可带 continuation 标记；它不等于按 BPE 的合并顺序重放。WordPiece 的训练实现并不完全统一，教学中用的频率比打分不能直接当作某个未公开训练器的唯一公式。Unigram 给候选片段分配概率，逐步删减词表，编码时寻找总概率高的切分，还能采样不同切分以增加训练变化。

SentencePiece 可直接从原始文本学习，统一处理空白与归一化，减少对语言专用预分词的依赖。Byte-level BPE 从字节覆盖输入，通常能表示未见字符，但归一化、预分词与特殊 token 仍可能影响精确还原。选 tokenizer 要比较语言覆盖、文本长度、代码与空白保留以及成本；使用已有 checkpoint 时必须保持兼容，不能把中文先用 jieba 切好就认为原模型仍会读到相同的 token。

#### 易错点

- SentencePiece 是支持多种模型的工具，不能与 BPE、Unigram 当作同层级算法并列。
- token 不一定等于一个汉字或一个英文词。

#### 追问

- 字节级 tokenizer 为什么能表示未见过的字符？
- 文本归一化为什么可能让编码后无法逐字还原原输入？

<a id="pre-003"></a>
### PRE-003 · 词表越大越好吗，扩词表有哪些代价？

**L2**

#### 答案

词表更大并不一定更好，它是在序列长度、模型规模和每个 token 的学习充分度之间取舍。大词表能把更多常见片段作为一个 token，同一段文本通常会更短，注意力和生成步数可能减少；但输入向量表、输出预测矩阵以及每一步词表打分都会变大。

如果词表有 V 个 token、隐藏维度是 d，输入 embedding 大约有 Vd 个参数；输出 LM head 若另用一张矩阵，还要再增加约 Vd，共享权重时可以省掉这份重复，bias 另算。比如增加很多罕见专有名词，虽然某些文本变短，这些新行却可能只有很少训练样本，表示并不可靠。对多语言或代码任务，应分别统计每字符或每字节的 token 数，并同时测效果、显存和延迟。

给已有模型扩词表还涉及兼容问题：旧 token 的 ID 不能乱移，新增输入与输出行需要合理初始化并得到训练，改变切分后也会改变训练序列分布。使用参数高效微调时，要确认新增 embedding 与输出行确实可更新，否则新增 token 只是没有学好的随机参数。只有当某领域的切分明显低效、数据足够且综合收益值得时，扩词表才是合理选择；token 数变少本身不证明模型理解更好。

#### 易错点

- 扩词表需要训练新增行和检查兼容性，并不是免费优化。
- 文本 token 变少，不直接证明语义理解变好。

#### 追问

- 怎样判断中文切分效率已经差到值得扩词表？
- 参数高效微调时，怎样确保新增 embedding 和输出行能够更新？

<a id="topic-2"></a>
## 序列任务与经典 NLP

<a id="bas-002"></a>
### BAS-002 · POS、NER 如何做序列标注？HMM、独立分类与 CRF 有什么区别？

**L2**

#### 答案

序列标注就是给一句话中的每个位置贴标签：POS 标注名词、动词等词性，NER 标注人名、地点等实体及其边界。比如“北京大学”可标成一个机构实体，BIO 标签用 B 表示实体开头、I 表示内部、O 表示实体之外；BIOES 则进一步区分结束和单字实体。

HMM 是生成式模型，联合建模标签之间怎样转移，以及标签怎样产生观察到的词。神经网络加独立分类器则先理解上下文，再在每个位置选得分最高的标签，简单直接，但可能输出“I-机构”前面没有实体开头这样的非法序列。CRF，即条件随机场，会同时考虑每个位置的标签分数和相邻标签的转移分数，对整条标签序列计算概率。公式里的 e 是位置分数，A 是转移分数，s 是两者相加得到的序列总分。

训练 CRF 时，需要把正确序列得分与所有可能序列的总得分比较，分母可以用前向动态规划高效计算；预测时改成寻找最大值，用 Viterbi 回溯出最佳标签。长度为 T、标签数为 K 时，标准线性链算法约为 O(TK²)。它能约束标签衔接，却仍依赖编码器理解语义。实现还要排除 padding、对齐子词标签，并报告实体级 F1，因为每个字大多标对也可能把整个实体边界标错。若实体有重叠或嵌套，单条 BIO/BIOES 标签往往表达不了，需要多层标注或按候选跨度预测。

```math
\begin{aligned}s(x,y)&=\sum_t e_t(y_t)+\sum_t A_{y_{t-1},y_t}\\p(y\mid x)&=\frac{e^{s(x,y)}}{\sum_{y'}e^{s(x,y')}}\\\mathcal L&=-s(x,y)+\log\sum_{y'}e^{s(x,y')}\end{aligned}
```

#### 易错点

- CRF 对整条标签序列归一化，不是每个位置各做一次 softmax。
- 只报逐 token 准确率，可能掩盖整个实体边界标错的问题。

#### 追问

- CRF 训练的前向递推和预测的 Viterbi 递推，求和与取最大值分别在哪里？
- 一个词被 tokenizer 拆成多个子词后，怎样对齐原来的标签？

<a id="bas-003"></a>
### BAS-003 · 成分句法与依存句法分别预测什么，怎样评价解析结果？

**L2**

#### 答案

成分句法回答“哪些词组成一个短语”，依存句法回答“哪个词依赖哪个中心词，以及是什么关系”。比如“学生读书”，成分分析会组成名词短语和动词短语；依存分析会把“学生”连到“读”，标为主语，把“书”连到“读”，标为宾语。它们是同一句话的两种结构描述。

成分分析通常给连续片段打分，再把片段组合成合法的树。上下文无关文法规定短语能怎样展开，CKY 算法用动态规划重复利用已经分析好的小片段；标准二元文法下时间约为 O(n³|G|)，n 是词数，|G| 是文法规模。依存分析可以逐步执行连接动作，也可以先给候选中心词与依赖词的边打分，再求一棵满足每词一个中心词、无环且连通的树。不能让每个词独立选最高分边后就结束，否则可能得到环。

成分分析通常比较预测与真实的带标签短语跨度，报告 F1。依存分析中，UAS 看中心词是否选对，LAS 还要求关系标签正确；公式里 h 是中心词，r 是关系，N 是参与评价的词数。比较结果必须统一分词、标点是否计入等口径。显式句法有助于抽取或结构约束，但大模型并不需要先调用一个句法分析器才能处理语言。

```math
\mathrm{UAS}=\frac{\#\{i:\hat h_i=h_i\}}{N},\qquad\mathrm{LAS}=\frac{\#\{i:\hat h_i=h_i,\hat r_i=r_i\}}{N}
```

#### 易错点

- 成分树标注短语跨度，依存树标注中心词关系，二者不能直接混用。
- 比较 UAS、LAS 时必须统一分词与标点计分规则。

#### 追问

- 为什么让每个词独立选择最高分中心词，可能得不到合法依存树？
- 投射和非投射依存树的边是否交叉，对解析有什么影响？

<a id="bas-008"></a>
### BAS-008 · CNN、RNN 与 Transformer 在序列建模、长依赖和并行性上怎样比较？

**L1**

#### 答案

CNN、RNN 和 Transformer 的主要区别，是信息怎样在序列位置之间传递。CNN 用同一组卷积核扫描局部片段，擅长识别邻近模式；RNN 把上一时刻的状态传给下一时刻，边读边记；Transformer 用注意力让一个位置直接读取其他可见位置，再做逐位置的特征变换。

例如要把句首的人名和句尾的动作联系起来，普通 CNN 需要叠加多层扩大感受野，空洞卷积可以间隔取样、以较少层覆盖更远位置。RNN 的信息要经过中间每一步，梯度也沿这条长路径传播，容易衰减或放大。全注意力可以直接连接远处位置，路径短，但长度为 n 时要考虑约 n² 对位置关系，长序列的计算和内存成本高。公式中的 Q、K、V 分别用于查询、匹配和提供内容，M 控制哪些位置可见。

训练时，CNN 各位置和 Transformer 的注意力可以大量并行，RNN 的时间步通常有前后依赖。自回归 Transformer 虽然能并行训练一整句，生成时仍要等前一个 token 才能确定下一步输入。若任务需要持续接收流式输入或在边缘设备运行，RNN、CNN 依然可能合适；不存在脱离长度、数据和硬件的固定性能排名。

```math
\begin{aligned}h_t^{\rm RNN}&=\phi(W_xx_t+W_hh_{t-1}+b)\\h_t^{\rm CNN}&=\phi\!\left(\sum_{j=-k}^{k}W_jx_{t+j}+b\right)\\H^{\rm attn}&=\mathrm{softmax}(QK^\top/\sqrt{d_k}+M)V\end{aligned}
```

#### 易错点

- Transformer 的位置并行训练，不代表自回归生成的各步也能并行。
- 模型效果取决于数据、长度和硬件，某次实验不能给出永久排名。

#### 追问

- 空洞卷积怎样用间隔取样扩大可见范围？
- 双向 RNN 使用了未来信息，对在线预测有什么限制？

<a id="bas-009"></a>
### BAS-009 · LSTM 与 GRU 怎样缓解普通 RNN 的梯度问题，有何结构区别？

**L2**

#### 答案

LSTM 和 GRU 用可学习的门控制“保留多少旧信息、写入多少新信息”，从而缓解普通 RNN 长序列中的梯度问题。普通 RNN 每一步都变换旧状态，反传时要连续乘很多局部导数，持续偏小会让梯度消失，持续偏大则可能爆炸。

LSTM 额外维护一条 cell state，可以理解为长期记忆通道。遗忘门 f 决定保留多少旧记忆，输入门 i 决定写入多少候选记忆，输出门 o 决定把多少记忆暴露为当前 hidden state。公式中的逐元素乘号表示每个维度分别控制，核心更新是旧记忆与新内容相加；沿直接记忆通道，梯度不必每一步都经过完整的非线性变换。例如读到“他出生在北京”时可以暂存地点，到后面的问句再使用。

GRU 通常只维护一个状态，重置门 r 控制生成候选时参考多少旧状态，更新门 z 在旧状态和候选之间插值，因此结构更简洁、参数通常更少。这里约定 z 大表示更多采用候选，其他实现可能使用相反定义。重置门也可能放在线性变换之前或之后，两者一般不等价，手写或对照框架时要固定约定。门控只是改善优化条件，不能保证无限长记忆，梯度裁剪也只能控制爆炸。两者在时间轴仍然递归；双向版本能看未来上下文，不能直接用于未知未来的在线预测。

```math
\begin{aligned}c_t&=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\quad h_t=o_t\odot\tanh(c_t)\\\tilde h_t^{\rm GRU}&=\tanh(W_xx_t+W_h(r_t\odot h_{t-1})+b)\\h_t^{\rm GRU}&=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t\end{aligned}
```

#### 易错点

- 门控能改善梯度传播，但不能保证任意长度都不会梯度消失。
- 不同 GRU 实现可能把 z 与 1−z 的角色对调，解释前要固定约定。

#### 追问

- LSTM 为什么把长期记忆 cell 与当前输出 hidden 分开？
- 截断时间反向传播会丢失哪些更远历史的梯度贡献？

<a id="topic-3"></a>
## 数学、概率与统计

<a id="tfm-013"></a>
### TFM-013 · 交叉熵、KL 散度与 perplexity 的关系和实现注意点是什么？

**L1** · 阿里巴巴

#### 答案

交叉熵衡量模型给真实目标分布分配概率的好坏，KL 散度衡量两个分布的失配，困惑度则把语言模型的平均交叉熵转成更直观的数值。设 p 是目标、q 是预测，交叉熵 H(p,q) 可以拆成目标本身的熵 H(p) 加 KL(p∥q)。例如目标本来就有多种合理答案，即使模型完美匹配，交叉熵也不一定为零。

当 p 固定时，目标熵不随模型变化，因此最小化交叉熵与最小化这个方向的 KL 有相同梯度；如果 p 也参与训练，或教师输出没有停止梯度，就不能忽略它的变化。KL 非负但通常不对称，也不是距离度量。单样本 one-hot 目标的熵为零，两者都等于正确类别概率的负对数；这不意味着真实数据的条件熵为零。对归一化软标签，logits 梯度仍为 q−p，必须区分 logits 的梯度与概率自身的梯度。

公式中的有效 token 数用于计算平均负对数似然，自然对数下取指数得到 PPL。例如平均损失为 ln 4，PPL 就是 4，可粗略理解为平均有四个等可能选择的困难度。不同 batch 长度要按有效 token 加权，比较模型还必须统一 tokenizer、数据与窗口。CrossEntropyLoss 通常输入 logits；KLDivLoss 通常输入 log q、目标 p，顺序与数学记号不同。若目标在某位置为正而预测概率为零，交叉熵和 KL 都会无穷；实现应用稳定的对数计算。低 PPL 也不能替代事实性和指令遵循评价。

```math
\begin{aligned}H(p,q)&=H(p)+D_{\rm KL}(p\|q)\\\nabla_\theta H(p,q_\theta)&=\nabla_\theta D_{\rm KL}(p\|q_\theta)\quad(p\ \text{fixed})\\\mathrm{PPL}&=\exp\!\left(-\frac{1}{N_{\rm valid}}\sum_{t\in\mathcal T_{\rm valid}}\log p_\theta(x_t\mid x_{\lt t})\right)\end{aligned}
```

#### 易错点

- 交叉熵与 KL 相差目标熵，梯度等价需要目标分布固定。
- KLDivLoss 的接口顺序与归约不能机械当作数学 KL 的默认定义。

#### 追问

- 怎样从 softmax 导数推导交叉熵的 logits 梯度 q−p？
- 目标分布也可训练或使用双向 KL 时，优化目标如何改变？

<a id="tfm-017"></a>
### TFM-017 · 熵衡量什么？离散熵、条件熵与模型输出熵怎样区分？

**L1**

#### 答案

熵衡量的是一个概率分布有多难预测，分布越分散，平均不确定性越高。某事件的信息量是 −log p：确定会发生的事没有惊讶，罕见事件发生时信息量大。把所有事件的信息量按概率求平均，就是离散熵。用以 2 为底的对数，单位是 bit；用自然对数，单位是 nat。

例如公平硬币的两面各有一半概率，熵为 1 bit；若某一面概率为 1，熵为 0。固定有 V 个类别时，离散熵在 0 到 log V 之间，均匀分布达到最大值，零概率项按极限视为零。不能只看最大类别概率就判断整个分布的熵，还要看剩余概率如何分配。条件熵是在知道 Y 后，对各种 Y 情况的剩余不确定性取平均，公式里的链式法则把联合不确定性拆成先知道 Y、再知道 X 的两部分。平均上条件信息不会增加不确定性，但某个具体条件下的熵可能更大。

大模型日志里的输出熵描述模型自身有多犹豫，训练交叉熵描述它对真实目标预测得怎样，两者不能混用。模型可以自信地答错，因此低输出熵并不等于准确；温度和候选过滤也会改变熵，比较时要说明计算前后的分布与平均范围。连续变量的微分熵还会随单位和尺度变化，甚至可以为负，不能直接套用离散熵的上下界。

```math
\begin{aligned}H(p)&=-\sum_i p_i\log p_i\\H(X\mid Y)&=\sum_y p(y)H(p(X\mid Y=y))\\H(X,Y)&=H(Y)+H(X\mid Y)\end{aligned}
```

#### 易错点

- 模型低熵只表示自信，不能推出准确；真实数据熵也不同于预测熵。
- 条件熵下降是平均结论，不能保证每个具体条件都降低不确定性。

#### 追问

- 一个自信但错误的预测，为什么可以有很低的输出熵？
- 连续均匀分布区间缩小时，为什么微分熵可能变成负值？

<a id="tfm-018"></a>
### TFM-018 · 矩阵的秩与特征值如何计算，和奇异值有什么关系？

**L2**

#### 答案

矩阵的秩表示它包含多少个独立方向，特征值表示方阵在某些方向上怎样缩放，奇异值则适用于任何矩形矩阵，描述它沿主要输入方向的伸缩强度。秩可以通过消元后主元数计算，也等于精确非零奇异值的个数；对 m×n 矩阵，秩加上零空间维数等于 n。

特征值满足 Av=λv，v 不能是零向量。手算小方阵可以先解 det(λI−A)=0，再求对应方向；例如上三角矩阵 [[2,1],[0,3]] 的特征值是 2 和 3，秩为 2。一般矩形矩阵没有这种标准特征值定义。一般方阵也不能靠数非零特征值直接得到秩：[[0,1],[0,0]] 的特征值全为零，秩却是 1。只有可对角化等合适条件下，按重数数非零特征值才成立；满秩、行列式非零、所有特征值非零对任意方阵仍等价。

奇异值分解写作 A=UΣVᵀ，U、V 给出正交方向，Σ 给出非负的奇异值。按降序对应时，AᵀA 的特征值是奇异值的平方，其余维度补零；对实对称方阵，奇异值等于特征值绝对值。大矩阵一般由数值库求解，特征值可用 QR、Schur 等过程，对称矩阵宜用专用算法；直接构造 AᵀA 会放大条件数问题。浮点秩需要结合最大奇异值与容差判断，接近零不等于精确零。

截断 SVD 保留最大的 r 个奇异值，在谱范数和 Frobenius 范数下给出最佳秩 r 近似，谱范数误差是第 r+1 个奇异值，Frobenius 误差是尾部奇异值平方和的平方根。LoRA 则训练 BA 形式的低秩增量，其秩至多 r，不要求原权重低秩，也不保证学出的实际秩恰好是 r。矩阵近似误差最优，与下游任务损失最小仍是两个问题。

```math
\begin{aligned}\det(\lambda I-A)&=0,\quad(A-\lambda I)v=0,\quad v\ne0\\\mathrm{rank}(A)&=\#\{i:\sigma_i(A)\gt 0\}\\\lambda_i(A^\top A)&=\sigma_i(A)^2,\quad1\le i\le p=\min(m,n)\\\lambda_i(A^\top A)&=0,\quad p\lt i\le n\\A=P\Lambda P^{-1}&\implies\mathrm{rank}(A)=\#\{i:\lambda_i\ne0\}\\\mathrm{rank}(BA)&\le r\end{aligned}
```

#### 易错点

- 一般矩阵的秩不能靠非零特征值个数直接得到，非方阵也没有这种标准特征值定义。
- 奇异值不等于一般特征值；LoRA 的配置 r 只是增量秩的上界。

#### 追问

- 直接构造 AᵀA 求奇异值，为什么可能放大条件数和数值误差？
- 低秩近似、数值秩与 LoRA 的秩上界，分别描述什么？

<a id="topic-4"></a>
## 深度学习与优化基础

<a id="bas-010"></a>
### BAS-010 · Sigmoid、Tanh、ReLU、GELU 与 SiLU 的公式和梯度特点是什么？

**L1**

#### 答案

激活函数让网络能表达非线性关系；如果所有层都只做线性变换，再多层也可以合成一个线性层。Sigmoid 把输入压到 0 到 1，适合表示二元概率或门；Tanh 压到 −1 到 1，并以零为中心。它们在输入绝对值很大时趋于饱和，导数接近零，所以深层训练可能难以传回梯度。

ReLU 保留正值、把负值置零，正侧导数为 1、负侧为 0，计算简单。但某个单元若长期收到负输入，就可能一直没有梯度；Leaky ReLU 用一个小的负侧斜率，PReLU 则把这个斜率也训练出来。ReLU 在零点数学上不可导，框架会采用具体的实现约定。ELU 在负侧采用指数曲线并趋于饱和，RReLU 则在训练时随机取负侧斜率，推理常用其期望值，这些机制并不相同。

GELU 的 xΦ(x) 用标准正态累积分布 Φ 平滑调节输入，SiLU 的 xσ(x) 用 sigmoid 调节，二者允许部分负值通过，也都是非单调函数。Swish 写成 xσ(βx)，β=1 时等于 SiLU。Transformer 常用它们，但 SwiGLU 是两个线性投影分支相乘的门控结构，不能简单当成换一个激活名字。选择时还要一起检查初始化、归一化和中间维度，平滑或允许负值并不自动保证收敛更快。

```math
\begin{aligned}\sigma(x)&=(1+e^{-x})^{-1},\quad\sigma'(x)=\sigma(x)(1-\sigma(x))\\\tanh'(x)&=1-\tanh^2(x)\\\mathrm{ReLU}(x)&=\max(0,x)\\\mathrm{GELU}(x)&=x\Phi(x)\\\mathrm{SiLU}(x)&=x\sigma(x),\quad\mathrm{SiLU}'(x)=\sigma(x)+x\sigma(x)(1-\sigma(x))\end{aligned}
```

#### 易错点

- SwiGLU 是多投影分支相乘的门控结构，不只是一个 SiLU 激活。
- ReLU 在零点不可导，框架采用的是实现约定，不能当作唯一数学导数。

#### 追问

- GELU 的近似公式为什么与使用正态累积分布的精确式不同？
- 怎样从激活与梯度日志区分 ReLU 单元失活和梯度爆炸？

<a id="bas-011"></a>
### BAS-011 · SGD、Momentum、AdaGrad、RMSProp 与 Adam 的更新规则怎样理解？

**L2**

#### 答案

这些优化器都用梯度降低损失，区别在于是否记住过去的方向，以及是否为不同参数调节步长。小批量 SGD 直接沿当前批次梯度的反方向走；Momentum，也就是动量，会累积过去的方向，像有惯性的球一样减少左右振荡，在持续一致的方向上前进。

AdaGrad 为每个坐标累积过去的平方梯度，经常被更新的坐标分母更大、有效步长更小，适合一些稀疏特征，但长期累积可能使步长过早缩小。RMSProp 把累积改成指数滑动平均，旧信息会逐渐淡出。Adam 同时维护一阶平均 m 和平方梯度平均 v：用 m 判断方向，用 v 的平方根调节各坐标尺度。公式中 g 是当前梯度，β 控制记忆长度，η 是学习率，ε 防止分母过小。

m、v 从零开始会在早期偏小，所以 Adam 除以对应的 1−β 的 t 次方做偏差校正。这里 v 是未减均值的二阶矩，并不是梯度方差，也不等于精确曲率。梯度累积时，应汇总多个微批次后再更新优化器状态；每个微批次都更新 m、v 会改变算法。Nesterov 动量使用前瞻思想，AdaDelta还结合更新量的滑动统计；选型要结合任务验证，权重衰减是否解耦、ε 放在哪里等实现细节也会影响结果。

```math
\begin{aligned}m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t\\v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2\\\hat m_t&=m_t/(1-\beta_1^t),\quad\hat v_t=v_t/(1-\beta_2^t)\\\theta_{t+1}&=\theta_t-\eta_t\hat m_t/(\sqrt{\hat v_t}+\epsilon)\end{aligned}
```

#### 易错点

- Adam 的 v 是平方梯度平均，不是已经减去均值的方差。
- 梯度累积应在汇总后更新优化器；每个微批都更新状态会改变算法。

#### 追问

- 一阶与二阶矩从零开始，为什么早期需要偏差校正？
- AdaGrad 不断累积平方梯度，为什么会让有效学习率减小？

<a id="bas-012"></a>
### BAS-012 · 反向传播怎样使用链式法则？线性层、MSE 与 Softmax 交叉熵如何求梯度？

**L1**

#### 答案

反向传播就是把“损失对最终输出的影响”沿计算图一步步传回每个参数，核心规则是链式法则。某个参数若通过多条路径影响损失，需要把各条路径的贡献相加。自动微分通常直接计算上游梯度与局部导数的乘积，不必建立一个巨大的完整 Jacobian 矩阵，也不是靠扰动参数做数值差分来训练。

在线性层 Y=XW+b 中，若 X 是 n×d 的输入、W 是 d×k 的权重，Y 与上游梯度 G 都是 n×k。权重梯度 XᵀG 是 d×k，输入梯度 GWᵀ 是 n×d；同一个 b 用于所有样本，因此它的梯度要沿样本维相加。可以把它理解为：某输入特征出现得越强、对应输出越需要纠正，这个连接的更新贡献就越大。

MSE 的预测梯度与预测误差成正比，取平均还会除以元素数；softmax 与交叉熵合起来，对单样本 logits 的梯度是 p−y，p 是预测概率、y 是归一化目标。batch 或有效 token 平均会再引入相应除数。屏蔽某位置的损失不代表该输入完全没梯度，它仍可能通过注意力影响被监督位置。排错时可在小张量上做有限差分核对，并检查梯度清零、detach、原地修改和共享参数贡献是否正确累加。

```math
\begin{aligned}Y&=XW+b,\quad G=\partial\mathcal L/\partial Y\\\nabla_W\mathcal L&=X^\top G,\quad\nabla_X\mathcal L=GW^\top,\quad\nabla_b\mathcal L=\sum_iG_i\\\mathcal L_{\rm MSE}&=\frac1n\sum_i(\hat y_i-y_i)^2\\\frac{\partial\mathcal L_{\rm CE}}{\partial z_i}&=p_i-y_i\quad\text{(single example)}\end{aligned}
```

#### 易错点

- 共享参数的多次使用贡献必须累加，不能只留下最后一次。
- 求和、按样本平均和按有效 token 平均有不同梯度尺度。

#### 追问

- softmax 与交叉熵合并后，为什么 logits 梯度能简化成概率减目标？
- 有限差分步长太大或太小，各会带来什么梯度核对误差？

<a id="bas-013"></a>
### BAS-013 · 过拟合、欠拟合与数据泄露怎样区分，L1/L2、早停和 Dropout 分别做什么？

**L2**

#### 答案

欠拟合通常表现为训练数据也没学好，过拟合通常表现为训练好、独立验证差，而数据泄露是验证过程提前看到了本不该知道的信息。三者需要分开排查：训练和验证都差也可能是优化失败，训练验证差距大也可能来自分布变化，不能只看两条曲线就确定原因。

比如把同一篇文章的相邻片段随机分到训练与验证，模型很容易借助几乎重复的内容获得虚高成绩；在全数据上拟合词表统计或归一化，再切分，也可能泄露。应按文档、用户、实体或时间设置合适的分组边界，并保留没有参与调参的最终测试集。反复根据测试集选模型，实际上已经把测试集用成了验证集。

正则化帮助控制模型对训练数据的过度适应。L2 在数据损失上加参数平方惩罚，倾向于持续缩小权重；L1 加绝对值惩罚，可能使部分坐标变零，零点用次梯度理解。早停按验证趋势选择 checkpoint，Dropout 在训练时随机屏蔽并缩放部分激活，使模型减少对固定组合的依赖。λ 是正则强度，太大会妨碍学习。自适应优化器里的 L2 与解耦权重衰减并不等价；这些方法也不能修复泄露或自动解决分布漂移，因此应先核对数据与评测，再调整容量、训练时长和正则强度。

```math
\mathcal L_{L_1}=\mathcal L_{\rm data}+\lambda\lVert\theta\rVert_1,\qquad\mathcal L_{L_2}=\mathcal L_{\rm data}+\frac\lambda2\lVert\theta\rVert_2^2
```

#### 易错点

- 反复用于选择模型的验证集，不能继续当作最终独立测试集。
- 训练验证差距不只由模型过大导致，也可能来自数据分布或评测问题。

#### 追问

- 同一文档的相邻片段分进训练和验证，为什么容易产生泄露？
- L1 在参数为零时不可导，怎样用次梯度描述更新？

<a id="bas-014"></a>
### BAS-014 · Xavier、He 初始化与 embedding 乘 √d_model 的目的是什么？

**L2** · 腾讯

#### 答案

初始化是为了让信号和梯度经过很多层之后仍保持合适尺度；Xavier 与 He 初始化根据连接数量和激活特点选择权重方差。若一层把很多近似独立、零均值的输入相加，输出方差约等于输入连接数乘权重方差再乘输入方差。权重太大会逐层放大，太小则逐层衰减。

fan-in 是一个输出接收的输入数，fan-out 是一个输入连到的输出数。Xavier 同时照顾前向和反向尺度，方差约为 2/(fan-in+fan-out)，适合作近似线性或对称激活的起点；He 考虑 ReLU 会截掉约一半输入，用约 2/fan-in 补偿。例如输入维度翻倍时，单个权重的典型幅度应该缩小，而不是保持不变。正态或均匀分布都能实现同一目标方差。

原始 Transformer 把输入 embedding 乘 √d_model 后再加位置编码，用于配合其参数化并调节两种表示的相对尺度。这个操作与注意力分数除以 √d_k 是不同位置、不同目的的设计，现代模型也不一定缩放输入 embedding。独立性分析会受到残差、门控、归一化和训练后相关性的影响，因此初始化公式不是稳定性保证。实际可检查初始各层激活的均方根、logits 及梯度范数，再判断是否需要深度相关的残差缩放。

```math
\begin{aligned}\mathrm{Var}(Wx)_i&\approx\mathrm{fan\_in}\mathrm{Var}(W_{ij})\mathrm{Var}(x_j)\\\mathrm{Var}(W)_{\rm Xavier}&\approx\frac{2}{\mathrm{fan\_in}+\mathrm{fan\_out}}\\\mathrm{Var}(W)_{\rm He}&\approx\frac{2}{\mathrm{fan\_in}}\end{aligned}
```

#### 易错点

- embedding 乘 √d_model 是特定参数化，不是所有 Transformer 的必需操作。
- 初始化的独立方差分析不保证训练后每层仍满足同样假设。

#### 追问

- 深度增加、残差分支不断相加时，为什么可能需要额外尺度控制？
- 怎样为均匀分布和正态分布设置相同的初始化方差？

<a id="bas-016"></a>
### BAS-016 · 分类为什么通常用交叉熵而非 MSE？MSE 在数学上可行吗？

**L2** · 字节跳动

#### 答案

分类通常用交叉熵，因为它直接惩罚“给正确类别的概率太低”，而且模型自信地分错时仍能获得明显的纠正梯度；MSE 在数学上也可以做分类，但必须说明它比较的是什么。对多分类，先把 logits 转成概率 p，再与 one-hot 目标 y 比较，交叉熵就是正确类别概率的负对数。

例如真实是猫，模型却把狗的概率推到接近 1。交叉熵对 logits 的梯度为 p−y，猫的分数会被往上推、狗的分数往下拉。概率向量上的平方损失属于 Brier 类目标，也可以鼓励学习真实概率，但反传时还要乘 softmax 的导数；概率已经在错误类别上饱和时，梯度可能变小。公式里的平方损失取半和，因此梯度常数与按类别平均的实现不同，不能单凭损失数字更小就判断优化更好。

另一种做法是把类别编号直接回归，这对无序类别通常不合理：编号 1 与 2 看起来比 1 与 9 更近，但类别本身没有这种距离。连续值回归则很适合 MSE。实现多分类时，PyTorch 的 CrossEntropyLoss 接收原始 logits，内部稳定地计算 log-softmax，不要重复 softmax；二分类或多标签常用各类别独立的 sigmoid 和二元交叉熵。最终可根据校准、标签噪声和训练动态比较损失，而不能把 MSE 一概说成错误。

```math
\begin{aligned}p&=\mathrm{softmax}(z),\quad L_{\mathrm{CE}}=-\sum_i y_i\log p_i\\\frac{\partial L_{\mathrm{CE}}}{\partial z_j}&=p_j-y_j\\L_{\mathrm{sq}}&=\frac12\sum_i(p_i-y_i)^2\\\frac{\partial L_{\mathrm{sq}}}{\partial z_j}&=p_j\left[(p_j-y_j)-\sum_i p_i(p_i-y_i)\right]\end{aligned}
```

#### 易错点

- 概率向量的平方损失可以做分类，但回归类别编号会引入不合理的距离。
- 公式取半和；改成按类别平均等归约时，梯度常数也要变化。

#### 追问

- 模型自信地分错二分类时，CE 与概率 MSE 的 logits 梯度有什么不同？
- 目标换成软标签或标签平滑后，交叉熵梯度怎样变化？

## 参考资料

- [jieba 官方算法说明](https://github.com/fxsjy/jieba)
- [Speech and Language Processing：隐马尔可夫模型](https://web.stanford.edu/~jurafsky/slp3/A.pdf)
- [Speech and Language Processing：序列标注](https://web.stanford.edu/~jurafsky/slp3/18.pdf)
- [Speech and Language Processing：成分句法](https://web.stanford.edu/~jurafsky/slp3/19.pdf)
- [Speech and Language Processing：依存句法](https://web.stanford.edu/~jurafsky/slp3/20.pdf)
- [Speech and Language Processing：词嵌入](https://web.stanford.edu/~jurafsky/slp3/5.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/html/1810.04805v2)
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
- [Speech and Language Processing：N-gram语言模型](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- [Speech and Language Processing：RNN与LSTM](https://web.stanford.edu/~jurafsky/slp3/14.pdf)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Speech and Language Processing：神经网络](https://web.stanford.edu/~jurafsky/slp3/6.pdf)
- [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415)
- [Fast and Accurate Deep Network Learning by Exponential Linear Units](https://arxiv.org/abs/1511.07289)
- [Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [TextRank: Bringing Order into Text](https://aclanthology.org/W04-3252/)
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch MSELoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html)
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)
- [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162.pdf)
- [Hugging Face：WordPiece tokenization](https://huggingface.co/learn/llm-course/chapter6/6)
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
- [torch.nn.CrossEntropyLoss — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)
- [MIT 6.441 Chapter 1: Entropy and Divergence](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/2243edffb30f57181ed97dcb77691580_MIT6_441S16_chapter_1.pdf)
- [torch.nn.KLDivLoss — PyTorch 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.KLDivLoss.html)
- [TRL SFT Trainer — loss, labels and packing](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Stanford EE263: Eigenvectors and diagonalization](https://ee263.stanford.edu/lectures/eig.pdf)
- [Stanford EE263 Lecture 15: Symmetric matrices and SVD](https://web.stanford.edu/class/archive/ee/ee263/ee263.1082/lectures/symm.pdf)
- [torch.linalg.matrix_rank — PyTorch 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.matrix_rank.html)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [LAPACK Users' Guide: Eigenvalues, Eigenvectors and Schur Factorization](https://www.netlib.org/lapack/lug/node50.html)
- [PyTorch：torch.linalg.svd](https://docs.pytorch.org/docs/main/generated/torch.linalg.svd.html)
