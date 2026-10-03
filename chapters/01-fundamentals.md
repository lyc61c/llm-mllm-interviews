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

<a id="topic-1"></a>
## 分词与文本表示

<a id="bas-001"></a>
### BAS-001 · 中文分词有哪些难点？jieba 的 DAG、动态规划和 HMM 如何配合？

**L1**

#### 答案

中文文本缺少天然空格边界，难点是切分标准、歧义与未登录词，例如“研究生命起源”不能仅靠最长匹配决定词界。jieba 精确模式先用前缀词典枚举各位置可能结束的词，构成有向无环图，再以词频估计的对数概率做动态规划，选全句得分最大的路径。它优化整体路径，局部最长词不一定最好。

对词典无法可靠覆盖的片段，默认可用字符级 HMM，以 B/M/E/S 表示词首、词中、词尾和单字词，Viterbi 找最高概率标签路径。搜索模式会进一步切开长词以提高召回，全模式则枚举候选，通常不适合作唯一精确切分。业务中加入领域词典、检查频率与歧义例句，并分别测词界 F1 和下游检索效果；词级分词与 LLM 子词 tokenizer 的目标不同。

$$
D(i)=\max_{j\in\operatorname{DAG}(i)}\{\log P(x_{i:j})+D(j+1)\},\qquad D(n+1)=0
$$

#### 易错点

- 把 jieba 的最大概率路径等同于最长词优先。
- 把中文分词的词界直接当作模型 tokenizer 的 token 边界。

#### 追问

- 为什么新增领域词频会影响相邻词的切分？
- 词典法与序列标注法怎样处理未登录词？

<a id="bas-004"></a>
### BAS-004 · one-hot、静态词向量与上下文 embedding 的区别是什么？

**L1**

#### 答案

one-hot 为每个词分配独立坐标，没有编码语义相似性。静态词向量用一个查表矩阵把词映射到低维密集向量，共现或上下文预测目标使相似使用环境的词接近；同一个词形通常只有一个向量，难直接区分一词多义。上下文 embedding 则由序列编码器计算，词在“苹果发布新品”和“吃苹果”中能得到不同表示。

计数式向量可由共现矩阵、PPMI 和低秩分解得到，预测式向量可用 Word2Vec 学习；它们都依赖语料和上下文定义。ELMo 拼接分别训练的双向语言模型特征，BERT 在每层联合融合双向上下文，GPT 隐状态只依赖可见前缀。词向量平均或原始 BERT 的 CLS 并不自动成为高质量句向量，检索通常还需句级训练、pooling 与归一化。余弦相似度描述所选表示空间中的角度相似，不能单独证明事实一致。

$$
\begin{aligned}e(w)&=E^\top\operatorname{onehot}(w)\\h_t&=f_\theta(x_{1:T})_t\\\operatorname{cos}(u,v)&=\frac{u^\top v}{\lVert u\rVert\lVert v\rVert}\end{aligned}
$$

#### 易错点

- 把词 embedding、上下文 token 隐状态和检索用句向量混为一谈。
- 认为向量加减类比对任何语料与词汇都成立。

#### 追问

- 为什么静态词向量很难表达多义词？
- 均值 pooling 为什么需要正确排除 padding？

<a id="bas-005"></a>
### BAS-005 · Word2Vec 的 CBOW 和 Skip-gram 如何训练，有哪些关键细节？

**L1**

#### 答案

CBOW 用窗口内上下文词向量的平均或求和预测中心词，Skip-gram 用中心词分别预测窗口中的上下文词。两者都有输入向量表和输出向量表，并非仅有一个 embedding 矩阵。完整 softmax 下，Skip-gram 对每个中心—上下文词对最大化条件概率；实际可换成负采样或层次 softmax 以减少输出端计算。

CBOW 聚合上下文，计算较便宜；Skip-gram 从一个中心词生成多个训练对，常用于学习较少见词的表示，但优劣取决于数据与训练配方。常见技巧包括高频词 subsampling、随机窗口宽度和词频阈值，减少功能词支配训练；均值聚合通常丢失词序。训练后可取输入向量或组合两表，需明确口径。增量训练要正确更新词表和频率，并保留旧领域数据，否则新语料会改变已有向量空间；这不等同于冻结老向量只添新词。

$$
\begin{aligned}p(o\mid c)&=\frac{\exp(u_o^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}\\\mathcal L_{\rm SG}&=-\sum_{(c,o)}\log p(o\mid c)\\\bar v_C&=\frac1{|C|}\sum_{w\in C}v_w\quad(\mathrm{CBOW})\end{aligned}
$$

#### 易错点

- 把 Skip-gram 说成“多个上下文预测中心词”。
- 把窗口共现关系等同于全局语义真值。

#### 追问

- 为什么需要输入、输出两张向量表？
- 随机缩小窗口如何改变近邻词的训练权重？

<a id="bas-006"></a>
### BAS-006 · 负采样与层次 Softmax 怎样加速 Word2Vec，是否等价于原始 Softmax？

**L2**

#### 答案

完整 softmax 需要遍历整个输出词表。层次 softmax 用二叉树的路径概率表示词概率，只更新目标词路径上的节点；Huffman 树让高频词路径较短，典型路径长度约 log|V|，并保持一个归一化的分布。负采样把观察到的词对作为正例，从噪声分布抽 K 个负例，对每个词对做 sigmoid 二分类，只更新这些输出向量。

负采样不是完整 softmax 的精确无偏梯度，也不直接产生所有词的归一化条件概率；它换了训练目标，以高效学习词表示。原始 Word2Vec 常用词频的 3/4 次方构造噪声分布，K 和噪声质量影响效果。负例不等于“语义上完全无关的词”，误负例应通过统计而非绝对断言理解。层次 softmax 的树结构、负采样的随机种子和输入/输出向量选择都会影响复现。

$$
\begin{aligned}\mathcal L_{\rm NS}&=-\log\sigma(u_o^\top v_c)-\sum_{k=1}^{K}\log\sigma(-u_{n_k}^\top v_c)\\n_k&\sim P_n(w)\propto f(w)^{3/4}\\p_{\rm HS}(w\mid c)&=\prod_{j\in\operatorname{path}(w)}\sigma(b_j u_j^\top v_c),\quad b_j\in\{-1,1\}\end{aligned}
$$

#### 易错点

- 宣称负采样是原始多分类 softmax 的完全等价实现。
- 把所有抽出的负例都视为确定无关。

#### 追问

- 层次 softmax 为什么能够定义归一化词概率？
- SGNS 与 PMI 矩阵分解有什么联系？

<a id="bas-007"></a>
### BAS-007 · N-gram 与神经语言模型怎样估计序列概率，平滑解决什么问题？

**L1**

#### 答案

语言模型为 token 序列赋予概率，概率链式法则本身是精确的；N-gram 额外作有限阶 Markov 近似，只看前 N−1 个词，通过计数估计条件概率。未经平滑的最大似然估计会给未见组合零概率，使测试文本整体概率为零、perplexity 无穷。加法平滑、插值/回退和 Kneser-Ney 等方法为未见事件分配质量，同时调整已见事件。

神经语言模型用共享向量和网络预测分布，能在不同上下文间泛化；RNN 顺序更新状态，Transformer 按 mask 聚合上下文，仍须受训练数据和上下文窗口限制。N-gram 通常便宜、可解释且适合作基线，但稀疏性和长依赖弱；神经模型规模大不保证特定分布上更可靠。比较 perplexity 要固定 tokenizer、测试集、上下文和计分 token，否则其数值不具可比性。

NLP是处理人类语言的研究领域，LLM是其中一类语言建模技术，不能把二者视为互斥类别。传统方案常为分类、抽取等任务训练专门模型；LLM通过通用预训练、指令微调和上下文示例支持多任务，但小型专用模型也可能在固定任务的成本与精度上更合适。LLM没有统一的百亿参数起点，参数规模本身也不证明具备推理或指令能力。

$$
\begin{aligned}p(x_{1:T})&=\prod_{t=1}^T p(x_t\mid x_{<t})\\p_{N\text{-gram}}(x_t\mid x_{<t})&\approx\frac{C(x_{t-N+1:t})}{C(x_{t-N+1:t-1})}\\p_{\rm add\text{-}\alpha}(w\mid h)&=\frac{C(h,w)+\alpha}{C(h)+\alpha |V|}\end{aligned}
$$

#### 易错点

- 把概率链式法则和 N-gram 截断近似当作同一假设。
- 为平滑分配额外质量却不重归一化。

#### 追问

- 为什么 Kneser-Ney 更关注不同上下文中的出现次数？
- N-gram 和神经模型的 OOV 策略如何影响 PPL？

<a id="bas-015"></a>
### BAS-015 · TF-IDF 与 TextRank 如何提取关键词，各有哪些局限？

**L2**

#### 答案

TF-IDF给词赋予文档内频率与逆文档频率的乘积：当前文档频繁、在全语料较少见的词更显著。tf可以原始计数、归一化或log变换，idf常加平滑；实现需固定分词、停用词和语料范围，不能把高分直接理解为事实重要性。TextRank把词的局部共现构成图，通过PageRank式迭代累积邻接节点的重要性，不要求已标注关键词；也可用句子相似图提取摘要。

TF-IDF依赖背景语料统计，对同义词和词序不敏感；TextRank依赖窗口和图边，可能偏向连接度高的常见词，也不保证覆盖语义关键但罕见的实体。jieba提供这两类工具，添加领域词典可影响候选，但不能单靠工具名保证质量。评估可用人工关键词覆盖、冗余率及检索/分类的下游收益，对单篇短文本尤其要检查分词和停用词规则。

$$
\begin{aligned}\operatorname{TFIDF}(w,d)&=\operatorname{tf}(w,d)\log\frac{N+1}{\operatorname{df}(w)+1}\\S(v)&=(1-a)+a\sum_{u\to v}\frac{w_{uv}}{\sum_k w_{uk}}S(u)\end{aligned}
$$

#### 易错点

- 把tf-idf或中心性分数当作语义正确性的概率。
- 用测试集拟合idf却称没有数据泄露。

#### 追问

- 共现窗口变大如何改变TextRank的边与得分？
- 为什么关键词提取与LLM tokenizer分词目标不同？

<a id="pre-002"></a>
### PRE-002 · BPE、WordPiece、Unigram 与 SentencePiece 分别是什么？

**L1**

#### 答案

BPE 从较小单元出发，逐步合并高频相邻对，学习 merge 规则并按规则编码；子词表示能减少未知词问题。Unigram 为候选切分建立概率模型，逐步删减候选词表，用动态规划找高概率切分，也可采样切分。

SentencePiece 是支持 BPE、Unigram 等模型的训练工具，不是与它们并列的单一算法。它可直接从原始文本训练，处理空白与归一化，减少对语言特定预分词的依赖。字符、byte 和 Unicode 归一化设定都会影响可逆性、序列长度及代码字符串表现。

WordPiece 常按词表做最长前缀匹配，再对后续子词使用 continuation 标记；它通常保存最终词表，不按BPE的merge rank回放合并。训练目标/打分与实现并不唯一，Hugging Face课程的词对频率比是教学近似，不能当作Google未公开训练器的唯一公式。Byte-level BPE以字节为基本覆盖单元，可减轻未知字符问题，但normalization、预分词和special tokens仍影响精确可逆性。选择tokenizer要比较语言覆盖、序列长度、代码/空白保持与成本，不能先跑jieba替换原tokenizer而期待checkpoint仍兼容。

#### 易错点

- 把 SentencePiece 当作独立于 BPE、Unigram 的分词算法。
- 认为 token 总是一个汉字或一个英文单词。

#### 追问

- byte-level tokenizer 为什么通常能覆盖未知字符？
- 分词可逆性与文本归一化有什么冲突？

<a id="pre-003"></a>
### PRE-003 · 词表越大越好吗，扩词表有哪些代价？

**L2**

#### 答案

更大的词表可能降低同一文本的 token 数，改善部分语言的压缩率，但也增大 embedding、LM head 和 logits 计算。若输入/输出不共享权重，两套矩阵约有 $2Vd$ 参数，共享后约为 $Vd$，bias 另算；稀有或新增 token 出现太少时也难以学到稳定表示。

选词表应同时评估覆盖、压缩率、训练充分度和显存，在多语言、代码、数字与罕见字符上统计 tokens/byte 或 tokens/字符。继续训练时改变 tokenizer 还会改变序列分布，必须检查旧 token ID、初始化与新增行的训练，不能只看总 token 数减少。

#### 易错点

- 把 tokenizer 扩词表当作无需训练的免费优化。
- 直接把更少 token 等同于更好的语义理解。

#### 追问

- 什么时候值得为中文新增 token？
- 如何避免新增 token 的 embedding 未被 PEFT 更新？

<a id="topic-2"></a>
## 序列任务与经典 NLP

<a id="bas-002"></a>
### BAS-002 · POS、NER 如何做序列标注？HMM、独立分类与 CRF 有什么区别？

**L2**

#### 答案

词性标注为每个词分配语法类别，NER 通常用 BIO/BIOES 描述实体类别和边界。HMM 建模标签转移与观测发射的联合概率，利用生成式条件独立假设；神经编码器加逐位置 softmax 直接学习标签概率，但独立 argmax 可能得到非法 BIO 转移。线性链 CRF 在整条标签序列上归一化，把发射分数与相邻标签转移分数共同训练，能显式表达转移偏好。

训练用真实序列分数减 log-partition，分母通过前向动态规划的 logsumexp 计算；预测用 Viterbi，把求和换成 max 并保存回溯指针，复杂度通常为 O(TK²)。CRF 不会自动理解实体语义，质量仍依赖编码器和标注。需要正确 mask padding、约束非法转移、区分 token accuracy 与实体级 precision/recall/F1；嵌套实体未必能用单条 BIO 标签表达。

$$
\begin{aligned}s(x,y)&=\sum_t e_t(y_t)+\sum_t A_{y_{t-1},y_t}\\p(y\mid x)&=\frac{e^{s(x,y)}}{\sum_{y'}e^{s(x,y')}}\\\mathcal L&=-s(x,y)+\log\sum_{y'}e^{s(x,y')}\end{aligned}
$$

#### 易错点

- 把 CRF 的训练归一化当作逐 token softmax。
- 只报 token 准确率而忽略实体边界错误。

#### 追问

- CRF 前向算法与 Viterbi 的递推有什么不同？
- 子词 tokenizer 后如何对齐词级标签？

<a id="bas-003"></a>
### BAS-003 · 成分句法与依存句法分别预测什么，怎样评价解析结果？

**L2**

#### 答案

成分句法把连续片段组织为短语树，例如名词短语、动词短语，常以 CFG 或概率 CFG 表示；依存句法在词之间建立有向中心词关系，例如主语、宾语。两者描述同一句子的不同结构，不能把依存边直接视为短语标签。

成分解析可通过 chart/CKY 动态规划组合跨度，标准二元 CFG 的 CKY 复杂度约 O(n³|G|)，神经 span parser 为跨度打分后施加树约束。依存解析常见转移式解析或图式解析，后者为 head-dependent 边打分，再求满足单一中心词、连通与无环约束的树。成分结果常测标注跨度 F1；依存结果测 UAS（中心词正确）与 LAS（中心词和关系都正确），须说明标点与分词口径。解析能提供结构信号，但通用 LLM 不必先跑显式句法分析才能理解语言。

$$
\operatorname{UAS}=\frac{\#\{i:\hat h_i=h_i\}}{N},\qquad\operatorname{LAS}=\frac{\#\{i:\hat h_i=h_i,\hat r_i=r_i\}}{N}
$$

#### 易错点

- 混淆 constituency 的跨度标签与 dependency 的中心词边。
- 把不同分词口径的 UAS/LAS 直接比较。

#### 追问

- 图式解析为何不能对每个词独立选最优 head 后结束？
- 非投射依存树和投射树有什么差别？

<a id="bas-008"></a>
### BAS-008 · CNN、RNN 与 Transformer 在序列建模、长依赖和并行性上怎样比较？

**L1**

#### 答案

CNN 用共享局部卷积核提取邻近片段模式，堆叠、空洞卷积或较大卷积核可扩大感受野；RNN 把上一时刻状态传给下一时刻，天然支持在线顺序处理；Transformer 的注意力直接连接可见位置，再以 FFN 做逐位置变换。三者都能共享参数，差别在跨位置的信息传递规则。

RNN 的训练在时间轴有顺序依赖，长路径容易遇到梯度问题；CNN 的独立位置可并行，但远距离需要更多层或设计；全注意力的远距离路径较短，训练可在位置轴并行，代价是标准注意力的二次交互计算。Decoder Transformer 的 teacher forcing 训练可并行，普通自回归生成仍依赖前一个输出；这与层间串行是不同维度。不存在脱离数据、长度、硬件的固定效果排名，流式和边缘任务中 RNN/CNN 仍有合理应用。

$$
\begin{aligned}h_t^{\rm RNN}&=\phi(W_xx_t+W_hh_{t-1}+b)\\h_t^{\rm CNN}&=\phi\!\left(\sum_{j=-k}^{k}W_jx_{t+j}+b\right)\\H^{\rm attn}&=\operatorname{softmax}(QK^\top/\sqrt{d_k}+M)V\end{aligned}
$$

#### 易错点

- 把 Transformer 的训练并行推广为所有生成步骤并行。
- 以某次机器翻译结果宣称 CNN/RNN/Transformer 的效果排名永远固定。

#### 追问

- 空洞卷积怎样扩大感受野？
- causal RNN 与 bidirectional RNN 的在线使用条件有何区别？

<a id="bas-009"></a>
### BAS-009 · LSTM 与 GRU 怎样缓解普通 RNN 的梯度问题，有何结构区别？

**L2**

#### 答案

普通 RNN 的反向梯度跨时间连续乘 Jacobian，谱尺度持续小于或大于 1 时容易消失或爆炸。LSTM 维护 cell state，用遗忘门保留旧状态、输入门写入候选、输出门控制暴露的 hidden state；沿 cell 的直接路径是加法更新，能减轻长依赖的优化困难。GRU 通常只有一个状态，更新门在旧状态与候选之间插值，重置门调节生成候选时使用多少旧状态，参数通常较少。

门控并不保证任意长度无损记忆，门接近关闭时仍会阻断梯度，梯度裁剪也只控制爆炸。GRU 的更新门定义和重置门应用位置在不同文献/框架里可能不同，手写前先固定约定。LSTM/GRU 都在时间轴递归，双向版本能利用左右上下文但不能直接用于未知未来的在线预测。选择应根据序列长度、流式要求、吞吐和任务验证。

$$
\begin{aligned}c_t&=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\quad h_t=o_t\odot\tanh(c_t)\\\tilde h_t^{\rm GRU}&=\tanh(W_xx_t+W_h(r_t\odot h_{t-1})+b)\\h_t^{\rm GRU}&=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t\end{aligned}
$$

#### 易错点

- 说门控能彻底消除所有梯度消失。
- 忘记不同 GRU 实现可能把 z 与 1−z 的角色互换。

#### 追问

- LSTM 的 cell state 和 hidden state 为什么分开？
- 截断 BPTT 会牺牲什么梯度信息？

<a id="topic-3"></a>
## 数学、概率与统计

<a id="tfm-013"></a>
### TFM-013 · 交叉熵、KL 散度与 perplexity 的关系和实现注意点是什么？

**L1**

#### 答案

离散分布满足 $H(p,q)=H(p)+D_{\rm KL}(p\|q)$，其中 $H(p)=-\sum_i p_i\log p_i$、$H(p,q)=-\sum_i p_i\log q_i$、$D_{\rm KL}(p\|q)=\sum_i p_i\log(p_i/q_i)$。展开对数即可证明。两者须定义在同一支持空间并归一化；$0\log0$ 按极限为 0，若 $p_i>0,q_i=0$，交叉熵与 KL 为 $+\infty$。

KL 非负，$p=q$ 时取零，通常不对称，也不是距离度量。固定目标 $p$ 时，最小化 CE 与最小化该方向的 KL 等价；若目标也含待训练参数且未 detach，$H(p)$ 与 $p$ 的梯度不能忽略。蒸馏还须明确 teacher/student 的方向以及是否停止 teacher 梯度。

One-hot 监督样本的目标熵为零，单样本 CE 与 KL 都为 $-\log q(y)$，但真实数据的条件熵不一定为零。软标签与标签平滑的目标熵通常大于零；固定归一化 $p$、$q=\operatorname{softmax}(z)$ 时，对 logits 的梯度为 $q-p$，对概率 $q_i$ 本身为 $-p_i/q_i$。

`CrossEntropyLoss` 通常接收原始 logits，内部做稳定的 log-softmax/NLL；`KLDivLoss` 通常接收 `input=log q,target=p`，接口顺序与数学记号不同，reduction 也须按分布单位解释。自然对数下，PPL 是有效 token 平均 NLL 的指数；不同长度 batch 要按有效 token 加权，比较时统一 tokenizer、数据与窗口。低 PPL 不保证指令遵循、事实性或偏好更好。

$$
\begin{aligned}H(p,q)&=H(p)+D_{\rm KL}(p\|q)\\\nabla_\theta H(p,q_\theta)&=\nabla_\theta D_{\rm KL}(p\|q_\theta)\quad(p\ \text{fixed})\\\operatorname{PPL}&=\exp\!\left(-\frac{1}{N_{\rm valid}}\sum_{t\in\mathcal T_{\rm valid}}\log p_\theta(x_t\mid x_{<t})\right)\end{aligned}
$$

#### 易错点

- 无条件声称 CE 与 KL 完全相同，遗漏目标熵及 fixed-p 前提。
- 把 `KLDivLoss` 的输入顺序/mean 归一化当成数学 KL 的默认定义。

#### 追问

- 如何从 softmax 推导对 logits 的梯度 $q-p$？
- 可训练的软目标或双向 KL 会怎样改变优化目标？

<a id="tfm-017"></a>
### TFM-017 · 熵衡量什么？离散熵、条件熵与模型输出熵怎样区分？

**L1**

#### 答案

离散熵是自信息的期望：事件自信息为 $I(x)=-\log p(x)$，熵为 $H(X)=\mathbb E[I(X)]$。用 $\log_2$ 时单位是 bit，自然对数时是 nat，零概率项按 $0\log0=0$ 的极限处理。固定 $V$ 类时，$0\le H\le\log V$，确定分布取零，均匀分布达到上界；二元熵 $-a\log a-(1-a)\log(1-a)$ 在 $a=1/2$ 最大。熵由完整分布决定，不能只凭最大 token 概率比较。

条件熵为 $H(X\mid Y)=\sum_y p(y)H(p(X\mid y))$，满足链式法则 $H(X,Y)=H(Y)+H(X\mid Y)$。条件信息在平均意义上降低不确定性，但某个特定上下文的条件分布不一定比无条件分布熵更低；序列熵可按此链式展开。

真实数据熵 $H(p)$、模型输出熵 $H(q_\theta)$ 与训练交叉熵 $H(p,q_\theta)$ 要区分。模型可能对错误答案非常自信，交叉熵还含分布失配的 KL，所以降低输出熵不保证拟合正确。日志应明确过滤前后、mask、token 分布与上下文平均范围。连续变量的微分熵可以为负并随尺度变化，不能照搬离散熵的界。

$$
\begin{aligned}H(p)&=-\sum_i p_i\log p_i\\H(X\mid Y)&=\sum_y p(y)H(p(X\mid Y=y))\\H(X,Y)&=H(Y)+H(X\mid Y)\end{aligned}
$$

#### 易错点

- 低熵就是准确，高熵就是幻觉，或把数据熵与模型预测熵混同。
- 把平均条件熵下降说成所有特定条件都降低不确定性。

#### 追问

- 高置信度错误输出为何可以有低熵？
- 为什么连续均匀分布的微分熵会随区间长度变成负值？

<a id="tfm-018"></a>
### TFM-018 · 矩阵的秩与特征值如何计算，和奇异值有什么关系？

**L2**

#### 答案

矩阵的秩是行空间或列空间的维数，可由消元主元数或非零奇异值数求得。对 $A\in\mathbb R^{m\times n}$，有 $\operatorname{rank}(A)+\dim\ker A=n$。LoRA 的低秩分解满足 $\operatorname{rank}(BA)\le\min(\operatorname{rank}A,\operatorname{rank}B)\le r$，不要求底座 $W_0$ 低秩。

方阵特征值可以为复数，满足 $Av=\lambda v,v\ne0$，手算可先解 $\det(\lambda I-A)=0$，再解特征向量。例 $A=\begin{pmatrix}2&1\\0&3\end{pmatrix}$ 的特征值为 2、3，对应向量可取 $(1,0)^\top$、$(1,1)^\top$，秩为 2。一般矩形矩阵没有这种标准特征值定义；$A^\top A$ 的谱用于求奇异值，不能称为矩形 $A$ 自身的特征值。

大矩阵通常不显式展开特征多项式：数值库可经 Hessenberg 化和 QR 等过程求 Schur 分解，复 Schur 对角给特征值，实 Schur 的 $2\times2$ 块需另解。对称/Hermitian 矩阵可选专用求解器。

若 $A=P\Lambda P^{-1}$ 可对角化，秩等于按重数计的非零特征值个数；一般方阵不能这样数。例如 $\begin{pmatrix}0&1\\0&0\end{pmatrix}$ 的特征值全零，秩却为 1，且不可对角化。任意方阵仍满足行列式非零、全部特征值非零与满秩的等价关系。

将奇异值和 $A^\top A$ 的特征值分别按降序排列，令 $p=\min(m,n)$。SVD 写为 $A=U\Sigma V^\top$，非零奇异值的平方构成 $A^\top A$ 的非零谱，其余位置补零；实对称矩阵的奇异值为 $|\lambda_i|$。浮点数值秩按 $\sigma_i>\max(\mathrm{atol},\mathrm{rtol}\,\sigma_{\max})$ 判断，阈值取决于 dtype、规模与噪声，近零不能等同于精确零。

截断SVD保留最大r个奇异值，得到秩至多r的近似A_r=U_rΣ_rV_rᵀ；在谱范数或Frobenius范数下它是最佳秩r近似，误差分别由第r+1个奇异值或尾部奇异值平方和确定。它压缩已经存在的矩阵，而LoRA训练一个低秩增量，二者目标不同；对权重做低秩分解的最优矩阵误差也不保证下游任务损失最小。

$$
\begin{aligned}\det(\lambda I-A)&=0,\quad(A-\lambda I)v=0,\quad v\ne0\\\operatorname{rank}(A)&=\#\{i:\sigma_i(A)>0\}\\\lambda_i(A^\top A)&=\sigma_i(A)^2,\quad1\le i\le p=\min(m,n)\\\lambda_i(A^\top A)&=0,\quad p<i\le n\\A=P\Lambda P^{-1}&\implies\operatorname{rank}(A)=\#\{i:\lambda_i\ne0\}\\\operatorname{rank}(BA)&\le r\end{aligned}
$$

#### 易错点

- 所有矩阵的秩都等于非零特征值个数，或给非方阵直接定义标准特征值。
- 把奇异值与一般特征值逐项等同，或把训练配置 r 当作已学更新的实际秩。

#### 追问

- 为什么 $A^\top A$ 理论上可求奇异值，但数值计算可能放大条件数问题？
- 低秩近似、数值秩与 LoRA 的秩上界分别回答什么问题？

<a id="topic-4"></a>
## 深度学习与优化基础

<a id="bas-010"></a>
### BAS-010 · Sigmoid、Tanh、ReLU、GELU 与 SiLU 的公式和梯度特点是什么？

**L1**

#### 答案

没有非线性时，多层线性映射仍可合成一个线性映射。Sigmoid 将值压到 (0,1)，适合二元概率或门；Tanh 输出在 (−1,1)，但两者在绝对值大时都易饱和。ReLU 对正值梯度为 1、负值为 0，计算简单，但神经元可能长期落在负区间；Leaky ReLU/PReLU 用固定或可学习的小负侧斜率减轻这一问题。

GELU 为 xΦ(x)，平滑地按标准正态 CDF 调节输入；SiLU 为 xσ(x)，Swish 常写 xσ(βx)，β=1 时就是 SiLU。它们保留部分负输入，常用于 Transformer FFN；GLU/SwiGLU 是两个投影分支相乘的门控结构，不是仅替换一个标量激活。激活选择要和初始化、归一化、精度与 FFN 中间维度一起比较，不能因负值可通过就推断一定收敛更快。

ELU 在正侧为 x、负侧为 α(exp(x)−1)，负侧会饱和，α=1 时零点处一阶导数连续。RReLU 则在训练时随机取负侧斜率、推理时常用其期望值；PReLU 的斜率是学习得到的参数。三者的参数和随机性不同，早期 CNN 实验的优劣不能直接推广为所有 LLM 的结论。

$$
\begin{aligned}\sigma(x)&=(1+e^{-x})^{-1},\quad\sigma'(x)=\sigma(x)(1-\sigma(x))\\\tanh'(x)&=1-\tanh^2(x)\\\operatorname{ReLU}(x)&=\max(0,x)\\\operatorname{GELU}(x)&=x\Phi(x)\\\operatorname{SiLU}(x)&=x\sigma(x),\quad\operatorname{SiLU}'(x)=\sigma(x)+x\sigma(x)(1-\sigma(x))\end{aligned}
$$

#### 易错点

- 把 SwiGLU 当作只含一个投影的 SiLU 激活。
- 把 ReLU 在 0 点的实现导数约定说成唯一数学导数。

#### 追问

- GELU 的近似式为什么与精确式略有差异？
- ReLU 的死亡问题与梯度爆炸分别如何识别？

<a id="bas-011"></a>
### BAS-011 · SGD、Momentum、AdaGrad、RMSProp 与 Adam 的更新规则怎样理解？

**L2**

#### 答案

Mini-batch SGD 用当前批次的梯度更新参数，动量用过去梯度的指数滑动累积减少振荡、维持方向。AdaGrad 累积平方梯度，为频繁更新坐标降低有效学习率，适合某些稀疏问题但学习率可能持续衰减；RMSProp 用平方梯度的指数平均缓解无界累积。Adam 同时维护一阶与二阶矩估计，并在零初始化时做偏差校正。

Adam 中二阶矩是未中心化平方梯度的平均，不等同于梯度方差；除以其平方根也不意味着把所有坐标真实曲率准确求出。实现要区分 optimizer step 与 microbatch、明确 epsilon 放置和权重衰减；AdamW 的解耦衰减另见预训练题。Nesterov 动量在前瞻位置估计方向，AdaDelta 利用更新量与梯度量的滑动统计调节尺度，这些方法没有统一适用所有任务的排名。

$$
\begin{aligned}m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t\\v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2\\\hat m_t&=m_t/(1-\beta_1^t),\quad\hat v_t=v_t/(1-\beta_2^t)\\\theta_{t+1}&=\theta_t-\eta_t\hat m_t/(\sqrt{\hat v_t}+\epsilon)\end{aligned}
$$

#### 易错点

- 把二阶矩 v 当作已减均值的方差。
- 每个 microbatch 都更新 Adam 状态却称它等价梯度累积。

#### 追问

- 为什么初始化的动量需要偏差校正？
- AdaGrad 的有效学习率为什么通常随训练下降？

<a id="bas-012"></a>
### BAS-012 · 反向传播怎样使用链式法则？线性层、MSE 与 Softmax 交叉熵如何求梯度？

**L1**

#### 答案

反向传播按计算图的逆拓扑顺序把上游梯度乘局部 Jacobian，并累加所有分支对同一变量的贡献。它不是数值差分，也不是每次显式构造整个 Jacobian；自动微分通常计算 vector-Jacobian product。线性层 Y=XW+b 的梯度为 XᵀG、GWᵀ，bias 梯度沿样本维求和；共享参数必须累加每次使用的贡献。

MSE 用预测误差驱动回归，softmax 加交叉熵的 logits 梯度则是 p−y（在规范标签和相同归一化条件下）。mean reduction 会引入对应样本或 token 数的除数，loss mask 使未监督项梯度为零，但这些输入仍可能通过注意力影响被监督输出。定位梯度错误可用小尺寸有限差分核对、检查 detach/in-place 修改和梯度清零；有限差分只作验证，步长过小会受浮点消减影响。

$$
\begin{aligned}Y&=XW+b,\quad G=\partial\mathcal L/\partial Y\\\nabla_W\mathcal L&=X^\top G,\quad\nabla_X\mathcal L=GW^\top,\quad\nabla_b\mathcal L=\sum_iG_i\\\mathcal L_{\rm MSE}&=\frac1n\sum_i(\hat y_i-y_i)^2\\\frac{\partial\mathcal L_{\rm CE}}{\partial z_i}&=p_i-y_i\quad\text{(single example)}\end{aligned}
$$

#### 易错点

- 共享参数仅保留最后一次调用的梯度。
- 混淆 sum、按样本 mean 和按有效 token mean 的尺度。

#### 追问

- 为什么 logsoftmax 与 NLL 合并后的梯度比单看 softmax Jacobian 更直接？
- 梯度检查应该如何选择有限差分步长？

<a id="bas-013"></a>
### BAS-013 · 过拟合、欠拟合与数据泄露怎样区分，L1/L2、早停和 Dropout 分别做什么？

**L2**

#### 答案

训练与验证都差可能是欠拟合、目标不合适或优化失败；训练好而独立验证差常提示过拟合或分布差异，不能只由两条 loss 曲线确定原因。数据泄露会让验证异常乐观，例如相同文档的相邻块落到训练和验证、预处理在全数据拟合，或反复根据测试集调参。应按文档、用户、时间或实体分组切分，保留真正未参与选择的测试集。

L2 惩罚连续压小参数，L1 鼓励部分坐标变零；在自适应优化器里显式 L2 与解耦 weight decay 不等价。早停依据验证趋势选择 checkpoint，dropout 在训练中随机屏蔽与缩放激活以正则化；它们都无法替代高质量数据或消除分布漂移。实际排查先核对数据切分和评测实现，再比较训练曲线、数据量消融、容量与正则强度，不通过单纯增加训练 epoch“解决”泄露。

$$
\mathcal L_{L_1}=\mathcal L_{\rm data}+\lambda\lVert\theta\rVert_1,\qquad\mathcal L_{L_2}=\mathcal L_{\rm data}+\frac\lambda2\lVert\theta\rVert_2^2
$$

#### 易错点

- 把验证集反复用于决策后仍视为最终无偏测试。
- 只凭 train/val gap 断言问题必然是参数太多。

#### 追问

- 为什么同文档的 chunk 应避免跨训练/验证集合？
- L1 的零点处应该使用什么梯度概念？

<a id="bas-014"></a>
### BAS-014 · Xavier、He 初始化与 embedding 乘 √d_model 的目的是什么？

**L2**

#### 答案

初始化要控制深层网络前向激活和反向梯度的尺度。对近似独立零均值输入，线性层的输出方差与 fan-in、权重方差相乘；Xavier 综合 fan-in/fan-out，适用于近似线性或对称激活的分析；He 对 ReLU 丢弃约一半输入的情形使用约 2/fan-in 的权重方差。假设会受相关性、残差、归一化和门控影响，所以这是设计起点而非任意网络的稳定性证明。

原始 Transformer 将输入 embedding 乘 √d_model 后加位置编码，用来调节词表示相对于位置表示的尺度，并配合其共享 embedding 的参数化。现代模型可能直接使用未缩放 embedding，具体要看初始化和前向代码。输出 logits 是否缩放、Norm 放置和 residual 缩放是不同机制；不能见到“√d”就一概归因于注意力的缩放点积。初始化后可测各层激活 RMS、logits 和梯度范数，验证是否稳定。

$$
\begin{aligned}\operatorname{Var}(Wx)_i&\approx\operatorname{fan\_in}\operatorname{Var}(W_{ij})\operatorname{Var}(x_j)\\\operatorname{Var}(W)_{\rm Xavier}&\approx\frac{2}{\operatorname{fan\_in}+\operatorname{fan\_out}}\\\operatorname{Var}(W)_{\rm He}&\approx\frac{2}{\operatorname{fan\_in}}\end{aligned}
$$

#### 易错点

- 把 embedding 的 √d_model 缩放说成所有 Transformer 的必需组件。
- 把初始化方差的独立性分析当作训练全过程成立。

#### 追问

- 残差分支叠加为什么可能需要额外的深度缩放？
- 均匀分布和正态分布怎样匹配同一目标方差？

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
