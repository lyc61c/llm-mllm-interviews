# RAG、检索、重排与图检索

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 目录

- [文档切分与索引](#topic-1)
  - [RAG-001 · 一个可落地的 RAG 系统有哪些环节？](#rag-001)
  - [RAG-002 · RAG、微调与 Prompt/CoT 如何选择？VQA 中有何取舍？](#rag-002)
  - [RAG-003 · chunk 大小、overlap 与父子、链式、树式索引怎样设计？](#rag-003)
  - [RAG-008 · HNSW 的 M、ef_construction 和 ef_search 怎样权衡？](#rag-008)
  - [RAG-017 · IVF_FLAT、IVF_PQ 与 HNSW 如何检索，nlist/nprobe 和 PQ 有何权衡？](#rag-017)
- [召回、融合与重排](#topic-2)
  - [RAG-004 · embedding 模型与相似度应如何选？](#rag-004)
  - [RAG-005 · BM25 与向量检索各有什么优势？](#rag-005)
  - [RAG-006 · 混合检索的分数融合与 RRF 有什么区别？](#rag-006)
  - [RAG-007 · reranker 和 embedding 检索模型如何分工？](#rag-007)
  - [RAG-019 · 怎样训练 embedding/retriever，hard negatives、ICT、SEED 与 REALM 分别解决什么？](#rag-019)
- [查询优化与图检索](#topic-3)
  - [RAG-009 · query rewrite、multi-query 和 HyDE 何时有用？](#rag-009)
  - [RAG-010 · Self-RAG 和多跳检索怎样改善复杂问答？](#rag-010)
  - [RAG-016 · 知识图谱 RAG 与 GraphRAG 何时有用，和多 Agent 有什么关系？](#rag-016)
- [评测、权限与多模态 RAG](#topic-4)
  - [RAG-011 · RAG 如何建立分层评测并定位 bad case？](#rag-011)
  - [RAG-012 · RAG 为什么仍会幻觉，怎样设计引用与拒答？](#rag-012)
  - [RAG-013 · 企业 RAG 如何处理权限、更新与删除？](#rag-013)
  - [RAG-014 · 长上下文能否替代 RAG？](#rag-014)
  - [RAG-015 · 多模态 RAG 怎样检索图表、扫描 PDF 与视频？](#rag-015)
  - [RAG-018 · RAGFlow、Haystack、LlamaIndex 与 DSPy 怎样分工和选型？](#rag-018)

<a id="topic-1"></a>
## 文档切分与索引

<a id="rag-001"></a>
### RAG-001 · 一个可落地的 RAG 系统有哪些环节？

**L1**

#### 答案

RAG 的离线链路包括文档解析、清洗、切分和建立带元数据的索引；在线链路包括问题转换、召回、重排、证据上下文构造和生成。索引应保存文档 ID、页码、版本、权限与片段边界，以支持更新和可追溯引用。

生成输入需明确区分任务指令与外部资料，资料不足时允许澄清或拒答。检索、证据使用和生成应分别评测，因为端到端答错可能来自不同环节。现代 prompt RAG 可以不训练模型，原始 RAG 论文也研究了检索与生成的联合微调，因此训练与否不是定义边界。

搜索系统通常返回排序文档或片段，RAG在检索结果上生成综合答案；检索相关性和最终答案质量需分别评价。生成阶段不能把无证据的回答包装成来自搜索结果，也不能用答案正确掩盖来源错误。

#### 易错点

- 不要把 RAG 定义为必须零训练，也不要把它等同向量数据库。

#### 追问

- 怎样定位是没检索到还是模型没用好证据？

<a id="rag-002"></a>
### RAG-002 · RAG、微调与 Prompt/CoT 如何选择？VQA 中有何取舍？

**L1** · 字节跳动

#### 答案

Prompt engineering 改变当次任务约束，RAG 在推理时提供外部证据，微调则改变模型的行为分布。知识更新频繁、需要权限控制或溯源时，可优先评估 RAG；稳定输出格式或专门能力可考虑微调。微调不能充当可靠数据库，事实未必能精确写入、更新或撤回。

三者可以组合，例如训练模型更好地读取证据或改善检索能力，但 RAG 仍依赖解析、召回与证据质量。应使用代表性业务样本，在同等预算下对 prompt、RAG、SFT 及其组合做消融，比较效果、时延、数据维护与成本。

先按错误类型选择：看不清文字、漏检物体或混淆空间关系是感知问题，优先检查分辨率、裁剪、视觉编码与证据输入；缺少外部知识或需要更新事实时评估 RAG；已有足够事实但步骤组合失败时可以尝试分解问题、CoT 提示或受约束的多步推理。推理文本看似合理不代表视觉证据被正确使用。

CoT主要改变推理时的生成过程，部署快但增加 token 和时延，也可能放大初始感知错误。RAG 引入可追溯的外部证据，需要对图像对象/OCR、问题与文档进行检索和重排；召回错误与图文错配会把错误证据带给模型。SFT用稳定、可信的图文示范改变模型行为，适合反复出现的域内任务和输出格式，但需要标注、数据清洗和域外回归，也会有遗忘风险。

三者可组合：检索知识后生成答案，或用SFT教模型读取图文证据和适时调用检索。资源充足仍不意味着自动选SFT；数据少、业务事实频繁变化、根因是输入缺失时，继续训练不一定解决问题。用同一测试集与预算比较直接回答、CoT、RAG、SFT及组合，并分开报告感知、知识、推理错误、视觉幻觉、准确率和延迟。训练样本及评测按图片/文档来源分组去重，避免同图不同问法泄露。

#### 易错点

- “RAG 只学知识、微调只学能力”是粗略划分，不是严格理论。
- CoT、RAG、SFT是可组合的方法，不是互斥的模型架构。
- 资源足够不能替代高质量监督、可靠证据和受控消融。

#### 追问

- 如何设计遮图、换图和人工OCR输入的消融？
- 知识每天变化，但答案格式很稳定，该怎样组合检索与微调？

<a id="rag-003"></a>
### RAG-003 · chunk 大小、overlap 与父子、链式、树式索引怎样设计？

**L2**

#### 答案

Chunk 应覆盖可回答的信息单元，并适配 embedding 模型和生成器的预算。优先按标题、段落、表格等结构切分，再比较多个大小与 overlap：重叠能保留边界上下文，也会增加存储和重复召回，没有固定最优参数。

表格应保留表头、单位与关联行，代码和公式尽量避免在语义中间断开。小片段检索、返回较大父段是一种兼顾定位和上下文完整性的方式。评测需联合观察证据召回、答案质量、重复率及 token 成本；字符数不能直接当作 token 数，中文与代码尤其需要实际统计。

切块后还可保留节点的顺序与层级：链式/列表索引按文档顺序组织节点，适合遍历汇总或命中后扩展相邻上下文；LlamaIndex 的 Summary Index（原 List Index）默认取全部节点，并非默认只取向量 top-k。树式索引把原始块作为叶节点、子节点摘要作为父节点，可从摘要逐层选择分支定位证据；摘要可能丢失细节，回答仍应保留到原文的定位。顺序、父子关系与向量索引可以组合，例如先向量召回小块，再补邻居或父段；要共同评估召回、摘要误差与上下文成本。

#### 易错点

- 切分字符数不能自动等价为 token 数，中文与代码差异明显。

#### 追问

- 大 chunk 提高答案质量但降低召回时，你怎样定位原因？

<a id="rag-008"></a>
### RAG-008 · HNSW 的 M、ef_construction 和 ef_search 怎样权衡？

**L2**

#### 答案

HNSW 使用分层近邻图减少搜索量，以近似召回换取速度。`M` 影响图连接与内存，`ef_construction` 影响建图质量，`ef_search` 或实现中的查询 `ef` 影响搜索宽度与延迟。增大连接数或构建搜索宽度通常提高图质量，也增加内存或建库时间；增大查询宽度通常提高 ANN recall，同时增加耗时，且需满足实现对 `k` 的约束。

调参时先与暴力近邻结果比较索引 recall，再单独测文档的业务相关性，二者不能混为一谈。过滤条件、删除更新、距离函数和索引版本也会影响结果，需纳入验证。

#### 易错点

- ANN recall 高不代表找到业务所需证据；两种召回要分开。

#### 追问

- 精确搜索命中、ANN 未命中时你会调什么？

<a id="rag-017"></a>
### RAG-017 · IVF_FLAT、IVF_PQ 与 HNSW 如何检索，nlist/nprobe 和 PQ 有何权衡？

**L2**

#### 答案

IVF用粗量化器把向量划到nlist个簇，查询选择nprobe个簇，在对应倒排列表里寻找近邻。IVF_FLAT保存原始向量并在选中列表算精确距离，仍可能因为漏查其他簇错过全局近邻，所以不是整个索引的精确搜索。

PQ将向量拆成M个子空间，各自用有限码本编码，缩减存储并用距离查表近似计算；IVF_PQ常编码相对粗中心的残差。若每子空间用b位，编码约Mb/8字节，需另加ID、码本与索引开销。它同时存在粗筛漏召回和量化排序误差，可增大nprobe或使用原向量rerank，但成本会上升。

HNSW通过分层近邻图导航，通常无需先训练聚类码本，但图结构增加内存。选择取决于数据量、内存、召回/延迟目标、增删维护和是否有代表性训练数据。IVF的nlist过大可能使列表过小/训练不足，nprobe越大通常更准也更慢；应在相同业务集合上测实际曲线。

```math
M_{\mathrm{PQ\ codes}}\approx N\frac{Mb}{8}\ \mathrm{bytes},\qquad \mathrm{probed\ fraction}\approx\frac{n_{\mathrm{probe}}}{n_{\mathrm{list}}}
```

#### 易错点

- 列表大小常不均匀，nprobe/nlist只表示粗比例，不是严格耗时或召回公式。

#### 追问

- ANN召回差怎样区分粗量化漏簇、PQ误差与embedding本身不合适？

<a id="topic-2"></a>
## 召回、融合与重排

<a id="rag-004"></a>
### RAG-004 · embedding 模型与相似度应如何选？

**L2**

#### 答案

双塔分别编码 query 和文档，便于预计算文档表示与快速检索。模型选型应匹配语言、领域、文档长度以及 query/document 指令模板，并用业务检索标注集验证；专有实体、代码、跨语言和长段落的表现可能不同，通用榜单不能替代这些检查。

索引和查询必须处于兼容向量空间，向量归一化与距离度量也要一致。单位归一化后，余弦相似度与内积的排序一致；未归一化时未必如此。更换模型、模板或维度通常需要重编码或重建索引，并保留可回滚版本。

Embedding选型可用MTEB等公开任务筛候选，最终要在业务标注query/文档集测Recall@k、MRR与nDCG，区分语义相似、检索排名和事实支持。需要适应领域时可以对比训练并挖hard negatives，过滤假负例；更换编码器后索引与query向量必须同步版本。

#### 易错点

- 向量距离小不是事实正确，也不能解释为跨模型通用置信度。

#### 追问

- 如何区分 embedding 失效与 ANN 索引漏召回？

<a id="rag-005"></a>
### RAG-005 · BM25 与向量检索各有什么优势？

**L2**

#### 答案

BM25 通过词项匹配、逆文档频率和词频饱和打分，适合型号、人名和精确术语；向量检索更擅长语义近似，但可能模糊关键数字或否定条件，两者通常可以混合使用。BM25 的 $`k_1`$ 控制词频饱和，$`b`$ 控制文档长度归一化，具体取值需业务验证。

稀有型号、错误码可使用 `exact`/`keyword` 字段，语义问题使用 dense 路径。中文分词、字段配置、同义词、大小写与中英混合都会影响召回，应先检查这些因素，再判断是否是模型能力问题。

#### 易错点

- 关键词检索不是必然落后，也不能用 BM25 分数当概率。

#### 追问

- 涉及 SKU 与价格区间时如何组合结构过滤和语义召回？

<a id="rag-006"></a>
### RAG-006 · 混合检索的分数融合与 RRF 有什么区别？

**L2**

#### 答案

BM25 与向量检索分数的范围和分布不同，直接相加容易让一路占主导。融合可先校准分数再加权，也可用 RRF 累加各检索路的倒数排名贡献 $`1/(k+\mathrm{rank}_j(d))`$，避开原始分数之间的直接比较。

RRF 仍需验证排名常数 $`k`$、候选窗口与去重规则；未进入窗口的文档没有融合机会，重复文档应按稳定 ID 合并。评测按词项、语义和多跳 query 分层比较收益，同时监测 P95 延迟。RRF 分数不能视为余弦相似度或可靠性概率，也不能照搬原检索分数的阈值。

```math
\mathrm{RRF}(d)=\sum_{j:\,d\in L_j}\frac{1}{k+\mathrm{rank}_j(d)}
```

#### 易错点

- RRF 分数不等于余弦相似度或可靠性概率。

#### 追问

- 为何融合后的分数阈值不能照搬原来的 cosine 阈值？

<a id="rag-007"></a>
### RAG-007 · reranker 和 embedding 检索模型如何分工？

**L2**

#### 答案

Embedding 检索用独立表示快速缩小候选范围；cross-encoder 将问题与候选联合输入，利用交互信息判断相关性并重排，通常更精细但成本更高。精排不能找回粗召回完全遗漏的证据，因此先保证 Top-N 候选覆盖，再用 Top-K 精排控制上下文质量与预算。

业务标注还应验证 reranker 的分数校准、多语言和长片段能力。批处理、候选去重、轻量排序与缓存可降低延迟；重排的收益最终仍需由端到端正确率确认，不能只看相关性分数。

#### 易错点

- 不能把精排理解成一定手工加特征权重；模型类型与任务有关。

#### 追问

- rerank 提升相关性却降低端到端正确率时怎样排查？

<a id="rag-019"></a>
### RAG-019 · 怎样训练 embedding/retriever，hard negatives、ICT、SEED 与 REALM 分别解决什么？

**L3**

#### 答案

双塔用query与正/负文档的相似度进行对比训练。in-batch negatives便宜，但常偏容易且可能包含假负例；BM25、当前模型ANN召回或交叉编码器筛选能构造更难负例。按问题来源/实体/文档版本划分评测，防止相同问题改写或答案段落泄漏。

ANCE使用来自动态ANN索引的困难负例，缓解训练样本与检索阶段候选的差异；RocketQA强调跨批次负样本与去噪等训练设计。困难不等于错误标签可靠，应处理多个同样正确文档、近重复段落与领域歧义，避免把相关答案推远。

弱监督预训练中，ICT用句子与其上下文构造检索任务；SEED-Encoder借助较弱decoder形成瓶颈，促使encoder学习有用的全局表示；REALM把潜在文档检索纳入语言模型预训练，按检索文档条件化目标。这些方法改变训练任务或检索/语言目标耦合，不是换一个向量数据库。更换encoder后需同步索引，迭代采负例时关注索引滞后。

```math
\mathcal L_{\mathrm{retriever}}=-\log\frac{\exp(s(q,d^+)/\tau)}{\exp(s(q,d^+)/\tau)+\sum_{d^-}\exp(s(q,d^-)/\tau)}
```

#### 易错点

- 业务未标注为正例的文档不一定是真负例；盲目增加hard negatives可能损害召回。

#### 追问

- 检索模型只在旧索引上挖负例，为什么训练可能越来越不贴近线上？

<a id="topic-3"></a>
## 查询优化与图检索

<a id="rag-009"></a>
### RAG-009 · query rewrite、multi-query 和 HyDE 何时有用？

**L2**

#### 答案

Query rewrite 可补全指代或检索术语，multi-query 通过多种表达增加语义覆盖；HyDE 先生成假设文档，再用其表示检索真实资料。这些方式适用于问题与文档表达差距较大的场景，但可能改变原意、增加噪声和延迟，应保留原问题并比较真实召回收益。

基于历史补全实体时需记录依据，不能丢失关键数值或否定条件。多路结果合并后应去重，避免近似改写耗尽候选预算。HyDE 的假设文本可能有错，只能帮助检索，不能直接作为最终答案的事实来源。

#### 易错点

- 改写不是必加环节，简单准确的 query 可能被越改越差。

#### 追问

- 怎样检测 rewrite 错把用户需求改成另一个问题？

<a id="rag-010"></a>
### RAG-010 · Self-RAG 和多跳检索怎样改善复杂问答？

**L3**

#### 答案

多跳问答可能先定位实体，再根据中间证据检索下一跳；可将问题拆成可验证子问题，记录每跳证据与实体关系，迭代补足缺口。Self-RAG 则是进一步训练 reflection tokens 来控制检索与评价的具体框架，加入一句“自我反思”不等于复现该方法。

迭代步骤只有提供有效证据才有价值，也会增加成本并传播中间错误。应设跳数、token 和重复查询预算及停止条件，避免缺少证据时无限检索；模型自评也不能替代事实核验。

自适应/迭代检索应明确触发机制：CRAG以检索质量评估触发纠正，FLARE依据前瞻生成内容及不确定性触发检索，Self-RAG学习专用reflection tokens。它们不是仅让模型“想一想再搜索”的同名方法；按需检索可能减少无效调用，也可能受不准的自评影响，需要固定预算与停止条件。

#### 易错点

- 模型自评不是事实裁判，错误中间实体会污染后续检索。

#### 追问

- 两跳答案正确时，怎样证明每跳证据真的被使用？

<a id="rag-016"></a>
### RAG-016 · 知识图谱 RAG 与 GraphRAG 何时有用，和多 Agent 有什么关系？

**L2**

#### 答案

图增强检索适合关系密集、多跳或全局归纳的问题。知识图谱保存实体、关系与来源，可以沿结构关系寻找证据；向量检索擅长语义相似文本，二者可组合。LLM抽出的图可能遗漏、混淆实体或捏造关系，必须保留原文回链与权限/版本。

Microsoft GraphRAG先抽取图结构并生成社区报告。Local Search围绕与query相关的实体扩展邻居并结合原文，Global Search利用社区摘要回答语料整体主题类问题；不是所有GraphRAG都等于在图数据库里做几跳遍历，也不能把摘要当作天然无误的事实。

多Agent是决策/执行组织方式，GraphRAG是知识表示与检索方式；Agent可调用GraphRAG，也可以不用图，GraphRAG也不要求多Agent。比较方案应看关系问题的证据召回、事实支持、更新成本、实体链接质量与查询延迟，普通局部事实查询可能用baseline RAG更简单。

#### 易错点

- Graph of Thoughts、Agent执行图和知识图谱分别表示计算结构、控制依赖与实体关系，不能混为同一种图。

#### 追问

- “公司所有子公司”与“整个行业的主要风险”分别适合哪种检索？

<a id="topic-4"></a>
## 评测、权限与多模态 RAG

<a id="rag-011"></a>
### RAG-011 · RAG 如何建立分层评测并定位 bad case？

**L2**

#### 答案

RAG 评测应依次检查检索是否覆盖证据、上下文是否相关，以及答案是否忠于证据、正确且完整。准备带证据位置的业务集，按失败环节归因，并联合报告成本、延迟和拒答；只看 BLEU 或单一总分会掩盖链路瓶颈。

检索可使用 Recall@K、MRR 与 nDCG，多段联合证据还需检查是否完整覆盖。用 oracle evidence 替换真实检索，可分析生成器读取证据的能力上限。自动 judge 与无参考指标均有误差，应通过人工样本校准，并记录模型、prompt、语料和指标版本。

对每个query，MRR取第一个相关结果的倒数排名再平均，未命中记0；AP在每个二值相关命中位置计算Precision并按相关文档总数R_q归一，MAP对query平均。nDCG用等级相关性g_i的gain与位置折扣，再除以同一截断k的理想排序IDCG；下面使用2的g_i次方减1，也有直接用g_i的约定，需固定。R_q=0或IDCG=0时须预先规定剔除或记0，避免除零；AP@k的分母同样应声明。缺失标注可能把真实相关结果算负例，检索相关性也不等于答案证据支持。

```math
\begin{aligned}\mathrm{MRR}&=\frac1Q\sum_q\frac1{\mathrm{rank}_q^{\mathrm{first}}},\quad \text{无命中记 }0\\ \mathrm{AP}(q)&=\frac1{R_q}\sum_{i=1}^{N}\mathrm{Precision@}i\cdot\mathrm{rel}_{q,i}\\ \mathrm{MAP}&=\frac1Q\sum_q\mathrm{AP}(q)\\ \mathrm{DCG@}k&=\sum_{i=1}^k\frac{2^{g_i}-1}{\log_2(i+1)},\quad \mathrm{nDCG@}k=\frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}\end{aligned}
```

#### 易错点

- judge 分数和自动 reference-free 指标都有误差，不是无成本真值。

#### 追问

- 召回提升但答案质量不变，应做哪些消融？

<a id="rag-012"></a>
### RAG-012 · RAG 为什么仍会幻觉，怎样设计引用与拒答？

**L2**

#### 答案

RAG 仍可能检索错误、遗漏条件或让模型忽略证据，因此不能保证消除幻觉。应将回答拆为可核验事实，把关键陈述绑定真实片段，再检查该段落是否确实支持结论；链接存在本身不构成证据。

支持率需与答案覆盖度同时衡量，避免少说或空答获得虚高精度。保存资料的时间、版本、页码和检索结果，便于回查条件。文档冲突、过期或不足时应澄清、补检索或拒答，不能伪造出处；低 temperature 或“只依据资料”的提示也不提供正确性保证。

#### 易错点

- 低 temperature 或“只依据资料”提示不能构成正确性保证。

#### 追问

- 两份文档政策冲突时如何选择并解释？

<a id="rag-013"></a>
### RAG-013 · 企业 RAG 如何处理权限、更新与删除？

**L3**

#### 答案

企业 RAG 必须在检索与数据访问层执行权限控制，身份由服务端可信认证获得，检索前或检索中使用租户与 ACL 过滤。Chunk 应继承原文权限和版本，不能依赖生成后让模型过滤敏感信息，因为证据可能已经进入上下文。

更新需同步原文、索引、缓存与权限元数据，明确同步延迟和 SLA；删除或撤权时应清理残余片段与可见缓存，并验证用户边界。缓存键需考虑用户或租户、权限版本和文档版本，防止跨用户复用或继续命中旧权限结果。

#### 易错点

- 生成后再让模型过滤敏感片段太晚，证据可能已经进入上下文。

#### 追问

- 文档撤权后，如何发现并阻断旧缓存仍被使用？

<a id="rag-014"></a>
### RAG-014 · 长上下文能否替代 RAG？

**L2**

#### 答案

长上下文适合规模可控的完整材料阅读，RAG 适合大型、频繁更新且带权限的知识库按需取证，两者也可组合。选择应看实际证据覆盖、位置鲁棒性、时延与成本；窗口能容纳资料，不代表模型能同样可靠地利用所有位置。

使用单文档、多文档、多跳与长尾事实样本比较直接输入和检索，保持相同答案预算与采样设置，并分别计入索引、prefill 和生成成本。Lost in the Middle 是特定模型与任务的实验现象，当前系统仍需重新测量，不能据此断言某种方案总是更好或更便宜。

Lost in the Middle指某些实验中证据放在中段比首尾更难利用。测试时固定问题和证据内容，随机交换位置并控制干扰段长度，区分检索漏召回与生成漏利用；可尝试重排证据、去冗余、分层摘要或分段问答，但压缩也可能丢掉关键条件，需看位置分层指标。

#### 易错点

- 不要说长上下文必然淘汰 RAG，或 RAG 在一切场景永远更便宜。

#### 追问

- 若证据在正文中段，怎样验证模型是否漏读？

<a id="rag-015"></a>
### RAG-015 · 多模态 RAG 怎样检索图表、扫描 PDF 与视频？

**L3** · 商汤

#### 答案

多模态 RAG 可使用 OCR 文本、图像描述或视觉多向量表示检索，并保留页码、区域和时间戳。ColPali 直接为文档页图建立多向量表示，通过 late interaction 匹配。选择路径时需按信息类型比较 OCR 误差、视觉编码成本与图文混合检索效果，不能仅以图像命中判断读表是否正确。

结果应回传原图、相关区域和表头等证据，让生成器核验单位与位置。视频需先定义片段与时间粒度，多帧证据应能支持时序问题，不能只找到相似画面。视觉检索也不意味着可以省去全部 OCR、文本索引或结构化抽取。

#### 易错点

- 视觉检索不意味着所有 OCR、文本索引与结构化抽取都能省略。

#### 追问

- 一个检索正确但 VLM 读表错误的案例该怎么归因？

<a id="rag-018"></a>
### RAG-018 · RAGFlow、Haystack、LlamaIndex 与 DSPy 怎样分工和选型？

**L2**

#### 答案

RAG框架选型先看文档解析、chunk/元数据、检索与权限、生成编排、评测和部署。RAGFlow提供知识库接入、解析和检索应用的整合体验；Haystack以组件和pipeline构建检索/Agent应用；LlamaIndex围绕数据/索引/检索提供组件，也支持workflow/Agent。功能会重叠，不宜把它们固定分成“平台不会定制、库不能可视化”。

DSPy强调用声明的程序/模块与指标优化提示、示例或模型权重，不是向量数据库，也不是装上就自动得到更好的RAG。优化必须用训练/开发集，留出独立测试集并计算优化调用成本，防止指标过拟合。

用真实PDF/表格/权限更新等困难样本比较解析质量、证据召回、引用正确性、吞吐与恢复能力，确认能否导出数据、固定版本、插入自定义检索/重排与替换模型。更少编码的原型体验和生产控制力可以分别评估，不应以安装教程长度代替技术选型。

#### 易错点

- 框架默认分块和官方示例不保证适合中文合同、扫描PDF或跨页表格。

#### 追问

- 怎样验证框架升级后chunk边界和检索效果没有回归？

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Microsoft GraphRAG Local Search](https://github.com/microsoft/graphrag/blob/main/docs/query/local_search.md)
- [RAG for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge](https://arxiv.org/abs/1906.00067)
- [LLaVA 官方仓库](https://github.com/haotian-liu/LLaVA)
- [Chunk Documents — Azure AI Search](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents)
- [LlamaIndex: How Each Index Works](https://developers.llamaindex.ai/python/framework/module_guides/indexing/index_guide/)
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084)
- [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://arxiv.org/abs/2007.00808)
- [Similarity settings — Elasticsearch](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity)
- [Reciprocal rank fusion — Elasticsearch](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
- [Retrieve & Re-Rank — Sentence Transformers](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [hnswlib — official repository](https://github.com/nmslib/hnswlib)
- [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
- [Self-RAG](https://arxiv.org/abs/2310.11511)
- [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
- [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983)
- [Ragas](https://arxiv.org/abs/2309.15217)
- [Introduction to Information Retrieval: Evaluation of ranked retrieval results](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)
- [FActScore](https://arxiv.org/abs/2305.14251)
- [Document-Level Access Control — Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-document-level-access-overview)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- [ColPali](https://arxiv.org/abs/2407.01449)
- [Microsoft GraphRAG Global Search](https://microsoft.github.io/graphrag/query/global_search/)
- [Faiss IndexIVFFlat](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVFFlat.html)
- [Faiss IndexIVFPQ](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVFPQ.html)
- [RAGFlow documentation](https://ragflow.io/docs/)
- [Introduction to Haystack](https://docs.haystack.deepset.ai/docs)
- [LlamaIndex Framework](https://developers.llamaindex.ai/python/framework/)
- [DSPy Optimizers](https://dspy.ai/3.1.0/learn/optimization/optimizers/)
- [RocketQA](https://arxiv.org/abs/2010.08191)
- [Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://arxiv.org/abs/1906.00300)
- [Less is More: Pre-train a Strong Text Encoder for Dense Retrieval Using a Weak Decoder](https://arxiv.org/abs/2102.09206)
- [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)
