# RAG、检索与重排

[返回仓库首页](../README.md) · [来源目录](../SOURCES.md) · [阅读全部题目](../exports/questions.jsonl)

L1 基础 · L2 进阶 · L3 深入。难度为编辑判断；社区线索不等于雇主确认真题。所有答案为独立整理。

## 题目导航

- [RAG-001 · 一个可落地的 RAG 系统有哪些环节？](#rag-001)
- [RAG-002 · RAG、微调和 prompt engineering 如何选择？](#rag-002)
- [RAG-003 · chunk 大小与 overlap 怎样设置？](#rag-003)
- [RAG-004 · embedding 模型与相似度应如何选？](#rag-004)
- [RAG-005 · BM25 与向量检索各有什么优势？](#rag-005)
- [RAG-006 · 混合检索的分数融合与 RRF 有什么区别？](#rag-006)
- [RAG-007 · reranker 和 embedding 检索模型如何分工？](#rag-007)
- [RAG-008 · HNSW 的 M、ef_construction 和 ef_search 怎样权衡？](#rag-008)
- [RAG-009 · query rewrite、multi-query 和 HyDE 何时有用？](#rag-009)
- [RAG-010 · Self-RAG 和多跳检索怎样改善复杂问答？](#rag-010)
- [RAG-011 · RAG 如何建立分层评测并定位 bad case？](#rag-011)
- [RAG-012 · RAG 为什么仍会幻觉，怎样设计引用与拒答？](#rag-012)
- [RAG-013 · 企业 RAG 如何处理权限、更新与删除？](#rag-013)
- [RAG-014 · 长上下文能否替代 RAG？](#rag-014)
- [RAG-015 · 多模态 RAG 怎样检索图表、扫描 PDF 与视频？](#rag-015)

<a id="rag-001"></a>
## RAG-001 · 一个可落地的 RAG 系统有哪些环节？

**L1 · 社区题目线索** · 标签：RAG / pipeline

**30 秒回答**

离线将文档解析、清洗、切分并建立带元数据的索引；在线把问题转换为检索请求，召回、重排、构造证据上下文，再生成可追溯回答。要把检索、证据使用和生成分别评估，因为端到端答错可能来自完全不同的环节。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 索引保存文档 ID、页码、版本、权限与片段边界，便于更新和引用。
- 生成输入应区分任务指令与外部资料，资料不足时允许拒答或澄清。
- 现代 prompt RAG 不一定训练模型；原始 RAG 论文也研究了检索生成联合微调。

### 易错点

- 不要把 RAG 定义为必须零训练，也不要把它等同向量数据库。

### 面试官可能追问

- 怎样定位是没检索到还是模型没用好证据？

</details>

**技术依据**

- [APP-S116 · Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

**题目出处线索**

- [APP-S009 · 字节大模型算法实习生：电商业务（已 oc）](https://www.nowcoder.com/discuss/724319940982898688) · `reported_question`：正文明确问 RAG 理解与一般工作流程。

<a id="rag-002"></a>
## RAG-002 · RAG、微调和 prompt engineering 如何选择？

**L1 · 社区题目线索** · 标签：RAG / SFT / 选型

**30 秒回答**

prompt 改变当次任务约束，RAG 在推理时提供外部证据，微调改变模型行为分布。知识常更新、需权限和溯源时优先评估 RAG；稳定格式或专门能力可用微调。它们可以组合，仍要比较效果、时延、数据维护和成本。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 别把微调当可靠数据库写入方式：事实可能难以精确更新或撤回。
- RAG 需要解析、召回与证据质量，训练可改善模型读证据或检索能力。
- 用代表性业务样本对 prompt、RAG、SFT 及组合做同预算消融。

### 易错点

- “RAG 只学知识、微调只学能力”是粗略划分，不是严格理论。

### 面试官可能追问

- 客服政策每天更新但答复格式严格，该怎么组合？

</details>

**技术依据**

- [APP-S122 · RAG for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)

**题目出处线索**

- [APP-S004 · 2026 年最新 AI agent 面试（05）：RAG 基础应用](https://zhuanlan.zhihu.com/p/2063918418823230700) · `search_snippet`：知乎搜索摘要包含 RAG 与微调区别。

<a id="rag-003"></a>
## RAG-003 · chunk 大小与 overlap 怎样设置？

**L2 · 社区题目线索** · 标签：chunk / 文档解析

**30 秒回答**

chunk 应覆盖一个可回答的信息单元，同时适配 embedding 与生成器预算。优先按标题、段落、表格结构切分，再比较多个大小与重叠比例。重叠可保留边界上下文，也会重复召回和增加存储，没有固定最优参数。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 表格要保留表头、单位和关联行，代码或公式尽量不在语义中间断开。
- 可采用小片段检索、大父段返回，兼顾定位准确与上下文完整。
- 联合评估 evidence recall、答案质量、重复率与 token 成本。

### 易错点

- 切分字符数不能自动等价为 token 数，中文与代码差异明显。

### 面试官可能追问

- 大 chunk 提高答案质量但降低召回时，你怎样定位原因？

</details>

**技术依据**

- [APP-S124 · Chunk Documents — Azure AI Search](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents)

**题目出处线索**

- [APP-S002 · 字节跳动 AI 应用开发一面面经](https://api-cdn.nowcoder.com/feed/main/detail/15af3788a038477bba99f2f9d94b2cef) · `search_snippet`：摘要明确问切块大小、重叠度与参数对比。

<a id="rag-004"></a>
## RAG-004 · embedding 模型与相似度应如何选？

**L2 · 编辑补充题** · 标签：embedding / dual encoder

**30 秒回答**

双塔分别编码 query 和文档，支持预计算与快速检索。选型要看语言、领域、文档长度和 query/document 指令模板，用业务检索标注集验证。索引与查询必须使用兼容向量空间，归一化和距离度量也要一致。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 专有实体、代码、跨语言与长段落表现可能不同，不能只看通用榜单。
- 单位归一化向量下余弦和内积排序一致；未归一化时不一定。
- 换模型、模板或维度往往要重建/重编码索引，保留可回滚版本。

### 易错点

- 向量距离小不是事实正确，也不能解释为跨模型通用置信度。

### 面试官可能追问

- 如何区分 embedding 失效与 ANN 索引漏召回？

</details>

**技术依据**

- [APP-S117 · Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [APP-S118 · Sentence-BERT](https://arxiv.org/abs/1908.10084)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="rag-005"></a>
## RAG-005 · BM25 与向量检索各有什么优势？

**L2 · 编辑补充题** · 标签：BM25 / dense retrieval

**30 秒回答**

BM25 根据词项匹配、逆文档频率和词频饱和打分，适合型号、人名和精确术语；向量检索更擅长语义近似，但可能模糊关键数字或否定。中文分词与字段配置很关键，通常通过混合检索取长补短。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- k1 控制词频饱和，b 控制文档长度归一化；具体数值需验证。
- 稀有型号、错误码可用 exact/keyword 字段，语义问题走 dense 路径。
- 分词、同义词、大小写和中英混合先检查，别直接归因模型太小。

### 易错点

- 关键词检索不是必然落后，也不能用 BM25 分数当概率。

### 面试官可能追问

- 涉及 SKU 与价格区间时如何组合结构过滤和语义召回？

</details>

**技术依据**

- [APP-S125 · Similarity settings — Elasticsearch](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="rag-006"></a>
## RAG-006 · 混合检索的分数融合与 RRF 有什么区别？

**L2 · 社区题目线索** · 标签：hybrid search / RRF

**30 秒回答**

BM25 和向量分数的范围、分布不同，直接相加容易让一路占主导。可以先校准再加权，或用 RRF 将多路排名转为倒数排名贡献相加。RRF 避免直接比原始分数，但仍需验证排名常数、候选窗口和去重策略。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- RRF 对检索路 j 中排名 rank_j(d) 的文档累加 1/(k+rank_j(d))。
- 窗口决定候选是否有融合机会；重复文档应按稳定 ID 合并。
- 比较词项类、语义类、多跳类 query 的分层收益和 P95 延迟。

### 公式

```text
\operatorname{RRF}(d)=\sum_{j:d\in L_j}\frac{1}{k+\operatorname{rank}_j(d)}
```

### 易错点

- RRF 分数不等于余弦相似度或可靠性概率。

### 面试官可能追问

- 为何融合后的分数阈值不能照搬原来的 cosine 阈值？

</details>

**技术依据**

- [APP-S126 · Reciprocal rank fusion — Elasticsearch](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)

**题目出处线索**

- [APP-S002 · 字节跳动 AI 应用开发一面面经](https://api-cdn.nowcoder.com/feed/main/detail/15af3788a038477bba99f2f9d94b2cef) · `search_snippet`：摘要问向量召回不准时的关键词+语义混合方案。

<a id="rag-007"></a>
## RAG-007 · reranker 和 embedding 检索模型如何分工？

**L2 · 社区题目线索** · 标签：rerank / cross encoder

**30 秒回答**

embedding 检索用独立表示快速缩小候选，cross-encoder 将问题与候选联合输入，利用交互判断相关性后重排。重排一般更精细但成本更高，而且无法找回粗召回阶段完全漏掉的证据，应先保证候选覆盖。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 粗召回 Top-N 关注覆盖，精排 Top-K 关注上下文质量与预算。
- reranker 分数是否校准、支持多语和长片段，需要业务标注验证。
- 批处理、候选去重、轻量排序及缓存可降低精排延迟。

### 易错点

- 不能把精排理解成一定手工加特征权重；模型类型与任务有关。

### 面试官可能追问

- rerank 提升相关性却降低端到端正确率时怎样排查？

</details>

**技术依据**

- [APP-S127 · Retrieve & Re-Rank — Sentence Transformers](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)

**题目出处线索**

- [APP-S002 · 字节跳动 AI 应用开发一面面经](https://api-cdn.nowcoder.com/feed/main/detail/15af3788a038477bba99f2f9d94b2cef) · `search_snippet`：摘要明确列出 Rerank 实现与排序追问。

<a id="rag-008"></a>
## RAG-008 · HNSW 的 M、ef_construction 和 ef_search 怎样权衡？

**L2 · 编辑补充题** · 标签：ANN / HNSW

**30 秒回答**

HNSW 用分层近邻图减少搜索量，以近似召回换取速度。M 影响图连接及内存，ef_construction 影响建图质量，查询 ef 影响搜索宽度和延迟。先与暴力近邻结果比较索引 recall，再单独评估文档的语义相关性。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 增大 M/构建搜索宽度通常提高图质量，也增加内存或建库时间。
- 查询 ef 增大通常提高 ANN recall、增加耗时，需验证并满足实现的 k 约束。
- 过滤条件、删除更新、距离函数及索引版本都会影响结果。

### 易错点

- ANN recall 高不代表找到业务所需证据；两种召回要分开。

### 面试官可能追问

- 精确搜索命中、ANN 未命中时你会调什么？

</details>

**技术依据**

- [APP-S128 · hnswlib — official repository](https://github.com/nmslib/hnswlib)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="rag-009"></a>
## RAG-009 · query rewrite、multi-query 和 HyDE 何时有用？

**L2 · 社区题目线索** · 标签：query rewrite / HyDE

**30 秒回答**

改写可补全指代或检索术语，多查询增加语义覆盖；HyDE 先生成假设文档，再用其表示检索真实资料。它们适合用户问题与文档表达差距较大的场景，也可能改变原意、增加噪声和延迟，需要保留原问题并比较召回收益。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 对历史上下文补全实体时，记录补全依据；关键数值和否定不得丢失。
- 多路结果合并去重，避免相似改写浪费候选预算。
- HyDE 的假设文本可能含错误，不能作为最终答案的事实来源。

### 易错点

- 改写不是必加环节，简单准确的 query 可能被越改越差。

### 面试官可能追问

- 怎样检测 rewrite 错把用户需求改成另一个问题？

</details>

**技术依据**

- [APP-S119 · Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)

**题目出处线索**

- [APP-S004 · 2026 年最新 AI agent 面试（05）：RAG 基础应用](https://zhuanlan.zhihu.com/p/2063918418823230700) · `search_snippet`：知乎摘要描述 rewrite、HyDE 与多角度扩写。

<a id="rag-010"></a>
## RAG-010 · Self-RAG 和多跳检索怎样改善复杂问答？

**L3 · 编辑补充题** · 标签：Self-RAG / multi-hop

**30 秒回答**

复杂问题可能需要先找实体，再根据中间证据检索下一跳。可以迭代检索、验证缺口并设停止条件；Self-RAG 进一步训练 reflection tokens 控制检索与评价。额外步骤只有提供有效证据时才有价值，并会增加成本与传播错误风险。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 将问题拆为可验证子问题，记录每跳证据和实体关系。
- 设置跳数、token 与重复查询预算，防止在无证据时无限检索。
- Self-RAG 是具体训练框架，随便加一句“自我反思”不等于复现它。

### 易错点

- 模型自评不是事实裁判，错误中间实体会污染后续检索。

### 面试官可能追问

- 两跳答案正确时，怎样证明每跳证据真的被使用？

</details>

**技术依据**

- [APP-S120 · Self-RAG](https://arxiv.org/abs/2310.11511)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="rag-011"></a>
## RAG-011 · RAG 如何建立分层评测并定位 bad case？

**L2 · 社区题目线索** · 标签：RAG评测 / recall

**30 秒回答**

先评测检索是否覆盖证据，再评测上下文是否相关，最后看答案是否忠于证据、正确且完整。准备带证据位置的业务集，按失败环节做归因，并加入成本、延迟和拒答指标；只看 BLEU 或单一总分会掩盖链路瓶颈。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 检索可看 Recall@K、MRR/nDCG；多段联合证据需检查完整覆盖。
- 用 oracle evidence 替换真实检索，判断生成器读证据的上限。
- 自动 judge 要与人工样本校准，记录模型、prompt、语料与指标版本。

### 易错点

- judge 分数和自动 reference-free 指标都有误差，不是无成本真值。

### 面试官可能追问

- 召回提升但答案质量不变，应做哪些消融？

</details>

**技术依据**

- [APP-S123 · Ragas](https://arxiv.org/abs/2309.15217)

**题目出处线索**

- [APP-S006 · Awesome-LLM-Interview-Questions-and-Answers](https://github.com/DolbyUUU/Awesome-LLM-Interview-Questions-and-Answers) · `reported_topic`：README 明列 RAG 系统评测。

<a id="rag-012"></a>
## RAG-012 · RAG 为什么仍会幻觉，怎样设计引用与拒答？

**L2 · 社区题目线索** · 标签：幻觉 / grounding

**30 秒回答**

RAG 可能检索错误、遗漏条件或让模型忽略证据，因而不能保证消除幻觉。把答案拆成可验证事实，将关键陈述绑定真实片段，再检查证据是否支持。资料冲突、过期或不足时应澄清、补检索或拒答，不能伪造出处。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 验证引用不只看链接存在，还要看对应段落确实支持该结论。
- 事实支持率需与答案覆盖度结合，避免少说或空答拿高精度。
- 保存时间、版本、页码与检索结果，允许用户回查原文条件。

### 易错点

- 低 temperature 或“只依据资料”提示不能构成正确性保证。

### 面试官可能追问

- 两份文档政策冲突时如何选择并解释？

</details>

**技术依据**

- [APP-S151 · FActScore](https://arxiv.org/abs/2305.14251)

**题目出处线索**

- [APP-S001 · 字节多模态大模型面经一面](https://www.nowcoder.com/discuss/932594519835443200) · `search_snippet`：摘要中出现如何解决幻觉问题。

<a id="rag-013"></a>
## RAG-013 · 企业 RAG 如何处理权限、更新与删除？

**L3 · 编辑补充题** · 标签：ACL / 版本 / 企业RAG

**30 秒回答**

权限应在检索和数据访问层强制执行，文档拆成 chunk 后也要继承权限与版本；不能寄希望于模型拒绝输出。更新需同步原文、索引、缓存和权限元数据，删除时要清理残余片段与可见缓存，并验证用户边界。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 身份由服务端可信认证获得，检索前/中使用租户和 ACL 过滤。
- 权限元数据延迟同步会出现短暂可见性问题，需明确更新 SLA。
- 缓存键考虑用户/租户、权限版本、文档版本，避免跨用户复用泄露。

### 易错点

- 生成后再让模型过滤敏感片段太晚，证据可能已经进入上下文。

### 面试官可能追问

- 文档撤权后，如何发现并阻断旧缓存仍被使用？

</details>

**技术依据**

- [APP-S129 · Document-Level Access Control — Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-document-level-access-overview)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="rag-014"></a>
## RAG-014 · 长上下文能否替代 RAG？

**L2 · 社区题目线索** · 标签：long context / RAG

**30 秒回答**

长上下文适合规模可控的完整材料阅读，RAG 适合大型、频繁更新、带权限的知识库按需取证。两者可组合，选择取决于真实证据覆盖、位置鲁棒性、时延与成本；上下文窗口够长不等于模型能同样可靠地利用所有位置。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 使用单文档、多文档、多跳和长尾事实样本比较直接输入与检索。
- 保持相同答案预算和采样设置，分别计入索引、prefill 与生成成本。
- Lost in the Middle 是特定模型/任务实验现象，当前系统应重新测量。

### 易错点

- 不要说长上下文必然淘汰 RAG，或 RAG 在一切场景永远更便宜。

### 面试官可能追问

- 若证据在正文中段，怎样验证模型是否漏读？

</details>

**技术依据**

- [APP-S121 · Lost in the Middle](https://arxiv.org/abs/2307.03172)

**题目出处线索**

- [APP-S005 · 多模态大模型面试（六）：多模态 RAG/Agent](https://zhuanlan.zhihu.com/p/2059680845074638745) · `search_snippet`：知乎摘要明确问长上下文 VLM 是否淘汰多模态 RAG。

<a id="rag-015"></a>
## RAG-015 · 多模态 RAG 怎样检索图表、扫描 PDF 与视频？

**L3 · 社区题目线索** · 标签：multimodal RAG / ColPali

**30 秒回答**

多模态检索可用 OCR 文本、图像描述或视觉多向量表示，并保留页码、区域和时间戳。ColPali 直接对文档页图建立多向量表示，结合 late interaction 匹配；工程上应按信息类型评估，图像命中并不等于已正确读出表格数值。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 扫描件需对比 OCR 误差、视觉编码成本与图文混合检索效果。
- 结果回传保留原图/区域/表头等证据，让生成器可核对单位和位置。
- 视频先定义片段和时间粒度，多帧证据要支持时序问题而非仅相似画面。

### 易错点

- 视觉检索不意味着所有 OCR、文本索引与结构化抽取都能省略。

### 面试官可能追问

- 一个检索正确但 VLM 读表错误的案例该怎么归因？

</details>

**技术依据**

- [APP-S130 · ColPali](https://arxiv.org/abs/2407.01449)

**题目出处线索**

- [APP-S005 · 多模态大模型面试（六）：多模态 RAG/Agent](https://zhuanlan.zhihu.com/p/2059680845074638745) · `search_snippet`：文章标题和摘要明确覆盖多模态 RAG 与多轮多模态检索。
