# 后训练、RAG、Agent 与评测：检索及核验记录

访问日期：2026-10-02。数据文件为 `interviews/data/alignment_rag.json`，共 60 题：ALN 20、RAG 15、AGT 10、EVA 15。12 条社区来源用于发现题目或主题，65 条论文/官方来源用于核验。33 题带社区线索，27 题为编辑补充。

## 证据与访问口径

- `full` 表示能读取正文且阅读了相关章节，不表示逐字读完整篇长文。`partial` 表示正文公开部分可读、后部受付费限制。
- `snippet` 表示仅读搜索摘要或论文摘要；来源 notes 明确区分。搜索可见而正文受阻的页面仍记 `snippet`，不会伪装成读过全文。
- `reported_question` 是作者声称问过的题；`reported_topic` 是题目目录或概括主题；`search_snippet` 是搜索可见题目线索；`secondary_report` 是合集的二手报道。都不等于已核实的企业真题。
- `claimed_company` 来自页面自称，用于公司线索索引，不代表雇主确认，也不代表出现频率。
- 答案均独立综合，社区答案未照抄；不搬运大段题单、付费尾部或热度统计。单个社区页只抽取少量主题。
- 本分片没有直接读取小红书原帖；小红书访问与二手来源由工程分片单独记录。知乎正文超时/不可读时仅使用检索摘要。

## 查询记录

以下为实际执行的检索表达式汇总；论文编号和官方文档另以 URL 直接打开核验。

|检索方向|查询|对应来源|
|---|---|---|
|社区发现|site.zhihu.com 大模型 面试 DPO PPO RAG Agent 评测；site.nowcoder.com 大模型 面经 RLHF RAG DPO；site.zhihu.com/question 大模型 面试 RAG DPO PPO；site.zhihu.com/p 大模型 面经 RAG Agent|APP-S001—APP-S012|
|社区补充|site.nowcoder.com 面经 大模型 评测 幻觉 安全 RLHF；site.zhihu.com 面试 LLM-as-Judge 评测 数据泄露；site.nowcoder.com 面经 PPO GRPO 奖励模型 优势函数|APP-S007、APP-S010—APP-S012|
|检索工程|site.elastic.co docs BM25 RRF reciprocal rank fusion；site.sbert.net retrieve rerank cross encoder documentation；site.learn.microsoft.com azure ai search chunking document level access control；site.arxiv.org ColPali efficient document retrieval vision language models|APP-S124—APP-S130|
|Agent|site.docs.langchain.com oss python langgraph durable execution persistence memory；site.modelcontextprotocol.io specification architecture tools security；site.anthropic.com engineering building effective agents context engineering；site.arxiv.org indirect prompt injection compromising llm integrated applications|APP-S131—APP-S141、APP-S158、APP-S163|
|评测与安全|site.arxiv.org HELM holistic evaluation language models 2211.09110；site.arxiv.org Judging LLM-as-a-Judge MT-Bench Chatbot Arena 2306.05685；site.arxiv.org TruthfulQA Measuring How Models Mimic Human Falsehoods；site.arxiv.org Extracting Training Data from Large Language Models 2012.07805；site.arxiv.org benchmark contamination language models Detecting 2023|APP-S142—APP-S150|
|事实性与对齐|site.arxiv.org Safe RLHF Safe Reinforcement Learning Human Feedback 2023；site.arxiv.org FActScore fine grained factual precision long form text generation；site.arxiv.org Language Models Mostly Know What They Know|APP-S115、APP-S151—APP-S152|

## 社区线索及使用位置

每一题的 `community_evidence.note` 保留更细的对应说明。未取得社区证据的题保留 `editorial=true`。

|来源|访问范围|抽取范围与限制|对应题号|
|---|---|---|---|
|[APP-S001：字节多模态大模型面经一面](https://www.nowcoder.com/discuss/932594519835443200)|snippet|搜索可见 RLHF、DPO loss、PPO/DPO trade-off、幻觉；全文请求受阻。作者自述公司与经历未经核实。|ALN-006、ALN-008、RAG-012|
|[APP-S002：字节跳动 AI 应用开发一面面经](https://api-cdn.nowcoder.com/feed/main/detail/15af3788a038477bba99f2f9d94b2cef)|snippet|搜索可见 RAG 流程、chunk、混合召回、rerank、多 Agent；全文受阻，仅作搜索题目线索。|RAG-003、RAG-006、RAG-007、AGT-005|
|[APP-S003：面了一轮 Agent 岗，我把问过的问题整理成了文章](https://ac.nowcoder.com/discuss/1680599)|snippet|搜索可见上下文、幂等重试、多 Agent、RLHF/DPO、点赞点踩转偏好数据；全文受阻。|ALN-009、AGT-003、AGT-004|
|[APP-S004：2026 年最新 AI agent 面试（05）：RAG 基础应用](https://zhuanlan.zhihu.com/p/2063918418823230700)|snippet|搜索可见 RAG 与微调、HyDE、召回；全文超时。公司归属未核实，答案未采用。|RAG-002、RAG-009|
|[APP-S005：多模态大模型面试（六）：多模态 RAG/Agent](https://zhuanlan.zhihu.com/p/2059680845074638745)|snippet|搜索可见多模态检索、RAG/Agent、长上下文；全文超时。未采用固定成本倍数与绝对化结论。|RAG-014、RAG-015|
|[APP-S006：Awesome-LLM-Interview-Questions-and-Answers](https://github.com/DolbyUUU/Awesome-LLM-Interview-Questions-and-Answers)|full|读取公开 README 题目目录；作者自述面试经历，答案主要由模型生成，仅选少量主题重新核验。|ALN-010、ALN-016、RAG-011、AGT-001、AGT-006|
|[APP-S007：大模型算法岗面试题复盘：RAG、Agent、评测](https://nanhubrain.csdn.net/6a3ce7a8662f9a54cb841a94.html)|full|公开正文可读，只选自动指标、人工评测、KL penalty、PPO 稳定性主题。原答案有过度简化，未照抄。|ALN-003、ALN-005、EVA-002、EVA-005|
|[APP-S008：字节面经：大模型算法岗面经 04](https://www.nowcoder.com/discuss/922308546966847488)|partial|公开正文为汇总面经，尾部付费截断；只选 PPO advantage、GRPO clip、过程奖励主题。属于二手报道，公司归属未核实。|ALN-003、ALN-004、ALN-014|
|[APP-S009：字节大模型算法实习生：电商业务（已 oc）](https://www.nowcoder.com/discuss/724319940982898688)|full|公开正文可读，作者自述 SFT/DPO、损失、变种、评测、RAG；只选少量主题。|ALN-001、ALN-018、RAG-001|
|[APP-S010：字节大模型实习算法面经 55min](https://www.nowcoder.com/feed/main/detail/59b472d7ec6645cc95ac0735885234e3)|snippet|搜索摘要列出 DPO/PPO/GRPO、在线离线 RL、多目标奖励冲突；未读取全文，自述未核实。|ALN-019、ALN-020|
|[APP-S011：LLM as a Judge：如何给 AI 工作台做自动化测试？](https://zhuanlan.zhihu.com/p/2052331223993921976)|snippet|只读搜索摘要，是技术讨论而非面经，用于评判模型校准主题线索。|EVA-003|
|[APP-S012：月之暗面 AI Agent 开发岗一面面经（含答案）](https://www.nowcoder.com/discuss/922643334898647040)|snippet|搜索摘要含工具 schema、成本、幻觉与结果验证；公司归属未经独立核验。|AGT-002、AGT-010|

## 论文/官方材料核验与对应

摘要来源仅核验方法目标或总体结论。涉及损失的关键公式实际阅读正文/PDF/官方实现：DPO（APP-S104/159）、GAE（APP-S103）、PPO（APP-S105）、GRPO（APP-S106/107/162）、RM（APP-S160）、reference KL（APP-S161）、RRF（APP-S126）、pass@k（APP-S146）。题目中的工程建议和实验设计属于编辑综合，不声称都是原文结论。

|来源|读取范围|核验记录|对应题号|
|---|---|---|---|
|[APP-S101：Training language models to follow instructions with human feedback](https://arxiv.org/html/2203.02155v1)|full|读取 §3 RLHF、§4.2 alignment tax、PPO-ptx 公式。|ALN-001、ALN-016|
|[APP-S102：Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)|snippet|只核验 arXiv 摘要；clip 公式由官方 Spinning Up 另核对。|ALN-003|
|[APP-S103：Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438)|full|读取 PDF 第 4 页 Eq.16-18，核验 GAE 与 bias/variance 条件。|ALN-004|
|[APP-S104：Direct Preference Optimization](https://arxiv.org/html/2305.18290v3)|full|读取 §3-5 Eq.3-7、附录推导；核验 KL、BT 假设、分区函数抵消。|ALN-001、ALN-006|
|[APP-S105：PPO — Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)|full|读取 clip 公式、正负 advantage、KL early stopping。|ALN-003|
|[APP-S106：DeepSeekMath](https://arxiv.org/html/2402.03300v3)|full|读取 §4.1 Eq.21、outcome/process supervision，核验原始 GRPO。|ALN-010|
|[APP-S107：Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/html/2503.20783v2)|full|读取 §3.1-3.2：响应长度、问题难度重加权与 Dr. GRPO。|ALN-012|
|[APP-S108：Is DPO Superior to PPO for LLM Alignment?](https://arxiv.org/abs/2404.10719)|snippet|只核验摘要；单篇实验优势不外推为普遍排序。|ALN-008、ALN-019|
|[APP-S109：UltraFeedback](https://arxiv.org/html/2310.01377v1)|full|读取反馈构造、多维评价章节；清洗建议为独立综合。|ALN-009|
|[APP-S110：DeepSeek-R1](https://arxiv.org/html/2501.12948v1)|full|读取 rule-based reward、cold start；规则奖励不等于万能评分。|ALN-013|
|[APP-S111：Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)|snippet|只核验摘要：过程/结果监督与 MATH 实验范围。|ALN-014|
|[APP-S112：Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)|snippet|只核验摘要的代理奖励过优化，不外推拟合曲线。|ALN-015|
|[APP-S113：Constitutional AI](https://arxiv.org/abs/2212.08073)|snippet|只核验摘要的自我修订、AI 偏好、RLAIF 阶段。|ALN-017|
|[APP-S114：A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)|snippet|只核验 ΨPO/IPO 摘要与偏好转奖励假设，不提供未读公式。|ALN-018|
|[APP-S115：Safe RLHF](https://arxiv.org/abs/2310.12773)|snippet|只核验摘要：有用性/无害性冲突；工程步骤为独立综合。|ALN-020|
|[APP-S116：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)|snippet|只核验摘要中的参数/非参数记忆；现代工程链路为综合。|RAG-001|
|[APP-S117：Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)|snippet|只核验摘要的 dual encoder；不引用未读对比损失公式。|RAG-004|
|[APP-S118：Sentence-BERT](https://arxiv.org/abs/1908.10084)|snippet|只核验摘要的独立 embedding、余弦检索。|RAG-004|
|[APP-S119：Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)|snippet|只核验 HyDE 摘要：假设文档非证据，检索真实文档。|RAG-009|
|[APP-S120：Self-RAG](https://arxiv.org/abs/2310.11511)|snippet|只核验摘要中的按需检索与 reflection tokens。|RAG-010|
|[APP-S121：Lost in the Middle](https://arxiv.org/abs/2307.03172)|snippet|只核验摘要的位置敏感现象，不外推所有当前模型。|RAG-014|
|[APP-S122：RAG for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)|snippet|仅核验作者摘要和范围，选型为独立综合。|RAG-002|
|[APP-S123：Ragas](https://arxiv.org/abs/2309.15217)|snippet|只核验摘要的检索、忠实度、生成质量维度，不提供版本 API。|RAG-011|
|[APP-S124：Chunk Documents — Azure AI Search](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents)|full|读取切分因素、overlap；512 tokens/25% 仅起点，不当最优值。|RAG-003|
|[APP-S125：Similarity settings — Elasticsearch](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity)|full|读取 BM25 k1/b 定义，不把默认值当算法要求。|RAG-005|
|[APP-S126：Reciprocal rank fusion — Elasticsearch](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)|full|核验 RRF 公式、排名起点、rank_constant 与窗口。|RAG-006|
|[APP-S127：Retrieve & Re-Rank — Sentence Transformers](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)|full|读取 bi-encoder/cross-encoder 检索重排分工。|RAG-007|
|[APP-S128：hnswlib — official repository](https://github.com/nmslib/hnswlib)|full|读取 M、ef_construction、ef 与 recall/speed；ANN recall 不等于语义召回。|RAG-008|
|[APP-S129：Document-Level Access Control — Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-document-level-access-overview)|full|读取 query-time ACL 与权限同步；部分功能 preview，不外推 API。|RAG-013|
|[APP-S130：ColPali](https://arxiv.org/abs/2407.01449)|snippet|只核验作者摘要的页图、多向量、late interaction。|RAG-015|
|[APP-S131：ReAct](https://arxiv.org/abs/2210.03629)|snippet|只核验摘要 reasoning/action 交错，工程实现为综合。|AGT-001|
|[APP-S132：AutoGen](https://arxiv.org/abs/2308.08155)|snippet|只核验摘要的可配置 Agent 与对话编排。|AGT-005|
|[APP-S133：AgentBench](https://arxiv.org/abs/2308.03688)|snippet|只核验摘要的交互环境、长期决策评测。|AGT-008|
|[APP-S134：Persistence — LangGraph](https://docs.langchain.com/oss/python/langgraph/persistence)|full|durable-execution 重定向此页，读取 checkpointer/store 与恢复。|AGT-004|
|[APP-S135：Memory overview — LangChain](https://docs.langchain.com/oss/python/concepts/memory)|full|读取 thread scope、cross-thread memory、namespace。|AGT-004|
|[APP-S136：Building effective agents — Anthropic](https://www.anthropic.com/engineering/building-effective-agents)|full|读取 workflow/agent、停止条件、环境反馈；工具生态为历史版本。|AGT-009|
|[APP-S137：Writing effective tools for agents — Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)|full|读取严格输入输出、分页、截断、错误反馈。|AGT-002|
|[APP-S138：Effective context engineering — Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)|full|读取 compaction、structured notes、multi-agent。|AGT-010|
|[APP-S139：Demystifying evals for AI agents — Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)|full|读取 trace、task、grader、多次 trial；只采用评测结构。|EVA-015|
|[APP-S140：MCP Architecture (2025-06-18 revision)](https://modelcontextprotocol.io/specification/2025-06-18/architecture)|full|读取历史指定版本 host/client/server、JSON-RPC，不当 2026 最新版。|AGT-006|
|[APP-S141：Indirect Prompt Injection in LLM-integrated applications](https://arxiv.org/abs/2302.12173)|snippet|只核验摘要：被检索数据携带间接注入，数据与指令边界。|AGT-007|
|[APP-S142：Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)|snippet|只核验摘要的多场景多指标与标准化条件。|EVA-012|
|[APP-S143：Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/html/2306.05685v4)|full|读取 §3 偏差与人工一致性，保留 self-enhancement 无确定结论。|EVA-003、EVA-005|
|[APP-S144：TruthfulQA](https://arxiv.org/abs/2109.07958)|snippet|只核验摘要的模仿常见错误，不外推历史排名。|EVA-006|
|[APP-S145：Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805)|snippet|只核验摘要查询可恢复训练文本/个人信息，防御为综合。|EVA-010|
|[APP-S146：Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374)|full|读取 §3.1 Eq.1，第 2 页 pass@k 与功能正确性。|EVA-014|
|[APP-S147：MMMU](https://arxiv.org/abs/2311.16502)|snippet|只核验摘要的学科、图像类型、感知推理范围。|EVA-011|
|[APP-S148：HarmBench](https://arxiv.org/abs/2402.04249)|snippet|只核验摘要的红队、拒答标准化评测。|EVA-008|
|[APP-S149：Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)|snippet|只核验摘要的模型生成测试、危害多样性。|EVA-009|
|[APP-S150：Rethinking Benchmark and Contamination with Rephrased Samples](https://arxiv.org/abs/2311.04850)|snippet|只核验摘要的改写/翻译污染、新鲜测试，不外推污染比例。|EVA-001|
|[APP-S151：FActScore](https://arxiv.org/abs/2305.14251)|snippet|只核验摘要的原子事实支持率，不等于完整度。|RAG-012|
|[APP-S152：Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)|snippet|只核验摘要 P(True)/P(IK) 与跨任务校准困难。|EVA-007|
|[APP-S153：Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911)|snippet|只核验摘要的规则可验证指令，不等于通用语义质量。|EVA-013|
|[APP-S154：scipy.stats.bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)|full|读取 paired=True 同索引重采样与置信区间。|EVA-004|
|[APP-S155：BLEU: A Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)|snippet|读取 ACL 作者摘要与元数据，仅据原任务范围。|EVA-002|
|[APP-S156：ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)|snippet|读取 ACL 作者摘要与元数据，仅据原任务范围。|EVA-002|
|[APP-S157：HaluEval](https://arxiv.org/abs/2305.11747)|snippet|只核验摘要与来源冲突/无法验证幻觉定义、识别任务范围。|EVA-006|
|[APP-S158：MCP Architecture (2026-07-28 revision)](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/architecture/index.mdx)|snippet|仅核验搜索可见官方段落；请求/能力机制与 2025 版不同，具体实现需固定版本。|AGT-006|
|[APP-S159：DPO Trainer — TRL](https://huggingface.co/docs/trl/dpo_trainer)|full|读取 beta、implicit reward、logit 公式与 logged metrics；不固定当前默认参数。|ALN-007|
|[APP-S160：Reward Modeling — TRL](https://huggingface.co/docs/trl/reward_trainer)|full|读取 Bradley–Terry 损失与奖励平移不可辨识性。|ALN-002|
|[APP-S161：Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/pdf/2307.04964)|full|PDF 可读；核验参考策略 KL 与 PPO rollout/update 模型关系。|ALN-005|
|[APP-S162：GRPO Trainer — TRL](https://huggingface.co/docs/trl/grpo_trainer)|full|读取优势归一化、difficulty bias、loss reduction；文档与原始 GRPO 定义分开。|ALN-011|
|[APP-S163：Making retries safe with idempotent APIs — AWS Builders' Library](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)|full|读取 client request identifier、参数意图、原子幂等记录与迟到请求；Agent 场景建议为综合。|AGT-003|
|[APP-S164：Online DPO Trainer — TRL](https://huggingface.co/docs/trl/online_dpo_trainer)|full|读取 Overview、Expected dataset type：当前策略生成候选并由奖励模型或 judge 形成在线偏好；接口属实验模块，题库不依赖具体 API。|ALN-019|
|[APP-S165：XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://arxiv.org/abs/2308.01263)|snippet|仅读摘要，核验过度拒答及安全/不安全对照设计；未读全文或搬运提示词。|EVA-008|

## 易错点处理

- DPO 推导明确 BT 偏好模型、正 β、参考策略支持及同一 prompt 的分区函数抵消；不把有限数据优化说成与 PPO 完全等价。在线 DPO 用官方 TRL 文档核对，避免把 DPO 永远归为离线算法。
- PPO 的 rollout 旧策略与 RLHF 的冻结 reference 是两个对象；GAE 是一种优势估计，GRPO 的组内相对优势并非默认使用 GAE。
- GRPO 的标准差归一化重加权与响应长度归一化分别讨论。二值奖励全对/全错时 ε 不会制造区分信号；Dr. GRPO 的“无偏”限定在其分析目标，不扩张成通用保证。
- pass@k 阅读论文 Eq.(1)，注明 n≥k、固定生成分布的独立采样前提与测试 oracle 的条件；不把其直接当部署成功率。
- MCP 同时保留 2025-06-18 及 2026-07-28 架构的核验线索，版本变化明确标注，不复述旧版会话流程为永久事实。
- Judge 的位置/长度等偏差、评测集污染、成组 bootstrap 和任务切片均以独立评测设计表达；技术讨论来源 APP-S011 明确不是面经。

## 受阻、纠错与未采用

- 牛客 APP-S001—APP-S003 正文请求失败；知乎 APP-S004—APP-S005 正文超时。题库仅收搜索可见主题，并保留来源限制。
- APP-S008 公开部分后有付费截断，不读取或推断隐藏题目。它是汇总面经，相关题用 `secondary_report`。
- APP-S006 作者说明部分答案由模型生成；只取题目目录，技术答案另核验。APP-S007 原答存在过度简化，未采用其错误说明。
- 个别 arXiv HTML 路径不可用时改读 PDF 或摘要，例如 Safe RLHF 保持摘要口径、PPO implementation secrets 改读 PDF。一次检索误命中的材料主题不相关，已排除，未列为来源。
- 不采用社区声称的“高频”“必考”、固定成本倍数或无法复核的招聘/实验数字。发布日期未知保留 null，不把搜索引擎的抓取时间当发表日期。

## 自检

60 个题号唯一，类别计数符合要求；每题至少一个可追踪的 primary 引用，所有社区引用存在且类型正确。quick 长度均在 60–140 字内（本分片最短 88、最长 117），detail 为 3 条，pitfalls 和 followups 各 1 条；无社区证据的 27 题全部标为编辑补充。该核验不替代跨分片审阅。

## 同伴审阅后修订

2026-10-02 采纳多模态同伴审阅：ALN-001 补 DPO 原论文 APP-S104；EVA-008 实际读 XSTest 摘要并新增 APP-S165（snippet）；EVA-014 补固定生成分布/独立采样的估计前提，明确不能机械套到自适应或去重候选。未改变题目数量及社区证据标签。
