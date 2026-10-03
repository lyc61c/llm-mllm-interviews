# 评测、幻觉与安全

[返回仓库首页](../README.md) · [来源目录](../SOURCES.md) · [阅读全部题目](../exports/questions.jsonl)

L1 基础 · L2 进阶 · L3 深入。难度为编辑判断；社区线索不等于雇主确认真题。所有答案为独立整理。

## 题目导航

- [EVA-001 · 如何避免训练数据污染评测集？](#eva-001)
- [EVA-002 · 怎样评价文本生成质量？BLEU/ROUGE、PPL与人工评测如何组合？](#eva-002)
- [EVA-003 · LLM-as-a-Judge 有哪些偏差，如何校准？](#eva-003)
- [EVA-004 · 两个模型只差一个百分点，怎样判断提升可靠？](#eva-004)
- [EVA-005 · 如何设计人工评测，降低标注分歧？](#eva-005)
- [EVA-006 · 如何区分和测量事实幻觉与上下文不忠实？](#eva-006)
- [EVA-007 · 模型自报置信度可靠吗，怎样设计拒答阈值？](#eva-007)
- [EVA-008 · 安全评测如何兼顾越狱成功率与过度拒答？](#eva-008)
- [EVA-009 · 如何构建红队测试而不只刷固定攻击集？](#eva-009)
- [EVA-010 · 训练数据泄露与 RAG 越权泄露如何区分？](#eva-010)
- [EVA-011 · MLLM 评测如何区分感知、OCR 与推理错误？](#eva-011)
- [EVA-012 · 怎样评估 prompt 扰动、多语言和分布外鲁棒性？](#eva-012)
- [EVA-013 · 如何验证 prompt 优化确实有效？](#eva-013)
- [EVA-014 · 代码生成的 pass@k 是什么，如何避免错误估计？](#eva-014)
- [EVA-015 · 线上如何监控、验收并回归 LLM/Agent 系统？](#eva-015)

<a id="eva-001"></a>
## EVA-001 · 如何避免训练数据污染评测集？

**L2 · 编辑补充题** · 标签：data contamination / leakage

**30 秒回答**

训练/测试题目或答案重叠会夸大泛化能力，近重复、翻译和模型改写也可能造成污染。按来源、实体、用户或时间划分数据，做精确和语义去重，保留新鲜私有测试与版本；不知道训练语料时，应明确污染检测的局限。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 仅 n-gram 去重不足以发现语义改写和答案泄漏。
- 调参验证集与最终测试集分开，避免反复看测试结果选方案。
- 合成数据生成提示也可能包含公开题，需追踪源问题与教师输出。

### 易错点

- 检测不到重叠不等于证明无污染；不能只改几个词就称新基准。

### 面试官可能追问

- 项目数据量小，按用户拆分后类别不平衡怎么处理？

</details>

**技术依据**

- [APP-S150 · Rethinking Benchmark and Contamination with Rephrased Samples](https://arxiv.org/abs/2311.04850)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="eva-002"></a>
## EVA-002 · 怎样评价文本生成质量？BLEU/ROUGE、PPL与人工评测如何组合？

**L1 · 社区题目线索** · 标签：metrics / BLEU / ROUGE / PPL / TeacherForcing / 自由生成 / BlindPairwise / PromptStrata / Memorization

**30 秒回答**

文本评测先按任务选指标，再在固定生成协议下比较输出。BLEU/ROUGE刻画参考文本重合，teacher-forced PPL衡量真实前缀下的预测概率，都不足以代表自由生成质量。结合独立分层prompt、事实/执行检查与盲评成对比较，并排查污染和背诵。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- EM/token F1适合有明确参考答案的任务，但需固定规范化、中文分词和别名处理；BLEU原用于翻译，ROUGE常用于摘要。报告分数变体、tokenizer、corpus/sentence聚合与多参考处理；多参考可覆盖部分合理改写，但词面重合仍不保证事实、逻辑或指令遵循，任务适用性不能外推为通用LLM质量。
- Teacher-forced PPL用评测文本真实历史x_<t预测x_t，是有效目标平均NLL的指数；自由生成则会把自己的输出纳入后续历史，两种条件分布不同。比较PPL须统一tokenizer、数据、模板、loss mask和上下文窗口/stride；低PPL不保证回答有用、事实正确或在多轮错误后能恢复，也不能直接跨不同分词粒度排名。
- 建立未用于调参的held-out prompt集，按任务、领域、语言、难度、长度与风险分层，报告各层和按目标流量加权结果。检查与训练/调参材料的重复、近似改写和模板泄漏，加入新写问题及表述变体测试；背诵参考可抬高词面分数，不能据单个高分/低PPL就证明训练污染或泛化。训练数据未知时明确该核验边界。
- 自由生成固定并记录模型/checkpoint、system prompt、chat template、检索/工具条件、temperature/top-p/top-k、最大输出长度和停止规则，明确资源预算；随机采样用多次trial和seed记录，seed本身不保证跨后端逐bit重现。必要时同时比较多个预先声明的解码协议，以免把解码变化误当模型能力变化。
- 人工评测用含锚点的rubric分别评准确性、材料忠实性、完整性、遵循指令和可读性；隐藏模型身份，随机A/B顺序，允许平局、无法判定及分歧复核。事实任务用证据核验，代码用执行测试；LLM judge需校准位置/冗长偏差。报告样本数、一致性及按prompt统计的不确定度，不用一个平均词面分数替代全部结论。

### 公式

```text
PPL=exp[−Σ_{t∈valid} log p_θ(x_t|真实评测前缀x_<t)/N_valid]；自由生成质量另按固定decode协议评价
```

### 易错点

- 将BLEU/ROUGE或teacher-forced PPL当作所有开放生成任务的统一质量，或只看训练loss验证泛化。
- 一边改变prompt、检索、解码或输出预算一边比较模型，或把未查证的训练污染当成高分的唯一解释。

### 面试官可能追问

- 两个模型PPL相近，但盲评偏好明显不同，应怎样分层定位原因？
- 如何区分正确改写导致低ROUGE、事实错误导致高ROUGE，以及对参考答案的背诵？

</details>

**技术依据**

- [APP-S155 · BLEU: A Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf)
- [APP-S156 · ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)
- [CORE-S043 · Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)
- [APP-S143 · Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/html/2306.05685v4)
- [APP-S145 · Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805)
- [APP-S150 · Rethinking Benchmark and Contamination with Rephrased Samples](https://arxiv.org/abs/2311.04850)

**题目出处线索**

- [APP-S007 · 大模型算法岗面试题复盘：RAG、Agent、评测](https://nanhubrain.csdn.net/6a3ce7a8662f9a54cb841a94.html) · `reported_question`：正文明确问生成指标及 BLEU/ROUGE 局限。

<a id="eva-003"></a>
## EVA-003 · LLM-as-a-Judge 有哪些偏差，如何校准？

**L2 · 社区题目线索** · 标签：LLM-as-Judge / bias

**30 秒回答**

模型裁判能降低开放回答评测成本，但会受位置、长度、提示和领域能力影响。用盲测 rubric、交换候选顺序、可核验参考答案和人工金标准检查一致性，固定模型与提示版本。自动评判应报告不确定样本及系统性误判。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 记录评分理由和证据，数学、代码优先使用执行或参考验证。
- 随机顺序与双向比较缓解位置偏差；冗长答案需按内容质量打分。
- 用人工分层样本估计 judge 的召回/误报与领域差异。

### 易错点

- 原论文 self-enhancement 实验不足以下确定结论，不能笼统声称所有裁判必偏爱自己。

### 面试官可能追问

- 裁判和被评模型是同一家时，你如何检验风格偏差？

</details>

**技术依据**

- [APP-S143 · Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/html/2306.05685v4)

**题目出处线索**

- [APP-S011 · LLM as a Judge：如何给 AI 工作台做自动化测试？](https://zhuanlan.zhihu.com/p/2052331223993921976) · `search_snippet`：知乎技术文章摘要讨论 judge 质量与人工黄金集校准；并非面经。

<a id="eva-004"></a>
## EVA-004 · 两个模型只差一个百分点，怎样判断提升可靠？

**L2 · 编辑补充题** · 标签：statistics / bootstrap

**30 秒回答**

必须先看样本量与逐题配对差异，报告置信区间，必要时做配对检验或 bootstrap。对同一测试题比较时应一起重采样两模型结果，相关对话按用户/任务成组；还要考虑多次调参和不同随机种子的影响。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 保存逐样本结果，估计分数差的区间，比只看两个独立均值更有信息。
- 重复题、同用户和同文档不是独立样本，采用 cluster 级抽样更合理。
- 区分统计显著与业务收益，报告成本、延迟与关键分组退化。

### 易错点

- 区间是否重叠不能替代正确的差值检验，反复挑最好 seed 也会偏高。

### 面试官可能追问

- 如果总分升但关键安全分组下降，你会怎样验收？

</details>

**技术依据**

- [APP-S154 · scipy.stats.bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="eva-005"></a>
## EVA-005 · 如何设计人工评测，降低标注分歧？

**L2 · 社区题目线索** · 标签：human eval / rubric

**30 秒回答**

人工评测应定义准确性、完整性、证据、可读性等维度及锚点样例，隐藏模型身份并随机顺序。分层抽样覆盖普通与困难案例，保留平局、分歧和复核；专家不足时要限制领域结论，不能将个人偏好当统一真值。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 先做小规模试标，修订含糊规则，再独立多标注与分歧仲裁。
- 评估事实正确性时提供可信材料，偏好评估与 correctness 分开。
- 报告一致性、样本构成与不确定度，不仅给一个总胜率。

### 易错点

- 只挑中等分样本或成功案例会改变目标分布，须说明抽样与权重。

### 面试官可能追问

- 专家与普通用户偏好相反时如何制定业务 rubric？

</details>

**技术依据**

- [APP-S143 · Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/html/2306.05685v4)

**题目出处线索**

- [APP-S007 · 大模型算法岗面试题复盘：RAG、Agent、评测](https://nanhubrain.csdn.net/6a3ce7a8662f9a54cb841a94.html) · `reported_question`：正文明确问人工评测流程；本题修正其只选中等样本的建议。

<a id="eva-006"></a>
## EVA-006 · 如何区分和测量事实幻觉与上下文不忠实？

**L2 · 编辑补充题** · 标签：hallucination / faithfulness

**30 秒回答**

事实幻觉指陈述缺乏事实支持或错误，上下文不忠实指输出与给定材料矛盾或超出其依据。应先确定评价的证据范围，再对实体、数字、关系和引用分解核验。可用专门数据集做回归，但识别幻觉的能力不等于生成时从不幻觉。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 正确外部事实也可能超出任务限定材料；错误材料上的忠实答案也未必真实。
- 分开 unsupported、contradicted、不可验证与过期，避免全部粗记为错误。
- 用真实业务案例和可靠来源复核，覆盖缺证据、误导前提和冲突文档。

### 易错点

- 常见基准只覆盖部分类型，低温或更大参数不保证事实正确。

### 面试官可能追问

- 如何评价模型纠正了资料中的错误，但没有解释证据冲突？

</details>

**技术依据**

- [APP-S144 · TruthfulQA](https://arxiv.org/abs/2109.07958)
- [APP-S157 · HaluEval](https://arxiv.org/abs/2305.11747)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="eva-007"></a>
## EVA-007 · 模型自报置信度可靠吗，怎样设计拒答阈值？

**L3 · 编辑补充题** · 标签：calibration / abstention

**30 秒回答**

“我有九成把握”并不天然对应九成正确率。可结合验证集、答复正确标签和置信信号做校准，按风险选择拒答阈值并画覆盖率—错误率曲线。新领域或分布变化时需要重新验证，不能只看平均准确率。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- token 概率、答案语义置信和事实可信是不同对象，应明确所用信号。
- 拒答减少错误也降低覆盖，需同时评估未答率、错答成本与正确率。
- 证据检索、自检和多样采样可提供信号，但其有效性需要实测。

### 易错点

- 模型声称不确定也可能是模板行为，不代表具备准确的自我知识。

### 面试官可能追问

- 对高风险和低风险问题为什么应使用不同阈值？

</details>

**技术依据**

- [APP-S152 · Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="eva-008"></a>
## EVA-008 · 安全评测如何兼顾越狱成功率与过度拒答？

**L2 · 编辑补充题** · 标签：safety / jailbreak / over-refusal

**30 秒回答**

安全评测既要检查模型在恶意请求和攻击改写下是否违规，也要检查正常相似问题是否被误拒。按风险类别、语言、轮次与工具环境分层，固定攻击预算和判定标准，报告成功率、误拒率与不确定样本，避免只优化一句拒答模板。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 恶意与合法边界相近的成对样本能检测语义判断而非关键词拦截。
- 攻击成功必须核验实际内容/行动，不仅统计是否出现某些敏感词。
- 单轮文本测试不能覆盖多轮诱导、图像输入或工具副作用。

### 易错点

- 零次成功只是当前测试集与预算结果，不能证明不存在漏洞。

### 面试官可能追问

- 拒绝表达更礼貌了，但泄露率没下降，如何判断改进？

</details>

**技术依据**

- [APP-S148 · HarmBench](https://arxiv.org/abs/2402.04249)
- [APP-S165 · XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://arxiv.org/abs/2308.01263)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="eva-009"></a>
## EVA-009 · 如何构建红队测试而不只刷固定攻击集？

**L3 · 编辑补充题** · 标签：red teaming / coverage

**30 秒回答**

红队先定义资产与威胁模型，再设计人工、规则和模型生成的多样攻击，记录具体成功条件与复现轨迹。持续把新的失败类别加入回归，同时留出未见攻击验证，防止系统只适应公开模板；攻击与防御模型都需要独立评估。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 覆盖直接请求、间接资料、跨轮组合和工具权限边界。
- 模型可生成候选攻击，但自动裁判需人工抽检以避免伪成功。
- 同一目标下比较攻击预算、成功率、覆盖度与可复现性。

### 易错点

- 红队发现问题不等于全面量化风险，未测领域要明确列出。

### 面试官可能追问

- 如何避免同一失败模板大量重复导致覆盖率虚高？

</details>

**技术依据**

- [APP-S149 · Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="eva-010"></a>
## EVA-010 · 训练数据泄露与 RAG 越权泄露如何区分？

**L2 · 编辑补充题** · 标签：privacy / memorization

**30 秒回答**

训练数据泄露来自参数化记忆被查询诱导重现；RAG 越权来自检索、缓存或工具授权把不该访问的资料交给模型。前者需要数据最小化、去重和泄露测试等防护，后者必须在访问层隔离，不能用同一种拒答 prompt 解决。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 检查信息来源：未接入该资料仍复现，可能是记忆或其他上游输入泄漏。
- 测试时使用可控 canary 与访问矩阵，避免把真实敏感数据继续扩散。
- 删除索引不意味着模型参数内记忆删除；两类数据生命周期分开追踪。

### 易错点

- 输出里有个人信息不自动证明来自训练语料，须排查输入和工具路径。

### 面试官可能追问

- 企业要求撤回某用户数据时，哪些层需要分别处理？

</details>

**技术依据**

- [APP-S145 · Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="eva-011"></a>
## EVA-011 · MLLM 评测如何区分感知、OCR 与推理错误？

**L2 · 编辑补充题** · 标签：multimodal eval / MMMU

**30 秒回答**

多模态模型答错可能因为没看清文字、定位错对象、误读图表，也可能因为跨模态关系或推理失败。应按图像类型和能力拆分样本，并通过原图、裁剪、人工文本转写等消融定位瓶颈，同时固定分辨率、图片数与答案解析规则。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- OCR、计数、空间关系、图表、专业多学科推理应分别汇报。
- 文本-only 与真实图像输入对比能检查模型是否只凭语言先验猜答案。
- 替换或打乱图像后答案不变，提示视觉依赖不足，仍需排除多答案歧义。

### 易错点

- 一个综合榜单不能代表视频、定位或高分辨率读图的全部能力。

### 面试官可能追问

- 原图错、人工 OCR 文本对时，应优先改哪一环？

</details>

**技术依据**

- [APP-S147 · MMMU](https://arxiv.org/abs/2311.16502)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="eva-012"></a>
## EVA-012 · 怎样评估 prompt 扰动、多语言和分布外鲁棒性？

**L2 · 编辑补充题** · 标签：robustness / OOD

**30 秒回答**

准确率应在语言、领域、长度和输入表达变化下分别测量。构造语义等价改写、拼写噪声及格式变化，也保留真正更难或语义改变的任务。报告各组差异与成本，并检查 parser、检索与安全策略是否引入非模型退化。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 等价扰动可测一致性，事实改变的对照测模型是否随证据正确更新。
- 同一任务采用标准化 prompt 与解码，再测试多模板敏感性。
- 总体均值不能掩盖少数语言、长尾实体或长上下文的显著退化。

### 易错点

- 语义变了的样本不是纯鲁棒性测试，需人工确认变换标签。

### 面试官可能追问

- 翻译后性能下降如何区分知识缺失、检索和 tokenizer 问题？

</details>

**技术依据**

- [APP-S142 · Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="eva-013"></a>
## EVA-013 · 如何验证 prompt 优化确实有效？

**L2 · 编辑补充题** · 标签：prompt eval / instruction following

**30 秒回答**

把 prompt 优化视为有验证集的实验：定义可观察目标，固定模型和解码，每次改变明确因素，使用留出样本比较。格式、长度和关键词要求可规则检测，事实或语义质量需要其他验证；一个演示样例变好不能证明泛化。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 可执行约束如 JSON 合法性、指定字段和字数需用稳定 parser 检查。
- 任务级成功与单项指令遵循分开统计，多个约束全部满足才算整题通过。
- 版本化模板与 few-shot 样例，避免样例与测试内容重叠。

### 易错点

- prompt 中加入更多约束可能互相冲突，temperature 降低不必缩短输出。

### 面试官可能追问

- 同 prompt 训练集涨分、留出集退化时如何处理？

</details>

**技术依据**

- [APP-S153 · Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="eva-014"></a>
## EVA-014 · 代码生成的 pass@k 是什么，如何避免错误估计？

**L3 · 编辑补充题** · 标签：pass@k / code eval

**30 秒回答**

pass@k 衡量每题采样 k 个候选至少一个通过测试的概率，并对任务取平均。生成 n≥k 个候选、其中 c 个通过时，可用组合数估计，不能直接把经验 pass@1 代入 1−(1−p)^k 当无偏估计。比较时还需固定采样和测试预算。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 通过条件是可执行功能测试，不能用 BLEU 衡量等价代码正确性。
- 若 n−c<k，估计为一；实际计算用稳定乘积避免大组合数溢出。
- 测试覆盖、超时、沙箱和依赖版本会影响得分，需与生成条件同时记录。
- 无偏估计的标准解释以每题固定生成分布的独立采样为前提；自适应搜索、去重或依赖先前测试结果生成的候选需另定义评测。

### 公式

```text
\widehat{\mathrm{pass@}k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},\qquad n\ge k
```

### 易错点

- oracle pass@k 不等于用户拿到正确答案的概率，部署仍需要选择/验证候选。

### 面试官可能追问

- 为什么提高温度可能提升 pass@100 却降低 pass@1？

</details>

**技术依据**

- [APP-S146 · Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="eva-015"></a>
## EVA-015 · 线上如何监控、验收并回归 LLM/Agent 系统？

**L2 · 编辑补充题** · 标签：observability / regression

**30 秒回答**

离线通过后仍要观测线上任务成功、拒答、工具错误、事实反馈、时延和成本。保存能复现的版本与 trace，按风险和流量分层抽检，把新 bad case 加入回归并保留独立验收集。发布前后用同口径比较，设置退化回滚条件。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- trace 记录请求、检索、工具、结果与版本，敏感字段做最小化记录和访问控制。
- 重复 trial 检查随机性，故障注入验证限流、断线和恢复。
- 回归同时检查能力、合规边界与效率，避免一个平均分掩盖新问题。

### 易错点

- 把线上点击率当答案正确率会受曝光、用户群与界面变化混杂。

### 面试官可能追问

- 线上成本骤增但准确率没变，应怎样定位？

</details>

**技术依据**

- [APP-S139 · Demystifying evals for AI agents — Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。
