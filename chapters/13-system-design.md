# 系统设计与线上故障

[返回仓库首页](../README.md) · [来源目录](../SOURCES.md) · [阅读全部题目](../exports/questions.jsonl)

L1 基础 · L2 进阶 · L3 深入。难度为编辑判断；社区线索不等于雇主确认真题。所有答案为独立整理。

## 题目导航

- [SYS-001 · 设计企业文档问答系统，先明确哪些约束？](#sys-001)
- [SYS-002 · LLM 服务 P95 延迟突然升高，如何定位？](#sys-002)
- [SYS-003 · 如何测吞吐、并发、TTFT、TPOT，并避免错误比较？](#sys-003)
- [SYS-004 · Agent 上下文溢出，怎样压缩而保持任务连续性？](#sys-004)
- [SYS-005 · 工具调用失败后如何重试，怎样避免重复副作用？](#sys-005)
- [SYS-006 · 设计 Agent 可观测性：日志里应该记录什么？](#sys-006)
- [SYS-007 · 多租户 RAG 如何保证权限隔离？](#sys-007)
- [SYS-008 · 如何不停服更新知识库与 Embedding 模型？](#sys-008)
- [SYS-009 · RAG 回答错了，怎样区分检索错误与生成错误？](#sys-009)
- [SYS-010 · 高并发 LLM 服务如何限流与降级？](#sys-010)
- [SYS-011 · Prefix Cache 能提高多少性能，怎样设计缓存键？](#sys-011)
- [SYS-012 · 微调上线后质量衰减，如何诊断数据漂移？](#sys-012)
- [SYS-013 · 多模态服务图片/PDF 输入成本失控，怎么优化？](#sys-013)
- [SYS-014 · Prompt 优化“修好一类坏了另一类”，怎样控制回归？](#sys-014)
- [SYS-015 · 线上模型事故如何止损、回滚与复盘？](#sys-015)

<a id="sys-001"></a>
## SYS-001 · 设计企业文档问答系统，先明确哪些约束？

**L2 · 编辑补充题** · 标签：RAG / 系统设计 / SLO

**30 秒回答**

先定义用户、文档权限、更新时效、质量与延迟目标，再设计解析入库、召回重排、上下文生成与证据校验链路。离线质量与线上 SLO 都要可测；每个模块的复杂度应由实际瓶颈和失败案例驱动。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 入口鉴权→权限内召回→重排→带版本和引用的上下文→生成/拒答。
- 增量索引与文档删除要传播到缓存和检索存储，保留版本便于回放。
- 先建小规模基线，测召回、答案支持度、P95 延迟及成本，再决定图检索或多 Agent。

### 易错点

- 还没确认目标与权限就先画很复杂的框架图。

### 面试官可能追问

- 用户修改权限后旧缓存怎样失效？

</details>

**技术依据**

- [ENG-P24 · Azure AI Search Security Filter Pattern](https://learn.microsoft.com/en-us/azure/search/search-security-trimming-for-azure-search)
- [ENG-P25 · Google SRE Implementing SLOs](https://sre.google/workbook/implementing-slos/)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="sys-002"></a>
## SYS-002 · LLM 服务 P95 延迟突然升高，如何定位？

**L2 · 编辑补充题** · 标签：P95 / TTFT / TPOT / 排队

**30 秒回答**

将端到端延迟拆成排队、检索、prefill、decode 和外部工具，用 trace 关联请求与模型指标。分输入长度、输出长度、并发、租户及版本比较；先定位哪个分布发生变化，再选择限流、调度或算子优化。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- TTFT 包括首 token 前的等待；TPOT 衡量输出 token 间时间，不能混为一个 tokens/s。
- 检查到达率、活跃请求、KV 用量、缓存命中率、长请求比例和重试放大。
- 用匹配长度/并发的回放确认根因，避免拿低负载微基准解释线上尾延迟。

### 易错点

- 只看平均延迟或 GPU 利用率，忽略队列和长尾。

### 面试官可能追问

- 为何吞吐上升时单请求体验可能变差？

</details>

**技术依据**

- [ENG-P15 · vLLM Metrics](https://docs.vllm.ai/en/latest/design/metrics/)
- [ENG-P16 · Google SRE Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [ENG-P17 · OpenTelemetry Traces](https://opentelemetry.io/docs/concepts/signals/traces/)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="sys-003"></a>
## SYS-003 · 如何测吞吐、并发、TTFT、TPOT，并避免错误比较？

**L2 · 编辑补充题** · 标签：benchmark / SLO / 分位数

**30 秒回答**

固定模型、硬件、输入输出长度分布与采样设置，记录到达率、完成率、首 token、每 token 时间和端到端延迟。对服务最好做到达率驱动的测试并观察饱和点；有效吞吐应同时满足既定质量和延迟目标。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- requests/s 与 output tokens/s 都有用，但输出长度不同会改变解释。
- 说明 warmup、失败/取消请求是否计数、客户端是否成为瓶颈。
- P95/P99 应从合适的样本或可聚合直方图估计，不能平均各实例 P95。

### 易错点

- 只报峰值 tokens/s，没有输入分布和 SLO。

### 面试官可能追问

- 如何防止闭环负载测试低估过载时排队？

</details>

**技术依据**

- [ENG-P18 · Prometheus Histograms and Summaries](https://prometheus.io/docs/practices/histograms/)
- [ENG-P25 · Google SRE Implementing SLOs](https://sre.google/workbook/implementing-slos/)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="sys-004"></a>
## SYS-004 · Agent 上下文溢出，怎样压缩而保持任务连续性？

**L2 · 社区题目线索** · 标签：context / memory / compression

**30 秒回答**

保留目标、已确认事实、未完成事项、决策理由和关键工具结果，将长文与完整日志外置为可检索引用。压缩后检查证据和状态是否仍可恢复；不同任务适合摘要、选择性保留或结构化 state，不能只按 token 数截断。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 区分不可丢的状态字段与可重新读取的材料；保留来源及版本。
- 工具返回做边界明确的裁剪和分页，避免把全量数据库塞入 prompt。
- 用长任务回放对比成功率、遗漏率、延迟和 token 开销；摘要也可能幻觉。

### 易错点

- 从最早消息开始机械删除，把用户约束和失败经验一起丢掉。

### 面试官可能追问

- 哪些事实必须以结构化字段保存而不能只做摘要？

</details>

**技术依据**

- [ENG-P25 · Google SRE Implementing SLOs](https://sre.google/workbook/implementing-slos/)
- [ENG-P26 · Google SRE Effective Troubleshooting](https://sre.google/sre-book/effective-troubleshooting/)
- [APP-S138 · Effective context engineering — Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

**题目出处线索**

- [ENG-C03 · 小红书 AI Agent 开发岗位面试调研报告](https://holynova.github.io/ai-agent-interview-report/xiaohongshu-ai-agent-interview-report.html) · `secondary_report`：二手小红书调研出现上下文/记忆和长任务主题；未核验原帖，不使用报告答案。

<a id="sys-005"></a>
## SYS-005 · 工具调用失败后如何重试，怎样避免重复副作用？

**L2 · 社区题目线索** · 标签：idempotency / retry / timeout

**30 秒回答**

先区分可重试的瞬时失败、不可重试的业务错误和结果未知的超时。写操作用幂等键及持久化结果记录，查询已有操作状态后再决定重试；设全局 deadline、有限次数与退避，避免多层重试成倍放大负载。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 幂等键应代表同一业务意图，服务端检查重复键与参数一致性。
- timeout 不代表服务端没有执行；支付或发消息必须能查状态。
- 只读重试也消耗资源，应用预算、并发上限和熔断。

### 易错点

- 给所有异常无条件 retry，导致重复提交或重试风暴。

### 面试官可能追问

- 如果客户端取消但服务端已提交，怎样反馈最终状态？

</details>

**技术依据**

- [ENG-P28 · AWS Making retries safe with idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)
- [ENG-P23 · Google SRE Handling Overload](https://sre.google/sre-book/handling-overload/)

**题目出处线索**

- [ENG-C06 · 腾讯/百度 Agent 面经总结（公开搜索摘录）](https://www.nowcoder.com/discuss/878600528970735616) · `search_snippet`：转载搜索摘录含 Agent 中断、失败和重试安全主题。

<a id="sys-006"></a>
## SYS-006 · 设计 Agent 可观测性：日志里应该记录什么？

**L2 · 编辑补充题** · 标签：trace / replay / privacy

**30 秒回答**

每次运行关联 run_id/trace_id，记录模型版本、prompt 版本、检索结果版本、工具参数摘要、耗时、错误与终止原因。将规划、模型、工具作为 span，支持按失败类型回放；敏感内容脱敏与采样，不能以可观测性为由全量泄露。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 同一业务请求跨服务传播 trace context；异步子任务保留父子关系。
- 同时记录质量指标和资源指标，避免只看耗时而不知道任务是否完成。
- 回放时固定工具快照或 mock，记录外部状态变化，避免重放写操作。

### 易错点

- 只有最后一个错误堆栈，没有中间状态和版本信息。

### 面试官可能追问

- 如何定位模型生成错误与工具返回错误？

</details>

**技术依据**

- [ENG-P17 · OpenTelemetry Traces](https://opentelemetry.io/docs/concepts/signals/traces/)
- [ENG-P26 · Google SRE Effective Troubleshooting](https://sre.google/sre-book/effective-troubleshooting/)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="sys-007"></a>
## SYS-007 · 多租户 RAG 如何保证权限隔离？

**L3 · 编辑补充题** · 标签：ACL / tenant / cache

**30 秒回答**

用户身份与文档 ACL 必须在检索端约束候选集合，必要时在返回前再检查。向量、关键词、重排、引用和缓存都带租户与权限范围；权限变更应触发失效。仅靠 prompt 要求模型保密不能替代服务端授权。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 过滤属性应由可信服务端生成，不能直接接受用户传入的 tenant_id。
- 缓存键包含权限范围或版本，跨租户共享 prefix/答案需评估泄漏渠道。
- 测试撤权、文档删除、混合检索分支和多轮对话中的旧引用。

### 易错点

- 先全库检索再只隐藏链接，正文仍可能进模型上下文。

### 面试官可能追问

- 权限过滤使召回下降时怎样重新测 Recall@K？

</details>

**技术依据**

- [ENG-P24 · Azure AI Search Security Filter Pattern](https://learn.microsoft.com/en-us/azure/search/search-security-trimming-for-azure-search)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="sys-008"></a>
## SYS-008 · 如何不停服更新知识库与 Embedding 模型？

**L3 · 社区题目线索** · 标签：index version / blue-green / migration

**30 秒回答**

把文档版本、embedding 模型与索引 schema 绑定，后台构建新索引并校验质量、覆盖率与权限，双读或灰度后原子切换指向。新旧 embedding 空间通常不能直接混比；保留旧版本供回滚，并让增量更新和删除都能追上。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 切换前校验文档数、失败队列、检索评测和数据截止水位。
- 双写期间要处理先后顺序和幂等；文档 tombstone 防止旧事件复活删除数据。
- 灰度按租户或请求分桶，缓存键带索引版本。

### 易错点

- 只替换 query encoder 却继续检索旧 encoder 生成的向量。

### 面试官可能追问

- 全量重建期间新增文档怎样补到新索引？

</details>

**技术依据**

- [ENG-P19 · Kubernetes Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [ENG-P27 · Google SRE Canarying Releases](https://sre.google/workbook/canarying-releases/)

**题目出处线索**

- [ENG-C06 · 腾讯/百度 Agent 面经总结（公开搜索摘录）](https://www.nowcoder.com/discuss/878600528970735616) · `search_snippet`：公开搜索摘录出现 RAG 知识库不停服更新提问。

<a id="sys-009"></a>
## SYS-009 · RAG 回答错了，怎样区分检索错误与生成错误？

**L2 · 社区题目线索** · 标签：ablation / oracle / attribution

**30 秒回答**

保留题目、真值、召回证据与回答，先检查正确证据是否存在，再用人工确定的 oracle 证据替换检索结果。如果 oracle 下仍错，重点查生成、模板或推理；若变对则查召回和重排，之后做模块消融验证归因。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 分开统计未召回、证据冲突、上下文裁剪、引用错配和生成不忠实。
- oracle 实验也要保持相同上下文长度和提示规则，减少干扰因素。
- 检索 Recall@K 与最终答案正确率都测，不能用单一端到端分数定位全部问题。

### 易错点

- 把所有错误都叫幻觉，或只增大 K 不查噪声。

### 面试官可能追问

- 多跳问题怎样定义“足够证据”？

</details>

**技术依据**

- [ENG-P22 · HELM](https://arxiv.org/abs/2211.09110)
- [ENG-P26 · Google SRE Effective Troubleshooting](https://sre.google/sre-book/effective-troubleshooting/)

**题目出处线索**

- [ENG-C07 · 百度大模型算法岗面经-05（搜索可见）](https://www.nowcoder.com/discuss/927018708290015232) · `search_snippet`：搜索摘录有 RAG 检索错/生成错与归因实验主题。

<a id="sys-010"></a>
## SYS-010 · 高并发 LLM 服务如何限流与降级？

**L3 · 编辑补充题** · 标签：admission control / queue / load shedding

**30 秒回答**

按输入输出 token 预算与租户配额准入，限制排队长度和 deadline，饱和时主动拒绝或降级。优先保护已有请求和重要任务；降级可以减少候选、切换已验证的小模型或返回异步任务，但必须继续满足权限与质量底线。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- QPS 无法单独刻画长短请求的成本，应估计 token 与缓存资源。
- 避免队列无限增长；明确 429/503、重试建议与取消传播。
- 对降级路径单独做质量评测；限制重试放大并监控 shedding 比例。

### 易错点

- 过载时无限排队，最终所有请求超时。

### 面试官可能追问

- 如何避免短请求被长 prefill 饥饿？

</details>

**技术依据**

- [ENG-P23 · Google SRE Handling Overload](https://sre.google/sre-book/handling-overload/)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="sys-011"></a>
## SYS-011 · Prefix Cache 能提高多少性能，怎样设计缓存键？

**L2 · 编辑补充题** · 标签：prefix caching / 版本 / hit rate

**30 秒回答**

Prefix Cache 主要复用相同前缀的 KV 来减少 prefill 计算，收益依赖可复用前缀长度、命中率和调度。缓存键必须覆盖 token 序列、模型与 adapter 配置，必要时隔离租户；它不能直接省去后续 decode。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 短 prompt 或重复率低时收益可能很小；统计命中 token 数比请求命中率更有解释力。
- 视觉输入还应纳入内容标识和预处理版本，不能只用文件名。
- 预算需同时考虑缓存保留与活跃请求的 KV 竞争。

### 易错点

- 把 prefix cache 当答案 cache，或忽视 adapter/图像变化导致错误复用。

### 面试官可能追问

- 系统 prompt 很长但用户输入不同，哪些块能共享？

</details>

**技术依据**

- [ENG-P15 · vLLM Metrics](https://docs.vllm.ai/en/latest/design/metrics/)
- [ENG-P30 · vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- [ENG-P32 · vLLM Prefix Caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="sys-012"></a>
## SYS-012 · 微调上线后质量衰减，如何诊断数据漂移？

**L3 · 社区题目线索** · 标签：drift / monitoring / SFT

**30 秒回答**

先检查版本、模板和输入处理是否一致，再比较线上与训练/评测分布，按领域、语言、长度、用户类型切片评测。采样标注新错误并回放到旧模型作对照，区分分布变化、接口退化与模型遗忘，按证据决定回滚或更新数据。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 监控输入统计、拒答率、工具失败率和人评抽检，单看训练 loss 无法诊断。
- 确保评测时间切分与线上场景相符，防止用泄漏数据证明恢复。
- 对新数据微调做旧能力回归与灰度，不断反馈而非一次上线后停止。

### 易错点

- 质量变差就直接增大训练步数，不先排查服务配置。

### 面试官可能追问

- 无实时标签时可用哪些代理信号，局限是什么？

</details>

**技术依据**

- [ENG-P20 · Google Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [ENG-P27 · Google SRE Canarying Releases](https://sre.google/workbook/canarying-releases/)

**题目出处线索**

- [ENG-C05 · 你的大模型项目，能扛住面试官的几连问？](https://zhuanlan.zhihu.com/p/2039673415901106968) · `search_snippet`：知乎搜索可见“微调上线后效果衰减”提问；正文未读。

<a id="sys-013"></a>
## SYS-013 · 多模态服务图片/PDF 输入成本失控，怎么优化？

**L3 · 编辑补充题** · 标签：image tokens / OCR / budget

**30 秒回答**

先测解析、OCR、视觉编码和 LLM prefill 各段成本，按任务选择页面或区域，并限制分辨率、页数与视觉 token 预算。OCR 文本、图片区域与布局可组合使用；所有压缩要在细字、表格和跨页任务上测质量损失。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 预处理/视觉特征缓存带文件内容 hash、模型和裁剪版本。
- 先检索相关页面再送 VLM，有利于长文档；需防止检索漏掉跨页证据。
- 统计每成功任务成本，不能只按图片数量估算不同动态分辨率模型。

### 易错点

- 一律缩成低分辨率，导致 OCR 和细粒度定位失效。

### 面试官可能追问

- 如果主要问题是视觉 encoder 耗时而非 decode，量化 LLM 是否能解决？

</details>

**技术依据**

- [ENG-P16 · Google SRE Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [ENG-P25 · Google SRE Implementing SLOs](https://sre.google/workbook/implementing-slos/)
- [ENG-P31 · Transformers Qwen2.5-VL documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="sys-014"></a>
## SYS-014 · Prompt 优化“修好一类坏了另一类”，怎样控制回归？

**L2 · 社区题目线索** · 标签：prompt regression / versioning / test set

**30 秒回答**

把 prompt 作为版本化配置，维护按能力、业务与历史错误分层的固定回归集和新鲜验证集，比较每个切片而非只看均值。为改动定义可接受回归与收益，盲评和灰度确认；少量例子改好不代表全局改进。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 记录完整模板、模型、工具 schema 与采样设置，避免多个因素同时变化。
- 先做失败归因，再决定修改通用指令还是局部路由/示例。
- 锁定测试集，避免不断调到对固定题库过拟合。

### 易错点

- 每次只拿新坏例验证，不跑历史回归。

### 面试官可能追问

- 如何判断问题应该改 prompt、数据还是模型？

</details>

**技术依据**

- [ENG-P27 · Google SRE Canarying Releases](https://sre.google/workbook/canarying-releases/)
- [ENG-P22 · HELM](https://arxiv.org/abs/2211.09110)

**题目出处线索**

- [ENG-C03 · 小红书 AI Agent 开发岗位面试调研报告](https://holynova.github.io/ai-agent-interview-report/xiaohongshu-ai-agent-interview-report.html) · `secondary_report`：调研报告对应字节 Agent 二面条目含 prompt 修好一类坏另一类；原帖未核验。

<a id="sys-015"></a>
## SYS-015 · 线上模型事故如何止损、回滚与复盘？

**L3 · 编辑补充题** · 标签：incident / rollback / postmortem

**30 秒回答**

先按用户影响止损，可切回已验证版本、限制问题路径或暂停写操作，再保存日志与配置快照。用时间线、切片和回放定位触发条件，提出可验证的防复发措施。回滚范围不仅是权重，还包括模板、索引、adapter 与工具 schema。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 发布前定义回滚触发指标与兼容性，确保旧版本仍能读取状态。
- 复盘区分触发因素、根因与放大因素，避免只找个人责任。
- 把事故转成回归样例、监控信号和演练，验证措施能捕获同类问题。

### 易错点

- 只回滚模型权重，漏掉同时更新的 tokenizer 或索引。

### 面试官可能追问

- 新旧数据 schema 不兼容时怎样回滚？

</details>

**技术依据**

- [ENG-P19 · Kubernetes Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [ENG-P26 · Google SRE Effective Troubleshooting](https://sre.google/sre-book/effective-troubleshooting/)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。
