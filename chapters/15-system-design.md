# 系统设计、性能与可靠性

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 目录

- [系统设计与资源预算](#topic-1)
  - [SYS-001 · 设计企业文档问答系统，先明确哪些约束？](#sys-001)
  - [SYS-004 · Agent 上下文溢出，怎样压缩而保持任务连续性？](#sys-004)
  - [SYS-013 · 多模态服务图片/PDF 输入成本失控，怎么优化？](#sys-013)
- [性能、缓存与并发](#topic-2)
  - [SYS-002 · LLM 服务 P95 延迟突然升高，如何定位？](#sys-002)
  - [SYS-003 · 如何测吞吐、并发、TTFT、TPOT，并避免错误比较？](#sys-003)
  - [SYS-010 · 高并发 LLM 服务如何限流与降级？](#sys-010)
  - [SYS-011 · Prefix Cache 能提高多少性能，怎样设计缓存键？](#sys-011)
  - [SYS-016 · 大规模 PDF 解析怎样组合多线程、多进程与 GPU 批处理？](#sys-016)
  - [SYS-018 · FastAPI 与 vLLM 怎样分工，如何避免 worker 数量导致模型重复加载？](#sys-018)
- [质量诊断与版本治理](#topic-3)
  - [SYS-008 · 如何不停服更新知识库与 Embedding 模型？](#sys-008)
  - [SYS-009 · RAG 回答错了，怎样区分检索错误与生成错误？](#sys-009)
  - [SYS-012 · 微调上线后质量衰减，如何诊断数据漂移？](#sys-012)
  - [SYS-014 · Prompt 优化“修好一类坏了另一类”，怎样控制回归？](#sys-014)
- [权限、容错与可观测性](#topic-4)
  - [SYS-005 · 工具调用失败后如何重试，怎样避免重复副作用？](#sys-005)
  - [SYS-006 · 设计 Agent 可观测性：日志里应该记录什么？](#sys-006)
  - [SYS-007 · 多租户 RAG 如何保证权限隔离？](#sys-007)
  - [SYS-015 · 线上模型事故如何止损、回滚与复盘？](#sys-015)
  - [SYS-017 · 个人多模态记忆怎样存储、检索、更新并处理相互矛盾的信息？](#sys-017)

<a id="topic-1"></a>
## 系统设计与资源预算

<a id="sys-001"></a>
### SYS-001 · 设计企业文档问答系统，先明确哪些约束？

**L2**

#### 答案

先定义用户、文档权限、更新时效、质量和延迟目标，再设计“入口鉴权 → 权限内召回 → 重排 → 带版本和引用的上下文 → 生成或拒答”的链路。

增量更新与文档删除应同步到检索存储和缓存，保留版本便于回放。先建立小规模基线，测召回、答案支持度、P95 延迟与成本，再依据瓶颈和失败案例决定是否增加图检索或多 Agent；离线质量和线上 SLO 都需要可测。

#### 易错点

- 还没确认目标与权限就先画很复杂的框架图。

#### 追问

- 用户修改权限后旧缓存怎样失效？

<a id="sys-004"></a>
### SYS-004 · Agent 上下文溢出，怎样压缩而保持任务连续性？

**L2**

#### 答案

保留目标、已确认事实、未完成事项、决策理由和关键工具结果，将长文与完整日志外置为可检索引用。区分不可丢的任务状态和可重新读取的材料，并保留来源及版本。

按任务选择摘要、选择性保留或结构化 state；工具返回使用边界明确的裁剪和分页，避免全量结果挤占上下文。压缩后检查证据与状态是否可恢复，并通过长任务回放比较成功率、遗漏率、延迟和 token 开销；摘要本身也可能产生幻觉。

#### 易错点

- 从最早消息开始机械删除，把用户约束和失败经验一起丢掉。

#### 追问

- 哪些事实必须以结构化字段保存而不能只做摘要？

<a id="sys-013"></a>
### SYS-013 · 多模态服务图片/PDF 输入成本失控，怎么优化？

**L3**

#### 答案

先测解析、OCR、视觉编码和 LLM prefill 各环节成本，再按任务选择页面或区域，限制分辨率、页数和视觉 token 预算。可以组合 OCR 文本、图像区域与布局，但压缩方案需在细字、表格及跨页任务上验证质量损失。

预处理与视觉特征缓存携带文件内容 hash、模型和裁剪版本。长文档可以先检索相关页面再交给 VLM，同时检查是否漏掉跨页证据。比较每个成功任务的成本，图片数量不足以代表动态分辨率模型的实际消耗。

#### 易错点

- 一律缩成低分辨率，导致 OCR 和细粒度定位失效。

#### 追问

- 如果主要问题是视觉 encoder 耗时而非 decode，量化 LLM 是否能解决？

<a id="topic-2"></a>
## 性能、缓存与并发

<a id="sys-002"></a>
### SYS-002 · LLM 服务 P95 延迟突然升高，如何定位？

**L2**

#### 答案

把端到端延迟拆成排队、检索、prefill、decode 和外部工具，用 trace 关联请求与模型指标，再按输入长度、输出长度、并发、租户和版本切片比较。

TTFT 包含首 token 前的等待，TPOT 衡量输出 token 间时间，不能合并成一个 tokens/s。检查到达率、活跃请求、KV 用量、缓存命中率、长请求比例和重试放大，找到变化环节后再选限流、调度或算子优化。用匹配长度与并发的回放确认根因，低负载微基准无法充分解释线上尾延迟。

#### 易错点

- 只看平均延迟或 GPU 利用率，忽略队列和长尾。

#### 追问

- 为何吞吐上升时单请求体验可能变差？

<a id="sys-003"></a>
### SYS-003 · 如何测吞吐、并发、TTFT、TPOT，并避免错误比较？

**L2**

#### 答案

固定模型、硬件、输入输出长度分布和采样设置，记录到达率、完成率、TTFT、TPOT 与端到端延迟。服务压测宜由到达率驱动，观察饱和点；有效吞吐需要同时满足质量和延迟目标。

requests/s 与 output tokens/s 各有价值，输出长度分布不同会改变比较结果。说明 warmup、失败或取消请求的计数方式，确认客户端没有成为瓶颈。P95/P99 应从请求样本或可聚合直方图估计，不能直接平均各实例的 P95。

#### 易错点

- 只报峰值 tokens/s，没有输入分布和 SLO。

#### 追问

- 如何防止闭环负载测试低估过载时排队？

<a id="sys-010"></a>
### SYS-010 · 高并发 LLM 服务如何限流与降级？

**L3**

#### 答案

按输入输出 token 预算和租户配额准入，限制队列长度与 deadline，饱和时主动拒绝或降级，优先保护已有请求和重要任务。QPS 不能单独代表长短请求成本，还需估计 token 和 KV 资源。

降级可减少候选、切换已验证的小模型或返回异步任务，但要单独评测质量并继续满足权限约束。明确 429/503、重试建议和取消传播，限制重试放大，并监控被主动丢弃的请求比例。

#### 易错点

- 过载时无限排队，最终所有请求超时。

#### 追问

- 如何避免短请求被长 prefill 饥饿？

<a id="sys-011"></a>
### SYS-011 · Prefix Cache 能提高多少性能，怎样设计缓存键？

**L2**

#### 答案

Prefix Cache 复用相同前缀的 KV，主要减少 prefill 计算，收益取决于可复用前缀长度、命中率和调度；后续 decode 仍需执行。短 prompt 或重复率低时收益可能较小，命中 token 数通常比请求命中率更能解释收益。

缓存键覆盖 token 序列、模型与 adapter 配置，必要时隔离租户。视觉输入还要包含内容标识和预处理版本，不能只按文件名缓存。预算需考虑保留前缀缓存与活跃请求 KV 的资源竞争。

#### 易错点

- 把 prefix cache 当答案 cache，或忽视 adapter/图像变化导致错误复用。

#### 追问

- 系统 prompt 很长但用户输入不同，哪些块能共享？

<a id="sys-016"></a>
### SYS-016 · 大规模 PDF 解析怎样组合多线程、多进程与 GPU 批处理？

**L3**

#### 答案

先把流程分成下载/读取、PDF 解码与页面渲染、OCR/版面识别、清洗切块、embedding 和入库，分别测 CPU、I/O、GPU 与队列等待。一般用线程或异步重叠网络和文件 I/O，用进程并行 CPU 密集的 Python 处理，把 GPU OCR/embedding 集中到受控批处理服务。默认带 GIL 的 CPython 不能让纯 Python CPU 任务仅靠线程实现多核；释放 GIL 的库或 free-threaded 构建须另看条件。

库的线程安全优先于这条经验。PyMuPDF 官方不支持多线程调用，应由独立进程各自打开文档，传文件路径和页范围，不跨进程传 Document 对象或整页大图。结果携带文档 hash、页号、解析版本和状态，便于排序、重试和增量去重。

用有界队列施加背压，避免解析快而 OCR 慢时把内存堆满。限制进程内 BLAS/OCR 线程，避免进程数乘线程数过度争抢；控制 GPU 模型实例数量，批量推理与失败页隔离并行设计。优化目标是每秒成功处理页数及峰值资源，不是启动最多 worker。

```math
\mathrm{throughput}_{\mathrm{pipeline}}\lesssim\min_i\mathrm{capacity}_i
```

#### 易错点

- PyMuPDF 的线程安全限制不能被“C 扩展可能释放 GIL”这个经验覆盖。
- 无限队列和每个 worker 都加载一份 GPU 模型会放大内存与显存峰值。

#### 追问

- 怎样判断该按文档分片还是按页范围分片？
- Windows spawn 模式下为什么入口保护、序列化和 worker 初始化重要？

<a id="sys-018"></a>
### SYS-018 · FastAPI 与 vLLM 怎样分工，如何避免 worker 数量导致模型重复加载？

**L2**

#### 答案

FastAPI 负责应用 HTTP 接口、鉴权、请求校验、业务路由和流式转发；vLLM engine 负责模型执行、KV 分配、批处理与生成调度。vLLM 已提供兼容 API 的 serving 入口，常见做法是让轻量应用网关异步调用独立模型服务，而不是在每个 Web worker 的请求函数里同步加载模型。

多个 ASGI worker 是多个进程，各自初始化模型会复制权重、上下文与缓存，可能一启动就 OOM。应明确模型进程/实例数、GPU 分配和并行组；网关扩容与模型副本扩容分开。流式接口及时处理客户端断连与取消，限制在途请求、输入/输出 token 和超时，避免已无人接收的生成继续占用容量。

做负载测试时把网关排队、网络、prefill 和 decode 分开记录，并验证聊天模板、stop、采样参数及错误码能正确透传。async 允许重叠等待，不会让单次 GPU 前向自行变快；吞吐最终受模型调度和资源约束。

#### 易错点

- FastAPI 的 async 接口不自动解决同步阻塞模型调用或 GPU 显存复制。
- 盲目增加 Web worker 数量，却没有确定模型实例与 GPU 的关系。

#### 追问

- 客户端中途断连时，怎样确认底层生成请求也被取消？
- 怎样分别扩容无状态网关和持有 KV 状态的模型服务？

<a id="topic-3"></a>
## 质量诊断与版本治理

<a id="sys-008"></a>
### SYS-008 · 如何不停服更新知识库与 Embedding 模型？

**L3**

#### 答案

把文档版本、embedding 模型和索引 schema 绑定，后台构建新索引，验证覆盖率、权限与检索质量，双读或灰度后原子切换指向，并保留旧版供回滚。新旧 embedding 空间通常不能直接混合比较。

切换前检查文档数、失败队列和数据截止水位，让增量更新与删除都能追上。双写需要处理顺序与幂等，使用 tombstone 防止旧事件复活已删除文档；灰度按租户或请求分桶，缓存键携带索引版本。

#### 易错点

- 只替换 query encoder 却继续检索旧 encoder 生成的向量。

#### 追问

- 全量重建期间新增文档怎样补到新索引？

<a id="sys-009"></a>
### SYS-009 · RAG 回答错了，怎样区分检索错误与生成错误？

**L2**

#### 答案

保留问题、真值、召回证据和回答，先确认正确证据是否存在，再用人工确定的 oracle 证据替换检索结果。若 oracle 下仍错，检查生成、模板或推理；若变对，检查召回与重排，再通过模块消融验证归因。

单独统计未召回、证据冲突、上下文裁剪、引用错配和不忠实生成。Oracle 对照保持相同上下文长度和提示规则，减少干扰；同时测 Recall@K 与最终答案正确率，单一端到端分数不足以定位全部错误。

#### 易错点

- 把所有错误都叫幻觉，或只增大 K 不查噪声。

#### 追问

- 多跳问题怎样定义“足够证据”？

<a id="sys-012"></a>
### SYS-012 · 微调上线后质量衰减，如何诊断数据漂移？

**L3**

#### 答案

先确认版本、模板与输入处理是否一致，再比较线上和训练/评测分布，按领域、语言、长度及用户类型切片。采样标注新错误，在旧模型上回放对照，区分分布漂移、接口退化和模型遗忘，再决定回滚或更新数据。

监控输入统计、拒答率、工具失败率及人评抽检，训练 loss 本身无法定位这些问题。评测采用符合线上场景的时间切分，避免数据泄漏；新数据微调需检查旧能力回归并灰度验证。

#### 易错点

- 质量变差就直接增大训练步数，不先排查服务配置。

#### 追问

- 无实时标签时可用哪些代理信号，局限是什么？

<a id="sys-014"></a>
### SYS-014 · Prompt 优化“修好一类坏了另一类”，怎样控制回归？

**L2**

#### 答案

把 prompt 作为版本化配置，维护按能力、业务和历史错误分层的固定回归集及新鲜验证集，比较各切片而非只看均值。为改动规定可接受的回归与收益，再用盲评和灰度确认。

记录完整模板、模型、工具 schema 与采样设置，避免多个因素同时变化。先归因失败，再选择修改通用指令还是局部路由或示例；锁定测试集，防止反复调到对固定题库过拟合。

#### 易错点

- 每次只拿新坏例验证，不跑历史回归。

#### 追问

- 如何判断问题应该改 prompt、数据还是模型？

<a id="topic-4"></a>
## 权限、容错与可观测性

<a id="sys-005"></a>
### SYS-005 · 工具调用失败后如何重试，怎样避免重复副作用？

**L2**

#### 答案

先区分瞬时失败、业务错误和结果未知的超时。写操作使用代表同一业务意图的幂等键，服务端校验重复键与参数一致性，并持久化执行结果；超时后先查询状态，再决定是否重试，因为超时不代表服务端未执行。

设置全局 deadline、有限次数和退避，避免多层重试放大负载。只读重试也消耗资源，需要预算、并发上限和熔断；支付或发消息等操作尤其要能查询既有结果。

#### 易错点

- 给所有异常无条件 retry，导致重复提交或重试风暴。

#### 追问

- 如果客户端取消但服务端已提交，怎样反馈最终状态？

<a id="sys-006"></a>
### SYS-006 · 设计 Agent 可观测性：日志里应该记录什么？

**L2**

#### 答案

用 `run_id/trace_id` 关联整次运行，记录模型与 prompt 版本、检索结果版本、工具参数摘要、耗时、错误和终止原因。规划、模型与工具调用各作为 span；跨服务传播 trace context，异步子任务保留父子关系。

同时记录质量和资源指标，才能判断任务是否完成及成本所在。敏感内容做脱敏与采样；回放使用固定工具快照或 mock，说明外部状态变化，并避免重放写操作。

#### 易错点

- 只有最后一个错误堆栈，没有中间状态和版本信息。

#### 追问

- 如何定位模型生成错误与工具返回错误？

<a id="sys-007"></a>
### SYS-007 · 多租户 RAG 如何保证权限隔离？

**L3**

#### 答案

在检索端用用户身份和文档 ACL 约束候选集合，必要时在返回前再次授权检查。过滤属性由可信服务端生成，不能直接相信用户传入的 `tenant_id`；prompt 中的保密指令无法替代服务端授权。

向量、关键词、重排、引用和缓存都携带租户与权限范围，缓存键包含权限范围或版本，权限变更触发失效。跨租户共享 prefix 或答案需评估泄漏渠道。测试撤权、删除、混合检索各分支和多轮对话里的旧引用。

#### 易错点

- 先全库检索再只隐藏链接，正文仍可能进模型上下文。

#### 追问

- 权限过滤使召回下降时怎样重新测 Recall@K？

<a id="sys-015"></a>
### SYS-015 · 线上模型事故如何止损、回滚与复盘？

**L3**

#### 答案

先按用户影响止损，可切回已验证版本、限制问题路径或暂停写操作，再保存日志和配置快照。回滚范围包含权重、模板、索引、adapter 与工具 schema；发布前要规定触发指标与兼容性，确保旧版仍可读取状态。

用时间线、切片和回放定位触发条件，区分触发因素、根因及放大因素。把事故转为回归样例、监控信号和演练，通过验证确认措施能捕获同类问题，避免复盘停留在个人责任归因。

#### 易错点

- 只回滚模型权重，漏掉同时更新的 tokenizer 或索引。

#### 追问

- 新旧数据 schema 不兼容时怎样回滚？

<a id="sys-017"></a>
### SYS-017 · 个人多模态记忆怎样存储、检索、更新并处理相互矛盾的信息？

**L3**

#### 答案

把短期会话状态与跨会话长期记忆分开。原图、音频或视频保存在对象存储；文本转写、摘要和视觉描述用于检索；结构化记录保存 user_id、来源、时间、图片区域或音视频片段坐标、内容 hash、置信度和版本。向量索引只负责相似召回，不能替代原证据或成为唯一事实数据库。

长期记忆区分稳定偏好、具体经历和任务状态。先按用户/权限、有效时间和类型过滤，再混合检索文本与模态向量，必要时重排；只把与当前任务有关的摘要和证据片段放入上下文。可以用相关性、时间与重要性的加权分数排序，但这些分数须在真实任务上校准。

写入时做去重和版本更新，记忆附证据；新的地址或偏好应终止旧记录的有效期，冲突不应随意取相似度最高者当真。用户纠正或删除时同步删除派生摘要、向量与缓存。评测覆盖跨会话召回、时间正确性、错误记忆写入和用户隔离，而不仅看检索命中率。

#### 易错点

- 视觉 embedding 相似只代表表示接近，不能证明图片里的事实或用户身份。
- 模型猜测写成稳定事实，或更新原记录却保留旧派生索引。

#### 追问

- 用户说“我搬家了”时，事实的时间有效性和旧经历应怎样分别保留？
- 怎样区分检索不到正确记忆与正确记忆被模型忽略？

## 参考资料

- [Azure AI Search Security Filter Pattern](https://learn.microsoft.com/en-us/azure/search/search-security-trimming-for-azure-search)
- [Google SRE Implementing SLOs](https://sre.google/workbook/implementing-slos/)
- [vLLM Metrics](https://docs.vllm.ai/en/latest/design/metrics/)
- [Google SRE Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [OpenTelemetry Traces](https://opentelemetry.io/docs/concepts/signals/traces/)
- [Prometheus Histograms and Summaries](https://prometheus.io/docs/practices/histograms/)
- [Google SRE Effective Troubleshooting](https://sre.google/sre-book/effective-troubleshooting/)
- [Effective context engineering — Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [AWS Making retries safe with idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)
- [Google SRE Handling Overload](https://sre.google/sre-book/handling-overload/)
- [Kubernetes Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Google SRE Canarying Releases](https://sre.google/workbook/canarying-releases/)
- [HELM](https://arxiv.org/abs/2211.09110)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- [vLLM Prefix Caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [Google Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Transformers Qwen2.5-VL documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl)
- [Python: threading](https://docs.python.org/3/library/threading.html)
- [Python: concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html)
- [PyMuPDF: Multiprocessing](https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html)
- [LangChain: Memory overview](https://docs.langchain.com/oss/python/concepts/memory)
- [Generative Agents](https://arxiv.org/html/2304.03442v2)
- [vLLM: Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
