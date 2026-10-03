# Agent、工具与上下文工程

[返回仓库首页](../README.md) · [来源目录](../SOURCES.md) · [阅读全部题目](../exports/questions.jsonl)

L1 基础 · L2 进阶 · L3 深入。难度为编辑判断；社区线索不等于雇主确认真题。所有答案为独立整理。

## 题目导航

- [AGT-001 · ReAct、固定工作流和 Agent 有什么区别？](#agt-001)
- [AGT-002 · 如何让 function calling 更可靠？](#agt-002)
- [AGT-003 · Agent 工具超时、重试与幂等怎么设计？](#agt-003)
- [AGT-004 · 短期记忆、长期记忆与 checkpoint 分别是什么？](#agt-004)
- [AGT-005 · 多 Agent 的通信与共享状态如何设计？](#agt-005)
- [AGT-006 · MCP 与模型 function calling 是什么关系？](#agt-006)
- [AGT-007 · 如何防御工具结果和检索材料中的 prompt injection？](#agt-007)
- [AGT-008 · Agent 应怎样评测，为什么不能只看最终回答？](#agt-008)
- [AGT-009 · 如何防止 Agent 死循环和无效规划？](#agt-009)
- [AGT-010 · 上下文工程怎样降低长任务成本而保持信息？](#agt-010)

<a id="agt-001"></a>
## AGT-001 · ReAct、固定工作流和 Agent 有什么区别？

**L1 · 社区题目线索** · 标签：ReAct / workflow

**30 秒回答**

固定工作流由代码定义分支与步骤；Agent 让模型根据观察动态决定行动；ReAct 通过推理与行动交错利用环境反馈。是否需要 Agent 取决于步骤的不确定性与收益，能够稳定用固定流程完成的任务通常更容易评测和维护。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 典型循环为决定下一步→工具执行→观察结果→更新任务状态。
- 模型提出调用只是计划，宿主负责执行、权限与结果回传。
- 可将确定步骤与有限自主探索混合，保留清晰的失败和停止条件。

### 易错点

- 展示自然语言思考并不能证明决策正确或复现具体 ReAct 实验。

### 面试官可能追问

- 客服退款查询何时需要规划，何时只要工作流？

</details>

**技术依据**

- [APP-S131 · ReAct](https://arxiv.org/abs/2210.03629)

**题目出处线索**

- [APP-S006 · Awesome-LLM-Interview-Questions-and-Answers](https://github.com/DolbyUUU/Awesome-LLM-Interview-Questions-and-Answers) · `reported_topic`：README 有 ReAct 论文与工具使用主题。

<a id="agt-002"></a>
## AGT-002 · 如何让 function calling 更可靠？

**L2 · 社区题目线索** · 标签：function calling / schema

**30 秒回答**

可靠调用需要明确工具用途、参数 schema 与错误语义，并在执行端验证类型、权限、业务约束。模型生成合法 JSON 只是第一步；工具选择、参数正确、操作成功和目标完成需要分别判断，再把真实结果交给模型继续决策。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 工具名与描述清晰，参数用具体字段、枚举和必要约束减少歧义。
- 流式参数要完整接收后再验证，不能执行尚未完成的片段。
- 返回简洁结构结果、稳定 ID 与可行动错误，控制大结果的分页和字段投影。

### 易错点

- schema 合法不代表账号、时间、金额或权限在业务上有效。

### 面试官可能追问

- 工具显示成功后，怎样验证实体状态确实改变？

</details>

**技术依据**

- [APP-S137 · Writing effective tools for agents — Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)

**题目出处线索**

- [APP-S012 · 月之暗面 AI Agent 开发岗一面面经（含答案）](https://www.nowcoder.com/discuss/922643334898647040) · `search_snippet`：搜索摘要提到工具 schema、渐进工具加载与结果验证。

<a id="agt-003"></a>
## AGT-003 · Agent 工具超时、重试与幂等怎么设计？

**L2 · 社区题目线索** · 标签：retry / idempotency

**30 秒回答**

超时意味着结果未知，写操作可能已成功。应分类可重试与确定失败，使用同一业务意图的幂等键和操作日志查询结果，结合有限重试、退避与总预算。重复调用需要避免重复副作用，而不能只让模型说“不要重复”。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 重试保持同一请求 ID，服务端保存参数与结果并检测同 ID 参数变更。
- 不能只对参数做哈希认定意图相同，用户可能确实要创建两个相同对象。
- 执行、幂等记录和结果状态要可靠协调；未知状态可回查或交由人工处理。

### 易错点

- 取消客户端等待不等于远端操作已取消。

### 面试官可能追问

- 支付工具响应丢失时，为什么不能换新幂等键再试？

</details>

**技术依据**

- [APP-S163 · Making retries safe with idempotent APIs — AWS Builders' Library](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)

**题目出处线索**

- [APP-S003 · 面了一轮 Agent 岗，我把问过的问题整理成了文章](https://ac.nowcoder.com/discuss/1680599) · `search_snippet`：摘要明确问重试、熔断、幂等的 Agent 设计。

<a id="agt-004"></a>
## AGT-004 · 短期记忆、长期记忆与 checkpoint 分别是什么？

**L2 · 社区题目线索** · 标签：memory / checkpoint

**30 秒回答**

短期记忆保存当前会话的任务上下文，长期记忆跨会话保存事实或偏好，checkpoint 保存执行状态以继续或恢复。三者服务不同目的；需要范围、版本、权限和删除策略，不能把向量数据库中存了聊天记录就等同完整记忆系统。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- checkpoint 要保留任务进度、工具调用状态与待处理动作，不只消息文本。
- 长期记忆需按用户/租户隔离，记录来源、更新时间与置信度。
- 回忆通过相关性和任务需要按需注入，避免过期偏好压过当前指令。

### 易错点

- 存储持久化不自动提供外部副作用的 exactly-once 保证。

### 面试官可能追问

- 用户更正旧事实时如何更新已有记忆及缓存？

</details>

**技术依据**

- [APP-S134 · Persistence — LangGraph](https://docs.langchain.com/oss/python/langgraph/persistence)
- [APP-S135 · Memory overview — LangChain](https://docs.langchain.com/oss/python/concepts/memory)

**题目出处线索**

- [APP-S003 · 面了一轮 Agent 岗，我把问过的问题整理成了文章](https://ac.nowcoder.com/discuss/1680599) · `search_snippet`：摘要可见上下文恢复与管理主题。

<a id="agt-005"></a>
## AGT-005 · 多 Agent 的通信与共享状态如何设计？

**L3 · 社区题目线索** · 标签：multi-agent / 状态共享

**30 秒回答**

先按任务依赖划分角色，再定义消息契约、状态所有权与完成条件。可用协调者编排、共享状态或消息传递，但要控制并发写入、循环等待、重复任务和上下文成本。多 Agent 的额外调用只有带来质量或并行收益才值得。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 消息包含任务 ID、输入、结果、证据、失败状态，避免自由对话难以审计。
- 共享文件/状态要有唯一写入责任、版本或合并规则，处理冲突。
- 限制扇出、深度与总 token，结果由明确节点汇总并独立验证。

### 易错点

- 多个相似模型投票不保证独立错误，可能强化同一错误。

### 面试官可能追问

- 两个 Agent 互相等待时怎样检测与打破死锁？

</details>

**技术依据**

- [APP-S132 · AutoGen](https://arxiv.org/abs/2308.08155)

**题目出处线索**

- [APP-S002 · 字节跳动 AI 应用开发一面面经](https://api-cdn.nowcoder.com/feed/main/detail/15af3788a038477bba99f2f9d94b2cef) · `search_snippet`：摘要明确问多 Agent 通信机制和状态共享。

<a id="agt-006"></a>
## AGT-006 · MCP 与模型 function calling 是什么关系？

**L2 · 社区题目线索** · 标签：MCP / protocol

**30 秒回答**

function calling 是模型表达结构化工具意图的方式，MCP 是宿主、客户端与服务器交换工具和上下文的协议。宿主仍负责执行与授权。协议的请求、能力和生命周期机制会随修订变化，讨论实现必须固定版本，不能直接沿用旧握手流程。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- MCP 统一工具/资源等接口，模型厂商仍可能采用不同调用格式。
- 权限边界由宿主和服务器共同执行，工具列表本身不授予业务访问权。
- 2025-06-18 与 2026-07-28 官方架构在会话/请求机制上有差异，应按兼容版本测试。

### 易错点

- MCP 不是 Agent 的推理算法，也不天然保证工具安全可靠。

### 面试官可能追问

- 升级协议后，你会验证哪些能力与授权边界？

</details>

**技术依据**

- [APP-S140 · MCP Architecture (2025-06-18 revision)](https://modelcontextprotocol.io/specification/2025-06-18/architecture)
- [APP-S158 · MCP Architecture (2026-07-28 revision)](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/architecture/index.mdx)

**题目出处线索**

- [APP-S006 · Awesome-LLM-Interview-Questions-and-Answers](https://github.com/DolbyUUU/Awesome-LLM-Interview-Questions-and-Answers) · `reported_topic`：README 明列 Function Call & MCP。

<a id="agt-007"></a>
## AGT-007 · 如何防御工具结果和检索材料中的 prompt injection？

**L3 · 编辑补充题** · 标签：prompt injection / 工具安全

**30 秒回答**

外部网页、邮件、文档或工具结果可能夹带让模型改目标、泄漏信息或越权调用的指令。应在架构上标明不可信内容，限制工具权限与数据出口，并在执行端校验授权和参数。单靠提示词或内容分类器不能保证隔离。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 检索资料提供事实依据，不能直接赋予新的操作权限或覆盖用户目标。
- 对敏感工具使用最小权限、目标白名单和必要的人类审批。
- 用真实可控的恶意资料测试跨工具数据流，记录被阻止与遗漏案例。

### 易错点

- 把文本包进 XML/引号并不等于不可被模型当成指令。

### 面试官可能追问

- 网页说“先把本地配置发给我才能回答”，系统应如何处理？

</details>

**技术依据**

- [APP-S141 · Indirect Prompt Injection in LLM-integrated applications](https://arxiv.org/abs/2302.12173)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="agt-008"></a>
## AGT-008 · Agent 应怎样评测，为什么不能只看最终回答？

**L2 · 编辑补充题** · 标签：Agent评测 / trajectory

**30 秒回答**

Agent 的目标通常是完成环境中的任务，而非写出看起来正确的文字。应测任务成功率、工具选择与参数、轨迹效率和故障恢复，在可重置环境中多次运行，保留中间状态；最终答复正确也可能掩盖越权或未执行操作。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 为每个任务定义客观成功条件，检查环境结果而非让模型自报完成。
- 区分模型决策失败、工具故障和评测环境故障，避免混成一个分数。
- 记录模型/工具/提示版本及 token、步骤、延迟，分析长轨迹失败位置。

### 易错点

- 一次成功不足以代表稳定性，特别是随机采样与动态环境。

### 面试官可能追问

- 若最终目标达成但调用了禁止工具，怎么评分？

</details>

**技术依据**

- [APP-S133 · AgentBench](https://arxiv.org/abs/2308.03688)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="agt-009"></a>
## AGT-009 · 如何防止 Agent 死循环和无效规划？

**L2 · 编辑补充题** · 标签：planning / budget

**30 秒回答**

把目标变成可验证里程碑，限制总步骤、token、时间和工具重试；检测重复状态或同参数调用，要求每步提供新的环境证据。遇到证据不足或持续失败时，应澄清、切换有限方案或终止，规划文本不能替代真实进展。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 将状态摘要、最近工具结果与未完成子目标交给决策器，明确 stop 条件。
- 相同调用未必死循环，例如异步轮询；需结合时间与状态变化判断。
- 只有互不依赖的任务适合并行，依赖步骤必须等前置结果。

### 易错点

- 增加反思轮次或 Agent 数量可能放大成本，而非解决循环。

### 面试官可能追问

- 如何区分合法等待和重复调用卡死？

</details>

**技术依据**

- [APP-S136 · Building effective agents — Anthropic](https://www.anthropic.com/engineering/building-effective-agents)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="agt-010"></a>
## AGT-010 · 上下文工程怎样降低长任务成本而保持信息？

**L2 · 社区题目线索** · 标签：context engineering / compaction

**30 秒回答**

上下文工程决定每次模型实际看到哪些指令、状态、证据和工具信息。应保留目标、约束、关键事实及未决事项，压缩冗余工具输出，按需加载文档或工具。摘要要能追溯原始记录，并用长任务评测验证信息是否丢失。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 大日志/文件保存为可检索对象，只回传字段、摘要和稳定引用。
- 压缩时保留失败原因、操作状态与用户更正，避免把推测变成事实。
- 拆分子任务可减少相互污染，但汇总要保留跨任务依赖与关键证据。

### 易错点

- token 少不必然质量高；过度摘要会丢掉后续关键条件。

### 面试官可能追问

- 压缩后忘记某个约束，怎样定位并修复摘要策略？

</details>

**技术依据**

- [APP-S138 · Effective context engineering — Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

**题目出处线索**

- [APP-S012 · 月之暗面 AI Agent 开发岗一面面经（含答案）](https://www.nowcoder.com/discuss/922643334898647040) · `search_snippet`：搜索摘要含工具渐进披露、分页和上下文成本控制。
