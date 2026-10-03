# 推理、KV Cache 与量化

[返回仓库首页](../README.md) · [来源目录](../SOURCES.md) · [阅读全部题目](../exports/questions.jsonl)

L1 基础 · L2 进阶 · L3 深入。难度为编辑判断；社区线索不等于雇主确认真题。所有答案为独立整理。

## 题目导航

- [INF-001 · KV cache 缓存什么，为什么通常不缓存历史 Q？](#inf-001)
- [INF-002 · 怎样估算推理权重、KV cache 与总显存？](#inf-002)
- [INF-003 · MHA、MQA 与 GQA 的结构、KV 显存及速度有何差异？](#inf-003)
- [INF-004 · prefill 与 decode 各在做什么，瓶颈为何不同？](#inf-004)
- [INF-005 · FlashAttention 的核心思想是什么，会改变注意力结果吗？](#inf-005)
- [INF-006 · PagedAttention 与 FlashAttention 解决的问题有何不同？](#inf-006)
- [INF-007 · continuous batching 与普通 dynamic batching 有何区别？](#inf-007)
- [INF-008 · temperature、top-k 与 top-p 如何影响生成？](#inf-008)
- [INF-009 · PTQ、QAT、W4A16 与 per-group quantization 是什么？](#inf-009)
- [INF-010 · GPTQ 为什么利用二阶信息进行逐层量化？](#inf-010)
- [INF-011 · AWQ 怎样保护重要权重，是否把 1% 权重保留为 FP16？](#inf-011)
- [INF-012 · SmoothQuant 为什么把激活的量化困难迁移到权重？](#inf-012)
- [INF-013 · 投机解码怎样保证目标模型的采样分布？](#inf-013)
- [INF-014 · prefix caching 与普通 KV cache 有何区别？](#inf-014)
- [INF-015 · 如何同时优化 TTFT、每 token 延迟与吞吐？](#inf-015)

<a id="inf-001"></a>
## INF-001 · KV cache 缓存什么，为什么通常不缓存历史 Q？

**L1 · 编辑补充题** · 标签：KVCache / 自回归 / 加速

**30 秒回答**

因果生成中已处理 token 的各层 K、V 可复用，下一步只计算新 token 并查询历史缓存。历史 Q 的注意力结果已用于对应位置，之后不用再查询，因此通常不缓存。KV cache 用显存换重复计算减少，不会免费降低总显存。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 缓存成立依赖因果性：新增未来 token 不改变已有位置的表示；双向 attention 一般不能原样复用。
- prefill 产生前缀 K/V，decode 逐步追加当前 token，各层都有自己的缓存。
- 新 query 仍需读取历史 K/V，因此单步 attention 成本随上下文长度增长。
- 改变模型权重、LoRA、前缀或位置约定会影响缓存有效性。

### 易错点

- KV cache 能把每个生成 token 的 attention 复杂度变成常数。
- 把节省重算等同于减少模型总显存。

### 面试官可能追问

- 为什么训练时通常不以同样方式使用 KV cache？
- 滑动窗口注意力能怎样限制缓存大小？

</details>

**技术依据**

- [CORE-S055 · Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/pdf/1911.02150)
- [CORE-S056 · GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="inf-002"></a>
## INF-002 · 怎样估算推理权重、KV cache 与总显存？

**L2 · 社区题目线索** · 标签：显存估算 / KVCache / GQA

**30 秒回答**

总显存应拆成权重、KV cache、激活和临时工作区及框架预留。标准全注意力 KV 大小约为 2BLTH_kv d_h s，使用 KV 头数而非 query 头数；实际还有分页、量化 scale、并行布局和碎片，理论值需与实测峰值比较。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- B 为并发序列数、L 为层数、T 为缓存长度、H_kv 为 KV 头数、d_h 为头维度、s 为每元素字节。
- 32 层、32 KV 头、头维 128、长度 4096、单序列、FP16，KV 为 2 GiB；8 KV 头时为 0.5 GiB。
- 7×10^9 参数 FP16 权重约 14 GB，即约 13.0 GiB，仅是权重部分。
- 不同请求长度应按 ΣT_i 算；tensor parallel 是否均分或复制 KV 要看具体实现。

### 公式

```text
M_KV≈2·B·L·T·H_kv·d_h·s
```

### 易错点

- 把 GB 和 GiB 混为一谈。
- 用 MHA 的 query 头数计算 GQA 的 KV 大小。

### 面试官可能追问

- 多模态视觉 token 怎样计入 KV 长度？
- 为什么 4-bit 权重文件大小不等于线上显存？

</details>

**技术依据**

- [CORE-S055 · Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/pdf/1911.02150)
- [CORE-S056 · GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245)
- [CORE-S058 · Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

**题目出处线索**

- [CORE-S003 · AgentGuide：公司面试案例整理](https://github.com/adongwanai/AgentGuide/blob/main/docs/04-interview/12-company-interview-cases.md) · `reported_question`：二次汇编题目列表出现相应提问；原始公司面经未逐条核验。

<a id="inf-003"></a>
## INF-003 · MHA、MQA 与 GQA 的结构、KV 显存及速度有何差异？

**L1 · 社区题目线索** · 标签：MHA / MQA / GQA / Shapes / KVBytes / TensorParallel

**30 秒回答**

MHA 为每个 query 头配独立 K/V，MQA 全部 query 头共享一组 K/V，GQA 按组共享。它们保留 query 头数量，主要减少 KV 投影、缓存与解码读带宽。性能收益受内核、并行布局和上下文影响，不能承诺质量无损或时延按 KV 头数等比例下降。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 设 H_q 为 query 头、H_kv 为 KV 头，MHA:H_kv=H_q，MQA:H_kv=1，GQA:1<H_kv<H_q。均匀分组通常要求 H_q%H_kv=0，query 头 i 使用组 g(i) 的 K/V；不是先把所有 Q 头平均成较少的 query。
- 典型形状 Q=[B,H_q,L_q,d_h]、K/V=[B,H_kv,L_k,d_h]，每个 query 头计算 softmax(Q_i K_g^T/√d_h)V_g，最后拼接输出。group repeat 只是表达逻辑映射；高效实现可避免物理复制 KV，显式 repeat 可能抹掉部分显存/带宽收益。
- 若 K/V head 维相同，N 层、总缓存 token 数 T_total、每元素 b bytes，则 KV≈2N T_total H_kv d_h b。保持其他量固定，从 32 KV 头降到 8 头是 1/4 的 KV 容量；query、输出投影和大部分 FFN 成本不会同步变成 1/4。
- decode 小 Q/长 K 时常受读 KV 带宽限制，因此共享 KV 更有利；prefill 仍需为各 query 头形成注意力匹配，速度还依赖 kernel、batch、序列长和硬件。TP 数超过 KV 头数时实现可能复制 KV，不能直接按 TP 数继续理想平分。
- 从已有 MHA 转换 GQA 时，原论文按组聚合 K/V 权重并做 uptraining；直接平均并上线不保证恢复质量。GQA 常提供容量与效率折中，需以目标任务、长上下文和实际服务压测比较，MQA 不是任意任务零精度代价。

### 公式

```text
KV_bytes≈2N T_total H_kv d_h b; Attention_i=softmax(Q_i K_{g(i)}^T/√d_h)V_{g(i)}; H_q%H_kv=0（均匀分组）
```

### 易错点

- 将 GQA 写成减少全部 attention heads，或认为逻辑共享等于张量实现完全不复制。
- 用 KV 头比例直接预测端到端 latency、训练 FLOPs 或质量变化。

### 面试官可能追问

- GQA 的 query-to-KV 分组在多 GPU 分片时如何保持一致？
- 同样的 KV 缓存预算下，增加并发和增加上下文长度怎样权衡？

</details>

**技术依据**

- [CORE-S055 · Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/pdf/1911.02150)
- [CORE-S056 · GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245)
- [CORE-S074 · PyTorch SDPA — masks, shapes and GQA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

**题目出处线索**

- [CORE-S005 · 大模型算法面经+问题+答案](https://www.nowcoder.com/discuss/891322059656052736?toCommentId=22719031) · `search_snippet`：正文抓取失败，搜索摘要明确出现相关问题主题，保留摘要级证据。

<a id="inf-004"></a>
## INF-004 · prefill 与 decode 各在做什么，瓶颈为何不同？

**L2 · 编辑补充题** · 标签：Prefill / Decode / Roofline

**30 秒回答**

prefill 一次处理输入前缀并生成缓存，序列并行度高；decode 通常每条请求每步处理一个新 token，并反复读取权重与 KV。前者常更偏计算，后者在小 batch 时常更偏带宽，但长上下文、batch 和硬件会改变瓶颈。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- prefill 的稠密注意力有二次长度项，线性层可在多个 token 上形成较大的 GEMM。
- decode 的新 query 关注历史 T 个 key，单步 attention 算术量约 O(Td)。
- 增加 batch 可摊薄权重读取并提高利用率，却增加排队、显存占用与单请求竞争。
- 用 profiler 判断权重带宽、KV 带宽、算子启动或计算瓶颈，再选择量化、batching 或融合。

### 易错点

- prefill 永远算力瓶颈、decode 永远带宽瓶颈。
- 只看总 tokens/s 而不看输入输出长度。

### 面试官可能追问

- chunked prefill 为什么有助于控制 decode 干扰？
- 如何用 roofline 判断量化是否可能加速？

</details>

**技术依据**

- [CORE-S055 · Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/pdf/1911.02150)
- [CORE-S056 · GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245)
- [CORE-S057 · FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
- [CORE-S059 · Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="inf-005"></a>
## INF-005 · FlashAttention 的核心思想是什么，会改变注意力结果吗？

**L2 · 社区题目线索** · 标签：FlashAttention / IOAware / OnlineSoftmax

**30 秒回答**

FlashAttention 通过分块与在线 softmax，在片上内存计算并累积注意力输出，避免把完整 T×T 分数矩阵写回显存。它计算精确稠密注意力，主要优化 IO 和中间存储；浮点结果可有小差异，算术复杂度仍是二次。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 分块加载 Q/K/V，在 SRAM 内计算局部分数，维护每行最大值及归一化累计量。
- 改变求和顺序产生浮点误差，不应声称逐 bit 完全一致，也不等同于近似稀疏 attention。
- 训练反向可重算部分统计和分数，少保存激活，以更多局部计算换更少 HBM IO。
- 收益依赖序列长度、dtype、硬件与内核支持，短序列或不兼容 mask 下不保证加速。

### 易错点

- 说它把稠密注意力 FLOPs 从 O(T²) 降到 O(T)。
- 把 FlashAttention 与 KV cache 视为同一技术。

### 面试官可能追问

- 在线 softmax 合并两个块时为什么需要重缩放？
- FlashAttention-2 进一步优化了什么？

</details>

**技术依据**

- [CORE-S057 · FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

**题目出处线索**

- [CORE-S001 · 网易大模型应用岗面经（一面、二面）](https://www.nowcoder.com/discuss/909223288612610048?sourceSSR=dynamic) · `reported_question`：公开面经题目列表直接出现这一主题；题目已改写，答案独立整理，公司归属未独立认证。

<a id="inf-006"></a>
## INF-006 · PagedAttention 与 FlashAttention 解决的问题有何不同？

**L2 · 编辑补充题** · 标签：PagedAttention / vLLM / 碎片

**30 秒回答**

PagedAttention 把 KV cache 分成块，通过逻辑到物理映射管理显存，降低预留浪费和碎片，并支持共享缓存。FlashAttention 优化注意力计算的 IO 和中间矩阵。二者可配合，分页不会减少保存同样 K/V 所需的理论元素数量。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 每个请求的逻辑 KV 块可映射到不连续的物理块，按需分配而非预留整个最大上下文。
- 块表使内核按块访问缓存；块尺寸影响内存浪费、映射开销及内核效率。
- 共享前缀或分支生成可用引用计数等管理共享，修改时需避免相互覆盖。
- 优化的是管理开销与冗余；极高并发下仍可能耗尽 KV，需要准入或抢占策略。

### 易错点

- 分页可以让单序列无限增长且不占显存。
- 以论文倍数直接承诺任意服务吞吐提升。

### 面试官可能追问

- 块尺寸如何影响尾部浪费？
- 共享块在分支生成时为什么可能需要 copy-on-write？

</details>

**技术依据**

- [CORE-S058 · Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [CORE-S057 · FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="inf-007"></a>
## INF-007 · continuous batching 与普通 dynamic batching 有何区别？

**L2 · 编辑补充题** · 标签：ContinuousBatching / 调度 / 吞吐

**30 秒回答**

普通 dynamic batching 常在请求开始前拼批；continuous batching 可以在生成迭代之间移除完成请求、加入新请求，减少等待整批结束的空闲。吞吐提高仍需满足延迟目标，调度还要处理 prefill、KV 容量及长请求公平性。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 迭代级调度在每轮重新选择请求，适合生成长度和到达时间不一致的工作负载。
- 退出的请求释放资源，新请求能利用空出的 batch 位置。
- 长 prefill 会干扰 decode，可用预算或分块处理控制单轮计算量。
- 固定请求率、输入输出长度分布和 SLO 做负载扫描，比较吞吐与尾部延迟。

### 易错点

- 把 continuous batching 简单解释为所有请求永远组成最大 batch。
- 只报告峰值 throughput，不报告排队时间。

### 面试官可能追问

- 短请求如何避免被长请求拖累？
- batch size 上限和每轮 token budget 含义有何不同？

</details>

**技术依据**

- [CORE-S059 · Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [CORE-S058 · Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="inf-008"></a>
## INF-008 · temperature、top-k 与 top-p 如何影响生成？

**L1 · 编辑补充题** · 标签：Sampling / Temperature / TopP

**30 秒回答**

temperature 调整 logits 尖锐程度，top-k 保留固定数量候选，top-p 保留累计概率达到阈值的最小候选集合。它们控制采样分布及多样性，不能直接保证事实正确；贪心解码还受并行数值与实现条件影响。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- temperature>0 时 p_i∝exp(z_i/τ)，较小 τ 通常更集中，τ=0 需按实现采用贪心等特殊规则。
- top-k 固定候选数量，top-p 随分布变化动态决定数量。
- 过滤后需重新归一化，多个参数联用的执行顺序要看具体框架。
- 开放创作与可验证问答可采用不同设置，固定随机种子有助复现实验但不保证跨环境逐 token 一致。

### 易错点

- temperature=0 就绝对不会出现幻觉。
- 把 top-p 当作独立地保留概率大于 p 的 token。

### 面试官可能追问

- 为什么 beam search 在开放生成中可能重复？
- top-p 与 temperature 的顺序为什么重要？

</details>

**技术依据**

- [CORE-S060 · The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [CORE-S068 · Transformers: Utilities for generation — sampling warpers](https://huggingface.co/docs/transformers/internal/generation_utils)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="inf-009"></a>
## INF-009 · PTQ、QAT、W4A16 与 per-group quantization 是什么？

**L1 · 社区题目线索** · 标签：Quantization / PTQ / QAT

**30 秒回答**

PTQ 在训练后进行量化，QAT 在训练中模拟或考虑量化误差。W4A16 指权重约 4 bit、激活约 16 bit，不表示所有算子都 INT4；per-group 为一组值共享量化参数，组越小通常误差更低，但元数据和实现成本增加。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 仿射量化常写 q=clip(round(x/s)+z)，反量化为 x̂=s(q−z)，s 为 scale、z 为 zero-point。
- per-tensor、per-channel、per-group 是参数共享粒度，不能只报 bit 数而不报粒度。
- 总存储还含 scale、zero-point、未量化层及对齐，不能只按参数×4/8估算。
- 是否更快取决于内核和硬件；校准须覆盖部署输入、长度与领域。

### 公式

```text
q=clip(round(x/s)+z); x̂=s(q−z)
```

### 易错点

- 低 bit 一定比 FP16 更快。
- 把权重量化与 KV cache 量化混成同一项。

### 面试官可能追问

- outlier 为什么会增大量化误差？
- 怎样选择校准集并评估任务精度损失？

</details>

**技术依据**

- [CORE-S047 · QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)
- [CORE-S061 · GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/pdf/2210.17323)
- [CORE-S062 · AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [CORE-S063 · SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/pdf/2211.10438)

**题目出处线索**

- [CORE-S002 · 腾讯Teg大模型暑期算法面经](https://api-cdn.nowcoder.com/feed/main/detail/f91200d9c116401090432a2d78e5f76d?sourceSSR=users) · `reported_question`：公开帖子列出架构、LoRA、量化原理；仅保留问题主题，未采纳社区答案。

<a id="inf-010"></a>
## INF-010 · GPTQ 为什么利用二阶信息进行逐层量化？

**L2 · 社区题目线索** · 标签：GPTQ / PTQ / Calibration

**30 秒回答**

GPTQ 利用校准输入形成近似二阶信息，逐步量化权重并补偿未量化权重，以降低层输出重构误差。它通常属于低 bit 权重后训练量化；效果依赖校准覆盖、分组与阻尼等设置，不是无损压缩。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 常用目标是最小化原层输出 WX 与量化层输出 ŴX 的差异，X 来自校准数据。
- 输入二阶统计提供权重误差对输出的敏感性，比独立 round-to-nearest 更有信息。
- 按列或块处理并更新剩余权重误差，工程上采用分块等降低计算成本。
- 校准集分布错配会影响领域、长上下文或多模态效果，需做端到端任务回归。

### 易错点

- GPTQ 会更新底座进行完整反向训练。
- 把论文中某模型的近乎无损结果推广到所有位宽和场景。

### 面试官可能追问

- 校准 Hessian 病态时为什么可能需要 damping？
- GPTQ 与普通舍入怎样公平比较？

</details>

**技术依据**

- [CORE-S061 · GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/pdf/2210.17323)

**题目出处线索**

- [CORE-S009 · 大厂问什么：2025-26 算法工程师面试常见问题整理（阿里系）](https://www.nowcoder.com/discuss/848942791164981248?sourceSSR=dynamic) · `search_snippet`：搜索摘录列出复杂度、GPTQ 或投机解码主题；未采用其自报次数/职级概率，公司归属为汇编者自述。

<a id="inf-011"></a>
## INF-011 · AWQ 怎样保护重要权重，是否把 1% 权重保留为 FP16？

**L2 · 编辑补充题** · 标签：AWQ / ActivationAware / WeightOnly

**30 秒回答**

AWQ 根据激活统计识别重要权重通道，通过等价通道缩放减少低 bit 量化误差。论文讨论少量显著权重的重要性，但正式方案避免硬件不友好的混合精度，因此不能概括为把 1% 权重直接保留在 FP16。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 权重是否敏感应结合对应激活大小，单看权重绝对值不足以判断。
- 放大重要权重通道、对激活作逆缩放，在未量化时保持线性层结果等价。
- scale 使用离线校准统计搜索，正式 AWQ 方案不依赖完整梯度训练。
- 常见 weight-only 低 bit 部署仍需有效内核，精度、长度及硬件速度都应实测。

### 易错点

- 把 AWQ 的启发性实验混成最终采用的混合精度方案。
- AWQ 只看权重统计，不看激活。

### 面试官可能追问

- 为什么通道缩放能改变量化误差却不改变未量化输出？
- AWQ 与 GPTQ 的校准目标有何差别？

</details>

**技术依据**

- [CORE-S062 · AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="inf-012"></a>
## INF-012 · SmoothQuant 为什么把激活的量化困难迁移到权重？

**L2 · 编辑补充题** · 标签：SmoothQuant / W8A8 / Outlier

**30 秒回答**

LLM 激活的少数大值通道使直接低精度量化困难。SmoothQuant 用离线等价通道缩放压低激活尺度、相应放大权重，让两者更适合 W8A8。缩放前后的浮点函数等价，量化后的误差仍需用代表性输入验证。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 线性层 XW 可重写为 (XS^{-1})(SW)，S 是按输入通道定义的正对角矩阵。
- 平滑系数需要平衡激活与权重的动态范围，压低一方可能增大另一方误差。
- 可把缩放吸收到前后相关参数中，减少运行时额外操作。
- 不同激活量化粒度、校准长度和硬件 INT8 支持决定实际效果。

### 公式

```text
XW=(XS^{-1})(SW)，S 为可逆对角缩放矩阵
```

### 易错点

- 浮点等价变换就意味着量化后零误差。
- 把 SmoothQuant 一概称为 W4A16 权重量化。

### 面试官可能追问

- 为什么逐输入通道激活 scale 不一定适合常规 INT8 GEMM？
- 部署输入出现新 outlier 时怎么办？

</details>

**技术依据**

- [CORE-S063 · SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/pdf/2211.10438)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="inf-013"></a>
## INF-013 · 投机解码怎样保证目标模型的采样分布？

**L3 · 社区题目线索** · 标签：SpeculativeDecoding / 拒绝采样 / 加速

**30 秒回答**

投机解码由较便宜的 draft 提议多个 token，target 并行验证，再按接受与修正采样规则输出。标准算法保证目标分布，而非要求每次随机运行生成完全相同文本；速度取决于接受率、draft 成本和验证开销。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 草稿分布 q、目标分布 p 时，对候选 x 用 min(1,p(x)/q(x)) 接受。
- 首次拒绝后从归一化的 max(p−q,0) 修正分布采样，不能直接从 p 重采而仍宣称同分布证明成立。
- 全接受后通常还能从 target 的下一位置分布采一个额外 token。
- tokenizer、采样变换与约束规则必须协调，长草稿不一定更快，低接受率或大 batch 下可能收益不足。

### 公式

```text
a(x)=min(1,p(x)/q(x)); 拒绝后的修正分布 ∝ max(p−q,0)
```

### 易错点

- 投机解码靠牺牲目标模型精度换速度。
- 分布一致就宣称固定种子下文本逐 token 必然一致。

### 面试官可能追问

- 怎样根据接受率选择草稿长度？
- 贪心验证与随机采样验证的规则有什么区别？

</details>

**技术依据**

- [CORE-S064 · Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192)

**题目出处线索**

- [CORE-S009 · 大厂问什么：2025-26 算法工程师面试常见问题整理（阿里系）](https://www.nowcoder.com/discuss/848942791164981248?sourceSSR=dynamic) · `search_snippet`：搜索摘录列出复杂度、GPTQ 或投机解码主题；未采用其自报次数/职级概率，公司归属为汇编者自述。

<a id="inf-014"></a>
## INF-014 · prefix caching 与普通 KV cache 有何区别？

**L2 · 编辑补充题** · 标签：PrefixCaching / 缓存失效 / 多模态

**30 秒回答**

普通 KV cache 复用同一生成请求的历史；prefix caching 在请求间复用相同前缀的已计算 KV，主要减少重复 prefill。缓存键要包含足够的模型、适配器、token 和模态信息，语义相似的文本并不能直接共享精确 KV。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 哈希通常包含父块哈希和当前块 token，确保相同局部文本但不同前文不会误共享。
- 模型权重、LoRA 及相关预处理变化后必须失效或隔离缓存。
- 多模态占位 token 相同但图像不同，需把模态内容哈希纳入键。
- 缓存占用会与活跃请求竞争，需 eviction；多租户共享要考虑隔离及时间侧信道。

### 易错点

- 只用当前文本片段哈希，不考虑前缀上下文。
- prefix cache 会加速输出 token 的全部 decode 计算。

### 面试官可能追问

- 相同 system prompt 放在最前面为什么更利于命中？
- 多适配器和多租户怎样隔离缓存？

</details>

**技术依据**

- [CORE-S065 · Automatic Prefix Caching — vLLM v0.20.1](https://docs.vllm.ai/en/v0.20.1/design/prefix_caching/)
- [CORE-S058 · Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="inf-015"></a>
## INF-015 · 如何同时优化 TTFT、每 token 延迟与吞吐？

**L2 · 社区题目线索** · 标签：TTFT / ITL / SLO

**30 秒回答**

TTFT 关注从请求到首 token，ITL 关注输出相邻 token 间隔，吞吐关注单位时间产出。优化应固定负载和输入输出长度分布，用延迟 SLO 内的有效吞吐比较，排队、prefill、decode 及客户端流式传输都会影响体验。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- TTFT 包含排队和前缀处理，不能只测模型 prefill kernel 时间。
- 记录 P50/P95/P99、失败率、请求率、输入/输出 token 及并发，区分客户端和服务端指标。
- 扩大 batch 可提高利用率，也可能提高排队和 ITL；chunked prefill、缓存与调度需联合权衡。
- 逐步增加到达率找饱和点，比较满足 SLO 的 goodput，而非只报离线最大 tokens/s。

### 易错点

- 首 token 很快就断言整体响应快。
- 把输入吞吐与输出吞吐混用或省略长度分布。

### 面试官可能追问

- 低并发与高并发应采用同一调度参数吗？
- 缓存命中率变化怎样影响 benchmark 公平性？

</details>

**技术依据**

- [CORE-S066 · Metrics — vLLM v0.20.1](https://docs.vllm.ai/en/v0.20.1/design/metrics/)
- [CORE-S059 · Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)

**题目出处线索**

- [CORE-S007 · 说说现在到底都在问什么（长文，慎入）](https://api-cdn.nowcoder.com/discuss/925432340174606336?sourceSSR=subject) · `reported_question`：公开面经汇总直接列出 Pre/Post-Norm、RoPE 外推或首 token/吞吐权衡；个人频率不采纳。
