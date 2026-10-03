# 推理、KV Cache 与量化

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 题目

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

**L1**

### 答案

KV cache 保存已处理 token 在各层的 K/V，prefill 建立前缀缓存，decode 只计算新 token 并追加。历史 Q 的注意力结果已用于原位置，未来通常不再用它查询，因此无需保存。复用依赖因果性：新 token 不改变历史位置表示；双向 attention 一般不能原样套用。

新 query 仍要读取历史 K/V，单步 attention 成本随上下文增长。缓存以显存换取重复计算减少，改变底座权重、LoRA、前缀或位置约定时须检查缓存是否仍有效。

### 易错点

- KV cache 能把每个生成 token 的 attention 复杂度变成常数。
- 把节省重算等同于减少模型总显存。

### 追问

- 为什么训练时通常不以同样方式使用 KV cache？
- 滑动窗口注意力能怎样限制缓存大小？

<a id="inf-002"></a>
## INF-002 · 怎样估算推理权重、KV cache 与总显存？

**L2**

### 答案

推理显存应拆为权重、KV cache、激活/工作区和框架预留。全注意力 KV 约为 $2BLTH_{\rm kv}d_hs$：$B$ 是并发序列数，$L$ 是层数，$T$ 是缓存长度，$H_{\rm kv}$ 是 KV 头数，$d_h$ 是头维，$s$ 是每元素字节；应使用 KV 头数而非 query 头数。

32 层、32 KV 头、头维 128、长度 4096、单序列 FP16 的 KV 为 2 GiB，改为 8 KV 头则为 0.5 GiB。70 亿参数 FP16 权重约 14 GB，即 13.0 GiB，仅指权重。请求长度不同时按 $\sum_iT_i$ 累加，并检查并行布局是否均分或复制 KV；分页、量化 scale、对齐与碎片都可能增加实际占用，应与实测峰值比较。

$$
M_{\rm KV}\approx2BLTH_{\rm kv}d_hs
$$

### 易错点

- 把 GB 和 GiB 混为一谈。
- 用 MHA 的 query 头数计算 GQA 的 KV 大小。

### 追问

- 多模态视觉 token 怎样计入 KV 长度？
- 为什么 4-bit 权重文件大小不等于线上显存？

<a id="inf-003"></a>
## INF-003 · MHA、MQA 与 GQA 的结构、KV 显存及速度有何差异？

**L1**

### 答案

MHA 的各 query 头有独立 K/V，MQA 的全部 query 头共用一组 K/V，GQA 按组共享，保留 query 头数 $H_q$。对应 $H_{\rm kv}=H_q$、$H_{\rm kv}=1$ 和 $1<H_{\rm kv}<H_q$，均匀分组通常要求 $H_q$ 能被 $H_{\rm kv}$ 整除；query 头 $i$ 使用组 $g(i)$，并非先平均 query。

典型形状为 `Q:[B,H_q,L_q,d_h]`、`K/V:[B,H_kv,L_k,d_h]`，每个 query 头单独匹配其组内 K/V，再拼接输出。逻辑上的重复映射可由内核直接实现，显式物理 repeat 可能抵消显存和带宽收益。

$N$ 层、总缓存 token 数 $T_{\rm total}$、每元素 $b$ 字节时，KV 约为 $2NT_{\rm total}H_{\rm kv}d_hb$。32 头改 8 头使理论 KV 降为 1/4，但 Q、输出投影与大部分 FFN 不随之下降。Decode 在小 Q、长 K 时常受 KV 读带宽限制；prefill 仍要处理各 query 头，速度还依赖 batch、内核、上下文与硬件。TP 超过 KV 头数时可能复制 KV，不能理想地继续均分。

已有 MHA 转换 GQA 可按组聚合 K/V 权重后再 uptraining，直接平均上线不能保证质量恢复。GQA 是容量与效率的折中，MQA 也可能有任务精度代价，应同时用任务评测、长上下文测试和服务压测验证。

$$
\begin{aligned}M_{\rm KV}&\approx2NT_{\rm total}H_{\rm kv}d_hb\\O_i&=\operatorname{softmax}\!\left(Q_iK_{g(i)}^\top/\sqrt{d_h}\right)V_{g(i)}\\H_q\bmod H_{\rm kv}&=0\end{aligned}
$$

### 易错点

- 将 GQA 写成减少全部 attention heads，或认为逻辑共享等于张量实现完全不复制。
- 用 KV 头比例直接预测端到端 latency、训练 FLOPs 或质量变化。

### 追问

- GQA 的 query-to-KV 分组在多 GPU 分片时如何保持一致？
- 同样的 KV 缓存预算下，增加并发和增加上下文长度怎样权衡？

<a id="inf-004"></a>
## INF-004 · prefill 与 decode 各在做什么，瓶颈为何不同？

**L2**

### 答案

Prefill 一次处理输入前缀并建立 KV cache，token 并行度较高，线性层能形成较大的 GEMM，稠密 attention 有长度二次项。Decode 通常每请求每步只处理一个新 token，反复读取权重和历史 KV，新 query 的 attention 计算约为 $O(Td)$。

前者常更偏计算，后者在小 batch 下常更偏带宽，但长上下文、batch 与硬件会改变瓶颈。扩大 batch 能摊薄权重读取、提高利用率，也增加排队、显存和请求间竞争。先用 profiler 区分权重/KV 带宽、算子启动和计算瓶颈，再选择量化、batching 或融合。

### 易错点

- prefill 永远算力瓶颈、decode 永远带宽瓶颈。
- 只看总 tokens/s 而不看输入输出长度。

### 追问

- chunked prefill 为什么有助于控制 decode 干扰？
- 如何用 roofline 判断量化是否可能加速？

<a id="inf-005"></a>
## INF-005 · FlashAttention 的核心思想是什么，会改变注意力结果吗？

**L2**

### 答案

FlashAttention 分块加载 Q/K/V，在片上 SRAM 计算局部分数，维护每行最大值与归一化累计量，以在线 softmax 累积输出，避免将完整 $T\times T$ 分数矩阵写回 HBM。它优化 IO 与中间存储，仍计算精确稠密注意力，算术复杂度为二次。

分块改变浮点求和次序，可产生小误差，并不保证逐 bit 相同，也不同于近似稀疏 attention。训练反向可重算部分分数和统计，以局部计算换更少保存与 IO。收益受长度、dtype、硬件、mask 和内核支持影响，短序列或不兼容场景不保证加速。

### 易错点

- 说它把稠密注意力 FLOPs 从 $O(T^2)$ 降到 $O(T)$。
- 把 FlashAttention 与 KV cache 视为同一技术。

### 追问

- 在线 softmax 合并两个块时为什么需要重缩放？
- FlashAttention-2 进一步优化了什么？

<a id="inf-006"></a>
## INF-006 · PagedAttention 与 FlashAttention 解决的问题有何不同？

**L2**

### 答案

PagedAttention 将 KV 分成块，通过逻辑块到不连续物理块的映射按需分配，减少最大窗口预留浪费和碎片。块表指导内核访问，块尺寸影响尾部浪费、映射开销与效率；共享前缀或分支可以用引用计数管理，并在修改时避免覆盖其他请求。

FlashAttention 优化注意力计算的 IO 和中间矩阵，PagedAttention 优化 KV 管理，两者可配合。分页不减少同样 K/V 的理论元素数，高并发仍可能耗尽容量，需要准入、抢占等调度策略。

### 易错点

- 分页可以让单序列无限增长且不占显存。
- 以论文倍数直接承诺任意服务吞吐提升。

### 追问

- 块尺寸如何影响尾部浪费？
- 共享块在分支生成时为什么可能需要 copy-on-write？

<a id="inf-007"></a>
## INF-007 · continuous batching 与普通 dynamic batching 有何区别？

**L2**

### 答案

普通 dynamic batching 常在开始处理请求前拼批，continuous batching 则在生成迭代间移除已完成请求、加入新请求，使空出的 batch 位置能立即利用，适合到达时间与生成长度不同的负载。

调度还要处理长 prefill 对 decode 的干扰、KV 容量和长请求公平性，可用单轮预算或分块 prefill 控制计算量。评估时固定请求率、输入/输出长度分布和 SLO，扫描负载比较吞吐与尾延迟，不能只看最大 batch 的吞吐。

### 易错点

- 把 continuous batching 简单解释为所有请求永远组成最大 batch。
- 只报告峰值 throughput，不报告排队时间。

### 追问

- 短请求如何避免被长请求拖累？
- batch size 上限和每轮 token budget 含义有何不同？

<a id="inf-008"></a>
## INF-008 · temperature、top-k 与 top-p 如何影响生成？

**L1**

### 答案

Temperature 用 $p_i\propto\exp(z_i/\tau)$ 调整分布尖锐程度，$\tau>0$ 时较小值通常更集中，$\tau=0$ 应按框架的贪心等特殊规则处理。Top-k 保留固定数量候选，top-p 保留累计概率达到阈值的最小候选集合，其大小随分布变化。

过滤后须重新归一化，多参数联用的执行顺序依实现而定。这些设置控制多样性，不保证事实正确；创作和可验证问答可用不同配置。固定种子有助于复现，但并行数值与运行环境仍可能使贪心或采样结果不能跨环境逐 token 一致。

### 易错点

- `temperature=0` 就绝对不会出现幻觉。
- 把 top-p 当作独立地保留概率大于 p 的 token。

### 追问

- 为什么 beam search 在开放生成中可能重复？
- top-p 与 temperature 的顺序为什么重要？

<a id="inf-009"></a>
## INF-009 · PTQ、QAT、W4A16 与 per-group quantization 是什么？

**L1**

### 答案

PTQ 在训练后量化，QAT 在训练中模拟或考虑量化误差。W4A16 表示权重约 4 bit、激活约 16 bit，不表示全部算子都使用 INT4。仿射量化常取 $q=\operatorname{clip}(\operatorname{round}(x/s)+z)$，反量化为 $\hat x=s(q-z)$，其中 $s$ 为 scale，$z$ 为 zero-point。

Per-tensor、per-channel、per-group 分别是量化参数的共享粒度；组越小通常误差更低，也增加元数据和实现成本。总存储还含 scale、zero-point、未量化层与对齐，不能只按参数量乘位宽估算。校准要覆盖部署输入、长度和领域，速度也要结合硬件及内核实测。

$$
\begin{aligned}q&=\operatorname{clip}(\operatorname{round}(x/s)+z)\\\hat x&=s(q-z)\end{aligned}
$$

### 易错点

- 低 bit 一定比 FP16 更快。
- 把权重量化与 KV cache 量化混成同一项。

### 追问

- outlier 为什么会增大量化误差？
- 怎样选择校准集并评估任务精度损失？

<a id="inf-010"></a>
## INF-010 · GPTQ 为什么利用二阶信息进行逐层量化？

**L2**

### 答案

GPTQ 用校准输入 $X$ 形成近似二阶信息，逐步量化权重，并补偿尚未量化权重的误差，目标是减少原层 $WX$ 与量化层 $\hat WX$ 的输出差异。输入二阶统计反映权重误差对输出的敏感性，比独立 round-to-nearest 更有信息。

工程上按列或块处理和更新以控制成本，通常属于低 bit 权重 PTQ。它仍是有损压缩，效果依赖校准覆盖、分组和阻尼；领域、长上下文或多模态校准错配须通过端到端回归检查。

### 易错点

- GPTQ 会更新底座进行完整反向训练。
- 把论文中某模型的近乎无损结果推广到所有位宽和场景。

### 追问

- 校准 Hessian 病态时为什么可能需要 damping？
- GPTQ 与普通舍入怎样公平比较？

<a id="inf-011"></a>
## INF-011 · AWQ 怎样保护重要权重，是否把 1% 权重保留为 FP16？

**L2**

### 答案

AWQ 利用激活统计识别重要权重通道，通过放大对应权重、对激活逆缩放，在未量化时保持线性结果等价，降低低 bit 量化误差。敏感性要结合激活幅度，单看权重绝对值不足以判断。

Scale 由离线校准搜索，正式方案不需完整梯度训练。论文讨论少量显著权重的重要性，但正式 AWQ 避免硬件不友好的混合精度，因此不能简化为把 1% 权重保留 FP16。部署仍需有效的 weight-only 内核，并实测任务精度、长度与硬件速度。

### 易错点

- 把 AWQ 的启发性实验混成最终采用的混合精度方案。
- AWQ 只看权重统计，不看激活。

### 追问

- 为什么通道缩放能改变量化误差却不改变未量化输出？
- AWQ 与 GPTQ 的校准目标有何差别？

<a id="inf-012"></a>
## INF-012 · SmoothQuant 为什么把激活的量化困难迁移到权重？

**L2**

### 答案

SmoothQuant 针对激活少数大值通道造成的量化困难，用离线等价通道缩放压低激活、相应放大权重，让两者更适合 W8A8。线性层可写为 $XW=(XS^{-1})(SW)$，$S$ 是按输入通道定义的可逆正对角矩阵。

缩放前浮点函数等价，量化后仍有误差；平滑系数需平衡两边动态范围，压低一方也可能增大另一方的误差。缩放可吸收到相关参数，减少额外运行操作，实际效果取决于激活量化粒度、代表性校准长度和 INT8 硬件支持。

$$
XW=(XS^{-1})(SW)
$$

### 易错点

- 浮点等价变换就意味着量化后零误差。
- 把 SmoothQuant 一概称为 W4A16 权重量化。

### 追问

- 为什么逐输入通道激活 scale 不一定适合常规 INT8 GEMM？
- 部署输入出现新 outlier 时怎么办？

<a id="inf-013"></a>
## INF-013 · 投机解码怎样保证目标模型的采样分布？

**L3**

### 答案

投机解码用便宜的 draft 模型提出多个 token，由 target 并行验证并执行接受/修正采样。对草稿分布 $q$ 提出的候选 $x$，以 $\min(1,p(x)/q(x))$ 接受；首次拒绝后从归一化的 $\max(p-q,0)$ 修正分布采样，不能直接从 $p$ 重采仍套用相同分布证明。全部接受时通常还能从 target 下一位置分布采额外 token。

标准算法保持目标模型分布，不要求每次随机运行文本完全相同。Tokenizer、采样变换与约束需协调；速度由接受率、draft 成本及验证开销决定，草稿越长不一定越快，低接受率或大 batch 可能收益不足。

$$
\begin{aligned}a(x)&=\min\!\left(1,\frac{p(x)}{q(x)}\right)\\p_{\rm correction}(x)&=\frac{\max(p(x)-q(x),0)}{\sum_y\max(p(y)-q(y),0)}\end{aligned}
$$

### 易错点

- 投机解码靠牺牲目标模型精度换速度。
- 分布一致就宣称固定种子下文本逐 token 必然一致。

### 追问

- 怎样根据接受率选择草稿长度？
- 贪心验证与随机采样验证的规则有什么区别？

<a id="inf-014"></a>
## INF-014 · prefix caching 与普通 KV cache 有何区别？

**L2**

### 答案

普通 KV cache 复用同一请求的历史，prefix caching 则在请求间复用完全相同前缀的 KV，主要节省重复 prefill。缓存键需包含父块上下文和当前 token，以及模型、LoRA、相关预处理等信息，防止同样局部文本在不同历史下误共享；语义相似不等于精确 KV 相同。

权重或适配器变化须失效或隔离缓存，多模态占位 token 相同但图像不同，还需模态内容哈希。缓存占用会与活跃请求竞争，需要淘汰；多租户共享应考虑隔离和时间侧信道。

### 易错点

- 只用当前文本片段哈希，不考虑前缀上下文。
- prefix cache 会加速输出 token 的全部 decode 计算。

### 追问

- 相同 system prompt 放在最前面为什么更利于命中？
- 多适配器和多租户怎样隔离缓存？

<a id="inf-015"></a>
## INF-015 · 如何同时优化 TTFT、每 token 延迟与吞吐？

**L2**

### 答案

TTFT 是从请求发出到首 token 的时间，包含排队与 prefill；ITL 是相邻输出 token 的间隔，吞吐是单位时间产出。体验还受客户端流式传输影响，不能只测 prefill 内核时间。

评估时固定负载与输入/输出长度分布，记录 P50/P95/P99、失败率、请求率、并发和 token 数，区分服务端与客户端口径。扩大 batch 可提升利用率，也可能增加排队和 ITL，需联合权衡 chunked prefill、缓存与调度。逐步增加到达率找饱和点，并比较延迟 SLO 内的有效吞吐 goodput，而不只报告离线最大 tokens/s。

### 易错点

- 首 token 很快就断言整体响应快。
- 把输入吞吐与输出吞吐混用或省略长度分布。

### 追问

- 低并发与高并发应采用同一调度参数吗？
- 缓存命中率变化怎样影响 benchmark 公平性？

## 参考资料

- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/pdf/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [PyTorch SDPA — masks, shapes and GQA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [Transformers: Utilities for generation — sampling warpers](https://huggingface.co/docs/transformers/internal/generation_utils)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/pdf/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/pdf/2211.10438)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192)
- [Automatic Prefix Caching — vLLM v0.20.1](https://docs.vllm.ai/en/v0.20.1/design/prefix_caching/)
- [Metrics — vLLM v0.20.1](https://docs.vllm.ai/en/v0.20.1/design/metrics/)
