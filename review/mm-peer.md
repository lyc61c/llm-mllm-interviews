# 多模态分片的同伴审阅

审阅日期：2026-10-02。审阅方式：逐题检查机制/公式前提、来源对应与题目证据标签；本轮不直接修改其他分片。工程分片、core 推理/微调和 alignment_rag 对齐/评测题目未发现阻塞数学或机制错误；以下记录来源与访问标记的补强，以及已处理状态。

## engineering.json

审阅覆盖：COD-001–015、SYS-001–015、PRJ-001–010；另静态查看 coding/reference.py 与 coding/torch_primitives.py。没有另行重复跑根任务已经通过的测试。

### E-01 [P2] COD-014 的 KV/GQA 显存公式缺少直接技术出处

- 题目：COD-014。
- 现有答案公式与 512 MiB 示例计算正确，已说明忽略碎片、scale 和运行时工作区。
- 问题：唯一 primary ENG-P15 是 [vLLM Metrics](https://docs.vllm.ai/en/latest/design/metrics/)，可核对缓存使用指标，正文没有 GQA 的 KV 头数/张量维度说明，因此不能直接核对本题主要公式。
- 修法：保留公式，补 [vLLM Paged Attention](https://docs.vllm.ai/en/latest/design/paged_attention/) 中 key/value cache 形状说明，或 GQA 原始论文/官方模型实现；notes 说明字节数是编辑按 shape 与 dtype 推导。
- 核查：以 2×1×4096×32×8×128×2 = 536,870,912 bytes，确认 512 MiB。
- 处理状态：已解决。根任务已追加 ENG-P29 官方 Paged Attention；本轮重新读取 JSON，确认 COD-014 已引用该 source。

### E-02 [P2] SYS-011 的 prefix cache 算法与缓存键来源不充分

- 题目：SYS-011。
- 机制表述正确，且已区分复用 prefill 与继续 decode。
- 问题：仅引用 ENG-P15 Metrics，指标页可验证命中统计，却不充分说明前缀 hash、LoRA/图像内容和租户隔离。
- 修法：追加 [vLLM APC 功能说明](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) 与 [APC 设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching/)。本轮已读 Limits 与 hash 组成段，覆盖 prefill 收益边界、前缀 token、LoRA ID、图像 hash 和 cache salt。
- 答案无需改核心内容；建议补一条“缓存以完整 block 为单位”的版本限定细节。
- 处理状态：主要边界已解决。根任务已追加 ENG-P30 APC 功能页；本轮确认 SYS-011 已引用。缓存键的具体结构仍建议引用仓库已有 CORE-S065 [固定版本 APC 设计文档](https://docs.vllm.ai/en/v0.20.1/design/prefix_caching/)，ENG-P30 的 notes 已正确把版本/租户键细节标为编辑建议。

### E-03 [P2] SYS-013 的多模态成本建议只引通用 SRE

- 题目：SYS-013。
- 内容合理，正确指出压缩影响 OCR/细字与跨页任务。
- 问题：ENG-P16 监控与 ENG-P25 SLO 是通用系统资料，未直接支持动态视觉 token 与跨页文档处理机制。
- 修法：追加已核查的 MM-S019 [Qwen2.5-VL HTML 技术报告](https://arxiv.org/html/2502.13923v1)（可复用全仓库 ID），或 MM-S020 [InternVL 动态切图官方实现](https://internvl.readthedocs.io/en/latest/internvl2.0/quick_start.html)。保留 SRE 来源支持诊断和成本指标。
- 可加“按页面/区域的选择要一起评估检索漏召回”作为有效边界。
- 处理状态：已解决。根任务已追加 ENG-P31 [官方 Qwen2.5-VL 文档](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl)，本轮确认 SYS-013 已引用；processor 与分辨率具有直接技术出处。

### 已确认的易错点

- COD-003：缓存 query 使用绝对位置 offset；非方阵 causal mask 的陷阱表述正确。
- COD-004：RoPE 相对内积符号与相邻配对约定正确；没有混用 split-half 权重。
- COD-007：LoRA 的 A/B 形状、B=0 初始化时 A 首步零梯度、merge 等价前提正确。
- COD-008：DPO 取序列 logprob 求和、只算回答、reference 冻结以及 softplus 符号正确。
- COD-009：全同奖励只消除奖励优势，不保证 KL 等项零梯度；表述正确。
- COD-014：GQA 应使用 KV 头数；GB 与 GiB 已区分。
- SYS-003：不能平均各实例 P95；已说明负载和质量/SLO。
- SYS-005：超时不等于未提交，幂等重试与 deadline 表述正确。
- SYS-007：权限约束进入候选检索集合，不能只在输出链接层隐藏。
- SYS-011：复用前缀的 KV 与答案缓存区分正确；视觉内容/adapter 变化被纳入键。
- SYS-013：任务成本不能只按图像张数，视觉 encoder 与 decode 瓶颈已区分。

## core.json

审阅覆盖：INF-001–015 全部；FT-001–015 全部；额外抽查 TFM-003–005。对 KV 数值、LoRA 梯度、SmoothQuant 矩阵等价和投机解码接受/修正公式做独立核对。未发现需要修改核心答案的错误。

### C-01 [P2] INF-005 的在线 softmax/重算细节超出已读摘要

- 题目：INF-005；关联来源 CORE-S057。
- 答案关于精确稠密 attention、二次 FLOPs 与浮点求和顺序的边界正确。
- 问题：CORE-S057 明确只读摘要，摘要不能充分核对 detail 中逐行最大值、归一化累积和 backward 重算。
- 修法：以同一论文的 [PDF 第 3.1 节与 Algorithm 1](https://arxiv.org/pdf/2205.14135) 补核验，并将该 source 的 URL/access/notes 改为实际已读正文的范围；本轮已读第 3.1 节分块最大值与重缩放、recomputation 段，确认答案正确。不要把“原 Transformer 论文”作为在线 softmax 的支持来源。

### C-02 [P2] INF-008 的采样参数与 FT-009 的 Prompt-Tuning 应追加直接出处

- 题目：INF-008、FT-009。
- 问题：INF-008 只引 nucleus sampling 摘要，难以核对 temperature 正数约束、top-k 与具体 API 行为；FT-009 只引 Adapter 与 Prefix-Tuning 摘要，没有 Prompt-Tuning 本身的主要来源。
- 修法：INF-008 追加 [Transformers generation utilities](https://huggingface.co/docs/transformers/internal/generation_utils) 的 TemperatureLogitsWarper/TopKLogitsWarper/TopPLogitsWarper；FT-009 追加 [PEFT Prompt tuning](https://huggingface.co/docs/peft/main/en/package_reference/prompt_tuning)。本轮已读相关定义，支持现有答案；不建议引入官方文档示例中未经本仓库验证的“常用参数范围”。

### C-03 [P2] CORE-S059 应按“只读摘要”统一访问状态

- 题目：INF-004、INF-007、INF-015；关联来源 CORE-S059。
- 现状：access=full，但 notes 为“已读会议官方页的完整摘要……未通读 PDF”。
- 问题：与仓库其他“摘要→snippet”的标记规则不一致，容易让读者把官方会议网页可访问理解为论文正文已核验。
- 修法：将 access 标为 snippet，并保留明确 notes；若再阅读 [Orca 正文 PDF](https://www.usenix.org/system/files/osdi22-yu.pdf) 的 scheduling 部分，才改为 full/partial 并记录具体范围。iteration-level scheduling 在公开摘要中明确出现，本题主要机制有来源支持。

### 已确认的易错点

- INF-002：32 KV 头 FP16 示例为 2 GiB，8 KV 头为 0.5 GiB；7B FP16 约 14 GB/13.0 GiB。并发缓存按总 token 数计，TP 复制条件已说明。
- INF-003：GQA 保留 query 头，减少 KV 头，MHA→GQA mean pooling 配合 uptraining；没有把共享 KV 写成减少 query 数。
- INF-004/005：decode 单步长度项仍增长；FlashAttention 不把稠密算术复杂度降至线性。
- INF-009/010/011：低位权重存储不等于低位算术；GPTQ 是校准重构与误差补偿；AWQ 正式方法不等于直接留下 1% FP16 权重。
- INF-012：XW=(XS⁻¹)(SW) 的矩阵维度与可逆正对角缩放前提正确；等价仅针对未量化运算。
- INF-013：min(1,p/q) 接受，首次拒绝后 norm(max(p−q,0))；不能直接从 p 重采仍沿用原分布证明。
- INF-014：父块上下文与多模态内容 hash 均纳入缓存；已正确区分 prefill 与 decode。
- FT-002：loss mask 不删掉 prompt attention，EOS 与 padding 按位置区别；避免重复 label shift。
- FT-005/006：LoRA 形状与 ∂L/∂A、∂L/∂B 梯度正确，双零初始化纯 BA 分支无法启动。
- FT-008/013/014：QLoRA 激活与工作区仍占显存；变长梯度累积按有效 token 总量归一化；activation checkpoint 不减少优化器状态。

## alignment_rag.json

审阅覆盖：ALN-001–020 与 EVA-001–015 全部。逐项检查 PPO、GAE、KL、DPO、GRPO、Dr. GRPO 与 pass@k 前提；另阅读 DeepSeekMath §4.1.2、Dr. GRPO §3.1–3.2、HumanEval §3.1/Appendix A 核对机制。没有发现阻塞公式或机制错误。

### A-01 [P2] ALN-001 的 DPO 部分应有独立 primary

- 题目：ALN-001。
- 问题：当前只引 APP-S101 InstructGPT；该文可支持 SFT/RM/PPO 流程，但不能作为后来 DPO 算法的原始出处。
- 修法：追加仓库已核验的 APP-S104 [DPO 原始论文](https://arxiv.org/html/2305.18290v3)，无需改核心答案。ALN-007 对 β 理论/实训区别也可复用该 source 与现有 TRL 文档。

### A-02 [P2] EVA-008 的“过度拒答/合法对照”应补直接研究

- 题目：EVA-008。
- 问题：目前唯一 APP-S148 HarmBench 主要支持有害行为与红队攻击评测；本题的一半是合法相似请求的误拒，以及安全/不安全对照，其直接出处不充分。
- 修法：保留 HarmBench，追加 [XSTest](https://arxiv.org/abs/2308.01263)。本轮只读作者摘要，应标 access=snippet；摘要明确说明安全提示与不安全对照，可直接支撑“语义判断而非关键词拦截”的核心建议。不要据摘要声称已核验全文评分细节。

### A-03 [P3] EVA-014 可补无偏估计的采样前提

- 题目：EVA-014。
- 当前公式与 n−c<k 边界正确，且正确指出不能把经验 pass@1 直接代入当无偏估计。
- 可读性问题：只写“固定采样和测试预算”，还未明确 n 个样本来自同一固定分布的独立采样；论文 Appendix A 的无偏推导使用 c~Binom(n,p)。
- 修法：在 detail 加“无偏结论对应固定生成分布的独立采样；去重、按失败反馈自适应搜索或互相依赖的候选应单独定义评测目标，不能直接沿用 iid 证明”。仍保留 oracle pass@k 与实际候选选择成功率的区别。

### 已确认的易错点

- ALN-002：BT 分差平移不变，奖励绝对零点不可由同 prompt 的偏好唯一识别。
- ALN-003：正负优势的 clip/min 方向正确，不把 clip 误写为严格概率比约束。
- ALN-004/005：GAE 的 γ/λ 区分清楚；reference KL 与 PPO old policy 分工不同，单 token log-ratio 可为负。
- ALN-006/007：最优策略参考分布乘 exp(r/β)，BT 下 Z(x) 抵消；没有把较大 β 机械解释为较小实训梯度。
- ALN-010/011：原始 outcome GRPO 无独立 critic，仍有 baseline；全同奖励没有相对信号，ε 不创造信号。
- ALN-012：短正答与长错答受不同的长度重加权；Dr. GRPO 移除两项归一化。与[论文正文 §3.1–3.2](https://arxiv.org/html/2503.20783v2)对应，且没有声称对一切目标无偏。
- ALN-013/014/015：结果验证不保证过程正确；PRM 训练与排序用途有区别；KL/换 judge 不保证消除奖励投机。
- ALN-019：未把所有 DPO 变体固定称为离线流程。
- EVA-004：使用逐题配对差值/cluster 重采样；没有用两个独立区间重叠替代差值检验。
- EVA-006/007：真实性与上下文忠实分开；自报置信与 token 概率不直接等于事实可信度。
- EVA-011：按感知、OCR、关系与推理消融；综合榜单不代表全部模态能力。
- EVA-014：[HumanEval 论文 Eq.1 与 Appendix A](https://arxiv.org/pdf/2107.03374) 支持组合数估计与经验代入偏差；公式没有遗漏 n≥k。
- EVA-015：点击率并非 correctness，版本化 trace 与独立验收集表述正确。
