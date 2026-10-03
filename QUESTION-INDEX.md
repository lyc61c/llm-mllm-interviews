# 题目索引

[返回首页](README.md)

支持按题号定位；全文搜索与筛选请打开 [离线题库](index.html)。

| 题号 | 类别 | 难度 | 题目 | 题目线索 |
|---|---|---|---|---|
| TFM-001 | Transformer 与数学基础 | L1 | [缩放点积注意力如何计算，为什么除以 √d_k？](chapters/01-transformer.md#tfm-001) | 社区线索 |
| TFM-002 | Transformer 与数学基础 | L1 | [多头注意力与单头注意力有什么区别？](chapters/01-transformer.md#tfm-002) | 编辑补充 |
| TFM-003 | Transformer 与数学基础 | L1 | [掩码注意力如何实现？causal、padding 与 loss mask 有何区别？](chapters/01-transformer.md#tfm-003) | 编辑补充 |
| TFM-004 | Transformer 与数学基础 | L1 | [Encoder-only、Decoder-only 与 Encoder-Decoder 怎样选择？](chapters/01-transformer.md#tfm-004) | 社区线索 |
| TFM-005 | Transformer 与数学基础 | L1 | [Transformer 为什么常用 LayerNorm，而不是 BatchNorm？](chapters/01-transformer.md#tfm-005) | 社区线索 |
| TFM-006 | Transformer 与数学基础 | L2 | [RMSNorm 与 LayerNorm 的公式和性质有什么区别？](chapters/01-transformer.md#tfm-006) | 社区线索 |
| TFM-007 | Transformer 与数学基础 | L2 | [Pre-Norm 与 Post-Norm 如何影响训练稳定性？](chapters/01-transformer.md#tfm-007) | 社区线索 |
| TFM-008 | Transformer 与数学基础 | L2 | [RoPE 如何表达相对位置，怎样与 KV cache 正确配合？](chapters/01-transformer.md#tfm-008) | 社区线索 |
| TFM-009 | Transformer 与数学基础 | L2 | [为什么不能只把 max_position_embeddings 改大来扩展上下文？](chapters/01-transformer.md#tfm-009) | 社区线索 |
| TFM-010 | Transformer 与数学基础 | L2 | [FFN 提供什么作用？SwiGLU 为什么常调整中间维度？](chapters/01-transformer.md#tfm-010) | 社区线索 |
| TFM-011 | Transformer 与数学基础 | L2 | [Transformer 一层的时间、空间复杂度如何估算？](chapters/01-transformer.md#tfm-011) | 社区线索 |
| TFM-012 | Transformer 与数学基础 | L1 | [残差连接为什么能帮助深层模型训练？](chapters/01-transformer.md#tfm-012) | 编辑补充 |
| TFM-013 | Transformer 与数学基础 | L1 | [交叉熵、KL 散度与 perplexity 的关系和实现注意点是什么？](chapters/01-transformer.md#tfm-013) | 社区线索 |
| TFM-014 | Transformer 与数学基础 | L2 | [输入 embedding 与输出 LM head 权重共享有什么利弊？](chapters/01-transformer.md#tfm-014) | 编辑补充 |
| TFM-015 | Transformer 与数学基础 | L2 | [MoE 与 Dense 的参数量和计算量应怎样比较？](chapters/01-transformer.md#tfm-015) | 社区线索 |
| TFM-016 | Transformer 与数学基础 | L1 | [Cross-attention 与 self-attention 有何区别，Q/K/V 从哪里来？](chapters/01-transformer.md#tfm-016) | 编辑补充 |
| TFM-017 | Transformer 与数学基础 | L1 | [熵衡量什么？离散熵、条件熵与模型输出熵怎样区分？](chapters/01-transformer.md#tfm-017) | 编辑补充 |
| TFM-018 | Transformer 与数学基础 | L2 | [矩阵的秩与特征值如何计算，和奇异值有什么关系？](chapters/01-transformer.md#tfm-018) | 编辑补充 |
| TFM-019 | Transformer 与数学基础 | L1 | [Transformer 为什么需要位置编码？绝对、相对、RoPE 与 ALiBi 怎样比较？](chapters/01-transformer.md#tfm-019) | 编辑补充 |
| TFM-020 | Transformer 与数学基础 | L1 | [Dropout 如何正则化？原始 Transformer 把它放在哪里？](chapters/01-transformer.md#tfm-020) | 编辑补充 |
| TFM-021 | Transformer 与数学基础 | L2 | [Attention 权重如何学到？权重较大就能解释模型决策吗？](chapters/01-transformer.md#tfm-021) | 编辑补充 |
| TFM-022 | Transformer 与数学基础 | L1 | [Transformer 为什么适合建模长距离依赖？O(1) 路径意味着什么？](chapters/01-transformer.md#tfm-022) | 编辑补充 |
| TFM-023 | Transformer 与数学基础 | L2 | [所谓 Negative Attention 是什么？低权重、负 logit 与负输出有何区别？](chapters/01-transformer.md#tfm-023) | 编辑补充 |
| PRE-001 | 预训练、数据与优化 | L1 | [自回归预训练的 next-token loss 怎样计算？](chapters/02-pretraining.md#pre-001) | 社区线索 |
| PRE-002 | 预训练、数据与优化 | L1 | [BPE、Unigram 与 SentencePiece 分别是什么？](chapters/02-pretraining.md#pre-002) | 社区线索 |
| PRE-003 | 预训练、数据与优化 | L2 | [词表越大越好吗，扩词表有哪些代价？](chapters/02-pretraining.md#pre-003) | 社区线索 |
| PRE-004 | 预训练、数据与优化 | L2 | [Scaling Law 描述的是什么，能直接预测下游能力吗？](chapters/02-pretraining.md#pre-004) | 社区线索 |
| PRE-005 | 预训练、数据与优化 | L2 | [Chinchilla 的结论是什么，tokens≈20×参数是硬规则吗？](chapters/02-pretraining.md#pre-005) | 编辑补充 |
| PRE-006 | 预训练、数据与优化 | L2 | [大规模预训练语料应怎样清洗？](chapters/02-pretraining.md#pre-006) | 编辑补充 |
| PRE-007 | 预训练、数据与优化 | L2 | [精确去重与近似去重有哪些方法和边界？](chapters/02-pretraining.md#pre-007) | 社区线索 |
| PRE-008 | 预训练、数据与优化 | L2 | [怎样检测与减少 benchmark contamination？](chapters/02-pretraining.md#pre-008) | 编辑补充 |
| PRE-009 | 预训练、数据与优化 | L2 | [代码、网页、多语言等预训练数据如何配比？](chapters/02-pretraining.md#pre-009) | 编辑补充 |
| PRE-010 | 预训练、数据与优化 | L2 | [SFT 数据 packing 怎样提高效率并保持跨样本隔离？](chapters/02-pretraining.md#pre-010) | 编辑补充 |
| PRE-011 | 预训练、数据与优化 | L1 | [继续预训练与 SFT 的数据和目标有什么区别？](chapters/02-pretraining.md#pre-011) | 编辑补充 |
| PRE-012 | 预训练、数据与优化 | L2 | [AdamW 与 Adam 加 L2 正则为什么不等价？](chapters/02-pretraining.md#pre-012) | 编辑补充 |
| PRE-013 | 预训练、数据与优化 | L1 | [FP16 与 BF16 的差异是什么，混合精度为何有用？](chapters/02-pretraining.md#pre-013) | 编辑补充 |
| PRE-014 | 预训练、数据与优化 | L2 | [warmup、学习率衰减和梯度裁剪各解决什么问题？](chapters/02-pretraining.md#pre-014) | 编辑补充 |
| PRE-015 | 预训练、数据与优化 | L1 | [perplexity 怎样计算才可公平比较？](chapters/02-pretraining.md#pre-015) | 社区线索 |
| FT-001 | SFT、LoRA 与参数高效微调 | L1 | [预训练与 SFT 的 loss 有何差异，label shift 如何对齐？](chapters/03-finetuning.md#ft-001) | 社区线索 |
| FT-002 | SFT、LoRA 与参数高效微调 | L1 | [SFT 怎样只对回答部分计算 loss？](chapters/03-finetuning.md#ft-002) | 社区线索 |
| FT-003 | SFT、LoRA 与参数高效微调 | L2 | [chat template 错配会导致哪些问题？](chapters/03-finetuning.md#ft-003) | 编辑补充 |
| FT-004 | SFT、LoRA 与参数高效微调 | L1 | [全参数微调、LoRA、Adapter 和 Prefix-Tuning 怎样选择？](chapters/03-finetuning.md#ft-004) | 社区线索 |
| FT-005 | SFT、LoRA 与参数高效微调 | L1 | [LoRA 的低秩更新公式及可训练参数量是什么？](chapters/03-finetuning.md#ft-005) | 社区线索 |
| FT-006 | SFT、LoRA 与参数高效微调 | L2 | [LoRA 怎样初始化，r、alpha 与 dropout 各控制什么？](chapters/03-finetuning.md#ft-006) | 社区线索 |
| FT-007 | SFT、LoRA 与参数高效微调 | L2 | [LoRA 的 rank、alpha、dropout 和 target_modules 应怎么调？](chapters/03-finetuning.md#ft-007) | 社区线索 |
| FT-008 | SFT、LoRA 与参数高效微调 | L2 | [LoRA 与 QLoRA 有何区别，NF4、双重量化与分页优化器做什么？](chapters/03-finetuning.md#ft-008) | 社区线索 |
| FT-009 | SFT、LoRA 与参数高效微调 | L2 | [Adapter、Prompt-Tuning 与 Prefix-Tuning 有何差别？](chapters/03-finetuning.md#ft-009) | 编辑补充 |
| FT-010 | SFT、LoRA 与参数高效微调 | L2 | [SFT 数据量越大越好吗，怎样构建高质量指令集？](chapters/03-finetuning.md#ft-010) | 编辑补充 |
| FT-011 | SFT、LoRA 与参数高效微调 | L2 | [用模型生成 SFT 数据怎样避免错误与同质化？](chapters/03-finetuning.md#ft-011) | 编辑补充 |
| FT-012 | SFT、LoRA 与参数高效微调 | L2 | [微调后的灾难性遗忘怎样发现和缓解？](chapters/03-finetuning.md#ft-012) | 编辑补充 |
| FT-013 | SFT、LoRA 与参数高效微调 | L2 | [梯度累积等价于大 batch 吗？变长样本怎么归一化？](chapters/03-finetuning.md#ft-013) | 编辑补充 |
| FT-014 | SFT、LoRA 与参数高效微调 | L2 | [gradient checkpointing 节省什么，为何会变慢？](chapters/03-finetuning.md#ft-014) | 编辑补充 |
| FT-015 | SFT、LoRA 与参数高效微调 | L2 | [SFT loss 下降但任务效果变差，应该怎样排查？](chapters/03-finetuning.md#ft-015) | 编辑补充 |
| INF-001 | 推理、KV Cache 与量化 | L1 | [KV cache 缓存什么，为什么通常不缓存历史 Q？](chapters/04-inference.md#inf-001) | 编辑补充 |
| INF-002 | 推理、KV Cache 与量化 | L2 | [怎样估算推理权重、KV cache 与总显存？](chapters/04-inference.md#inf-002) | 社区线索 |
| INF-003 | 推理、KV Cache 与量化 | L1 | [MHA、MQA 与 GQA 的结构、KV 显存及速度有何差异？](chapters/04-inference.md#inf-003) | 社区线索 |
| INF-004 | 推理、KV Cache 与量化 | L2 | [prefill 与 decode 各在做什么，瓶颈为何不同？](chapters/04-inference.md#inf-004) | 编辑补充 |
| INF-005 | 推理、KV Cache 与量化 | L2 | [FlashAttention 的核心思想是什么，会改变注意力结果吗？](chapters/04-inference.md#inf-005) | 社区线索 |
| INF-006 | 推理、KV Cache 与量化 | L2 | [PagedAttention 与 FlashAttention 解决的问题有何不同？](chapters/04-inference.md#inf-006) | 编辑补充 |
| INF-007 | 推理、KV Cache 与量化 | L2 | [continuous batching 与普通 dynamic batching 有何区别？](chapters/04-inference.md#inf-007) | 编辑补充 |
| INF-008 | 推理、KV Cache 与量化 | L1 | [temperature、top-k 与 top-p 如何影响生成？](chapters/04-inference.md#inf-008) | 编辑补充 |
| INF-009 | 推理、KV Cache 与量化 | L1 | [PTQ、QAT、W4A16 与 per-group quantization 是什么？](chapters/04-inference.md#inf-009) | 社区线索 |
| INF-010 | 推理、KV Cache 与量化 | L2 | [GPTQ 为什么利用二阶信息进行逐层量化？](chapters/04-inference.md#inf-010) | 社区线索 |
| INF-011 | 推理、KV Cache 与量化 | L2 | [AWQ 怎样保护重要权重，是否把 1% 权重保留为 FP16？](chapters/04-inference.md#inf-011) | 编辑补充 |
| INF-012 | 推理、KV Cache 与量化 | L2 | [SmoothQuant 为什么把激活的量化困难迁移到权重？](chapters/04-inference.md#inf-012) | 编辑补充 |
| INF-013 | 推理、KV Cache 与量化 | L3 | [投机解码怎样保证目标模型的采样分布？](chapters/04-inference.md#inf-013) | 社区线索 |
| INF-014 | 推理、KV Cache 与量化 | L2 | [prefix caching 与普通 KV cache 有何区别？](chapters/04-inference.md#inf-014) | 编辑补充 |
| INF-015 | 推理、KV Cache 与量化 | L2 | [如何同时优化 TTFT、每 token 延迟与吞吐？](chapters/04-inference.md#inf-015) | 社区线索 |
| ALN-001 | RLHF、DPO、PPO 与 GRPO | L1 | [SFT、RLHF 与 DPO 分别解决什么问题？](chapters/05-alignment.md#aln-001) | 社区线索 |
| ALN-002 | RLHF、DPO、PPO 与 GRPO | L2 | [奖励模型如何用成对偏好训练？](chapters/05-alignment.md#aln-002) | 编辑补充 |
| ALN-003 | RLHF、DPO、PPO 与 GRPO | L2 | [PPO 的概率比、clip 和 min 分别起什么作用？](chapters/05-alignment.md#aln-003) | 社区线索 |
| ALN-004 | RLHF、DPO、PPO 与 GRPO | L2 | [GAE 如何计算，λ 与 γ 如何影响优势估计？](chapters/05-alignment.md#aln-004) | 社区线索 |
| ALN-005 | RLHF、DPO、PPO 与 GRPO | L2 | [RLHF 的 KL 惩罚与 PPO 新旧策略约束有什么区别？](chapters/05-alignment.md#aln-005) | 社区线索 |
| ALN-006 | RLHF、DPO、PPO 与 GRPO | L3 | [DPO 的损失如何从 KL 正则化 RLHF 目标推出？](chapters/05-alignment.md#aln-006) | 社区线索 |
| ALN-007 | RLHF、DPO、PPO 与 GRPO | L2 | [DPO 的 β 和参考模型如何理解与调参？](chapters/05-alignment.md#aln-007) | 编辑补充 |
| ALN-008 | RLHF、DPO、PPO 与 GRPO | L2 | [PPO 与 DPO 在工程上如何选型？](chapters/05-alignment.md#aln-008) | 社区线索 |
| ALN-009 | RLHF、DPO、PPO 与 GRPO | L2 | [如何把点赞、点踩和日志变成高质量偏好数据？](chapters/05-alignment.md#aln-009) | 社区线索 |
| ALN-010 | RLHF、DPO、PPO 与 GRPO | L2 | [GRPO 为什么不需要独立价值模型？](chapters/05-alignment.md#aln-010) | 社区线索 |
| ALN-011 | RLHF、DPO、PPO 与 GRPO | L3 | [GRPO 组内标准差归一化带来哪些问题？](chapters/05-alignment.md#aln-011) | 编辑补充 |
| ALN-012 | RLHF、DPO、PPO 与 GRPO | L3 | [GRPO 的长度偏差与 Dr. GRPO 有什么关系？](chapters/05-alignment.md#aln-012) | 编辑补充 |
| ALN-013 | RLHF、DPO、PPO 与 GRPO | L2 | [RLVR 的可验证奖励如何设计？](chapters/05-alignment.md#aln-013) | 编辑补充 |
| ALN-014 | RLHF、DPO、PPO 与 GRPO | L2 | [ORM 与 PRM 的区别和信用分配难点是什么？](chapters/05-alignment.md#aln-014) | 社区线索 |
| ALN-015 | RLHF、DPO、PPO 与 GRPO | L2 | [如何识别和缓解 reward hacking？](chapters/05-alignment.md#aln-015) | 编辑补充 |
| ALN-016 | RLHF、DPO、PPO 与 GRPO | L2 | [后训练为什么会出现对齐税或遗忘？](chapters/05-alignment.md#aln-016) | 社区线索 |
| ALN-017 | RLHF、DPO、PPO 与 GRPO | L2 | [RLAIF 和 Constitutional AI 如何工作？](chapters/05-alignment.md#aln-017) | 编辑补充 |
| ALN-018 | RLHF、DPO、PPO 与 GRPO | L3 | [IPO 等 DPO 变种主要试图解决什么问题？](chapters/05-alignment.md#aln-018) | 社区线索 |
| ALN-019 | RLHF、DPO、PPO 与 GRPO | L2 | [离线偏好优化和在线 RL 的分布差异是什么？](chapters/05-alignment.md#aln-019) | 社区线索 |
| ALN-020 | RLHF、DPO、PPO 与 GRPO | L3 | [多目标奖励发生冲突时如何处理？](chapters/05-alignment.md#aln-020) | 社区线索 |
| RAG-001 | RAG、检索与重排 | L1 | [一个可落地的 RAG 系统有哪些环节？](chapters/06-rag.md#rag-001) | 社区线索 |
| RAG-002 | RAG、检索与重排 | L1 | [RAG、微调和 prompt engineering 如何选择？](chapters/06-rag.md#rag-002) | 社区线索 |
| RAG-003 | RAG、检索与重排 | L2 | [chunk 大小与 overlap 怎样设置？](chapters/06-rag.md#rag-003) | 社区线索 |
| RAG-004 | RAG、检索与重排 | L2 | [embedding 模型与相似度应如何选？](chapters/06-rag.md#rag-004) | 编辑补充 |
| RAG-005 | RAG、检索与重排 | L2 | [BM25 与向量检索各有什么优势？](chapters/06-rag.md#rag-005) | 编辑补充 |
| RAG-006 | RAG、检索与重排 | L2 | [混合检索的分数融合与 RRF 有什么区别？](chapters/06-rag.md#rag-006) | 社区线索 |
| RAG-007 | RAG、检索与重排 | L2 | [reranker 和 embedding 检索模型如何分工？](chapters/06-rag.md#rag-007) | 社区线索 |
| RAG-008 | RAG、检索与重排 | L2 | [HNSW 的 M、ef_construction 和 ef_search 怎样权衡？](chapters/06-rag.md#rag-008) | 编辑补充 |
| RAG-009 | RAG、检索与重排 | L2 | [query rewrite、multi-query 和 HyDE 何时有用？](chapters/06-rag.md#rag-009) | 社区线索 |
| RAG-010 | RAG、检索与重排 | L3 | [Self-RAG 和多跳检索怎样改善复杂问答？](chapters/06-rag.md#rag-010) | 编辑补充 |
| RAG-011 | RAG、检索与重排 | L2 | [RAG 如何建立分层评测并定位 bad case？](chapters/06-rag.md#rag-011) | 社区线索 |
| RAG-012 | RAG、检索与重排 | L2 | [RAG 为什么仍会幻觉，怎样设计引用与拒答？](chapters/06-rag.md#rag-012) | 社区线索 |
| RAG-013 | RAG、检索与重排 | L3 | [企业 RAG 如何处理权限、更新与删除？](chapters/06-rag.md#rag-013) | 编辑补充 |
| RAG-014 | RAG、检索与重排 | L2 | [长上下文能否替代 RAG？](chapters/06-rag.md#rag-014) | 社区线索 |
| RAG-015 | RAG、检索与重排 | L3 | [多模态 RAG 怎样检索图表、扫描 PDF 与视频？](chapters/06-rag.md#rag-015) | 社区线索 |
| AGT-001 | Agent、工具与上下文工程 | L1 | [ReAct、固定工作流和 Agent 有什么区别？](chapters/07-agents.md#agt-001) | 社区线索 |
| AGT-002 | Agent、工具与上下文工程 | L2 | [如何让 function calling 更可靠？](chapters/07-agents.md#agt-002) | 社区线索 |
| AGT-003 | Agent、工具与上下文工程 | L2 | [Agent 工具超时、重试与幂等怎么设计？](chapters/07-agents.md#agt-003) | 社区线索 |
| AGT-004 | Agent、工具与上下文工程 | L2 | [短期记忆、长期记忆与 checkpoint 分别是什么？](chapters/07-agents.md#agt-004) | 社区线索 |
| AGT-005 | Agent、工具与上下文工程 | L3 | [多 Agent 的通信与共享状态如何设计？](chapters/07-agents.md#agt-005) | 社区线索 |
| AGT-006 | Agent、工具与上下文工程 | L2 | [MCP 与模型 function calling 是什么关系？](chapters/07-agents.md#agt-006) | 社区线索 |
| AGT-007 | Agent、工具与上下文工程 | L3 | [如何防御工具结果和检索材料中的 prompt injection？](chapters/07-agents.md#agt-007) | 编辑补充 |
| AGT-008 | Agent、工具与上下文工程 | L2 | [Agent 应怎样评测，为什么不能只看最终回答？](chapters/07-agents.md#agt-008) | 编辑补充 |
| AGT-009 | Agent、工具与上下文工程 | L2 | [如何防止 Agent 死循环和无效规划？](chapters/07-agents.md#agt-009) | 编辑补充 |
| AGT-010 | Agent、工具与上下文工程 | L2 | [上下文工程怎样降低长任务成本而保持信息？](chapters/07-agents.md#agt-010) | 社区线索 |
| EVA-001 | 评测、幻觉与安全 | L2 | [如何避免训练数据污染评测集？](chapters/08-evaluation.md#eva-001) | 编辑补充 |
| EVA-002 | 评测、幻觉与安全 | L1 | [怎样评价文本生成质量？BLEU/ROUGE、PPL与人工评测如何组合？](chapters/08-evaluation.md#eva-002) | 社区线索 |
| EVA-003 | 评测、幻觉与安全 | L2 | [LLM-as-a-Judge 有哪些偏差，如何校准？](chapters/08-evaluation.md#eva-003) | 社区线索 |
| EVA-004 | 评测、幻觉与安全 | L2 | [两个模型只差一个百分点，怎样判断提升可靠？](chapters/08-evaluation.md#eva-004) | 编辑补充 |
| EVA-005 | 评测、幻觉与安全 | L2 | [如何设计人工评测，降低标注分歧？](chapters/08-evaluation.md#eva-005) | 社区线索 |
| EVA-006 | 评测、幻觉与安全 | L2 | [如何区分和测量事实幻觉与上下文不忠实？](chapters/08-evaluation.md#eva-006) | 编辑补充 |
| EVA-007 | 评测、幻觉与安全 | L3 | [模型自报置信度可靠吗，怎样设计拒答阈值？](chapters/08-evaluation.md#eva-007) | 编辑补充 |
| EVA-008 | 评测、幻觉与安全 | L2 | [安全评测如何兼顾越狱成功率与过度拒答？](chapters/08-evaluation.md#eva-008) | 编辑补充 |
| EVA-009 | 评测、幻觉与安全 | L3 | [如何构建红队测试而不只刷固定攻击集？](chapters/08-evaluation.md#eva-009) | 编辑补充 |
| EVA-010 | 评测、幻觉与安全 | L2 | [训练数据泄露与 RAG 越权泄露如何区分？](chapters/08-evaluation.md#eva-010) | 编辑补充 |
| EVA-011 | 评测、幻觉与安全 | L2 | [MLLM 评测如何区分感知、OCR 与推理错误？](chapters/08-evaluation.md#eva-011) | 编辑补充 |
| EVA-012 | 评测、幻觉与安全 | L2 | [怎样评估 prompt 扰动、多语言和分布外鲁棒性？](chapters/08-evaluation.md#eva-012) | 编辑补充 |
| EVA-013 | 评测、幻觉与安全 | L2 | [如何验证 prompt 优化确实有效？](chapters/08-evaluation.md#eva-013) | 编辑补充 |
| EVA-014 | 评测、幻觉与安全 | L3 | [代码生成的 pass@k 是什么，如何避免错误估计？](chapters/08-evaluation.md#eva-014) | 编辑补充 |
| EVA-015 | 评测、幻觉与安全 | L2 | [线上如何监控、验收并回归 LLM/Agent 系统？](chapters/08-evaluation.md#eva-015) | 编辑补充 |
| VLM-001 | 视觉语言与图文多模态 | L1 | [CLIP 的结构和双向对比损失是什么？](chapters/09-vision-language.md#vlm-001) | 社区线索 |
| VLM-002 | 视觉语言与图文多模态 | L1 | [CLIP 如何做零样本分类？提示模板有什么作用？](chapters/09-vision-language.md#vlm-002) | 编辑补充 |
| VLM-003 | 视觉语言与图文多模态 | L2 | [图文检索很好，为什么关系和计数题仍可能答错？](chapters/09-vision-language.md#vlm-003) | 编辑补充 |
| VLM-004 | 视觉语言与图文多模态 | L2 | [SigLIP 与 CLIP 的损失主要区别是什么？](chapters/09-vision-language.md#vlm-004) | 编辑补充 |
| VLM-005 | 视觉语言与图文多模态 | L1 | [BLIP 的 ITC、ITM、LM 三个目标各做什么？](chapters/09-vision-language.md#vlm-005) | 社区线索 |
| VLM-006 | 视觉语言与图文多模态 | L2 | [BLIP 的 CapFilt 为什么同时需要 captioner 和 filter？](chapters/09-vision-language.md#vlm-006) | 社区线索 |
| VLM-007 | 视觉语言与图文多模态 | L1 | [BLIP-2 的 Q-Former 为什么能连接冻结的视觉编码器和 LLM？](chapters/09-vision-language.md#vlm-007) | 社区线索 |
| VLM-008 | 视觉语言与图文多模态 | L2 | [Q-Former 与两层 MLP 连接器怎样选，MLP 已普遍淘汰 Q-Former 吗？](chapters/09-vision-language.md#vlm-008) | 社区线索 |
| VLM-009 | 视觉语言与图文多模态 | L1 | [原始 LLaVA 的两阶段训练分别更新哪些模块？](chapters/09-vision-language.md#vlm-009) | 编辑补充 |
| VLM-010 | 视觉语言与图文多模态 | L2 | [VLM 指令微调的 labels 应怎样掩码？](chapters/09-vision-language.md#vlm-010) | 编辑补充 |
| VLM-011 | 视觉语言与图文多模态 | L1 | [LLaVA-1.5 相比原始 LLaVA 的关键改进是什么？](chapters/09-vision-language.md#vlm-011) | 编辑补充 |
| VLM-012 | 视觉语言与图文多模态 | L2 | [Qwen2-VL 的动态分辨率解决了什么问题？](chapters/09-vision-language.md#vlm-012) | 编辑补充 |
| VLM-013 | 视觉语言与图文多模态 | L2 | [M-RoPE 如何统一文本、图像和视频位置？](chapters/09-vision-language.md#vlm-013) | 编辑补充 |
| VLM-014 | 视觉语言与图文多模态 | L2 | [Qwen2.5-VL 的视觉编码器和时间建模有哪些变化？](chapters/09-vision-language.md#vlm-014) | 社区线索 |
| VLM-015 | 视觉语言与图文多模态 | L2 | [InternVL 的动态切图与全局缩略图各有什么作用？](chapters/09-vision-language.md#vlm-015) | 编辑补充 |
| VLM-016 | 视觉语言与图文多模态 | L2 | [OCR/文档问答差，怎样判断是视觉瓶颈还是语言瓶颈？](chapters/09-vision-language.md#vlm-016) | 社区线索 |
| VLM-017 | 视觉语言与图文多模态 | L3 | [单图 VLM 怎样扩展到多页文档问答？](chapters/09-vision-language.md#vlm-017) | 社区线索 |
| VLM-018 | 视觉语言与图文多模态 | L2 | [视觉幻觉怎样定义、评测和缓解？](chapters/09-vision-language.md#vlm-018) | 社区线索 |
| VLM-019 | 视觉语言与图文多模态 | L3 | [多模态模型看起来不看图，如何排查？](chapters/09-vision-language.md#vlm-019) | 社区线索 |
| VLM-020 | 视觉语言与图文多模态 | L2 | [VLM 能力如何评估，为什么不能只报一个榜单分数？](chapters/09-vision-language.md#vlm-020) | 编辑补充 |
| VLM-021 | 视觉语言与图文多模态 | L3 | [微调 VLM 时，视觉骨干、连接器和 LLM 该怎样冻结？](chapters/09-vision-language.md#vlm-021) | 社区线索 |
| VLM-022 | 视觉语言与图文多模态 | L2 | [交错图文和多图输入如何保持图像与指代关系？](chapters/09-vision-language.md#vlm-022) | 编辑补充 |
| VLM-023 | 视觉语言与图文多模态 | L2 | [Grounding 与普通图像问答有什么区别？](chapters/09-vision-language.md#vlm-023) | 编辑补充 |
| VLM-024 | 视觉语言与图文多模态 | L3 | [多模态预训练和 SFT 的数据配比怎样设计？](chapters/09-vision-language.md#vlm-024) | 编辑补充 |
| VLM-025 | 视觉语言与图文多模态 | L2 | [多模态检索模型与生成式 VLM 问答怎样分工？](chapters/09-vision-language.md#vlm-025) | 编辑补充 |
| VLM-026 | 视觉语言与图文多模态 | L2 | [以 LLaVA-1.5-7B 与 InternVL2.5-8B 为例，除骨干和训练之外有哪些差异？](chapters/09-vision-language.md#vlm-026) | 编辑补充 |
| OMM-001 | 视频、语音与 Omni | L2 | [视频 VLM 为什么不能简单无限堆叠视频帧？](chapters/10-video-audio-omni.md#omm-001) | 编辑补充 |
| OMM-002 | 视频、语音与 Omni | L2 | [视频时间位置编码为什么要考虑真实时间与 FPS？](chapters/10-video-audio-omni.md#omm-002) | 编辑补充 |
| OMM-003 | 视频、语音与 Omni | L3 | [长视频理解怎样压缩视觉 token，又有什么代价？](chapters/10-video-audio-omni.md#omm-003) | 编辑补充 |
| OMM-004 | 视频、语音与 Omni | L2 | [视频 moment retrieval 与视频问答如何评估？](chapters/10-video-audio-omni.md#omm-004) | 编辑补充 |
| OMM-005 | 视频、语音与 Omni | L1 | [Whisper 的输入、架构和多任务接口是什么？](chapters/10-video-audio-omni.md#omm-005) | 社区线索 |
| OMM-006 | 视频、语音与 Omni | L1 | [WER 怎样计算？为什么中文 ASR 常同时报 CER？](chapters/10-video-audio-omni.md#omm-006) | 社区线索 |
| OMM-007 | 视频、语音与 Omni | L2 | [CTC 与自回归语音解码有什么区别？](chapters/10-video-audio-omni.md#omm-007) | 编辑补充 |
| OMM-008 | 视频、语音与 Omni | L2 | [音频连续特征与离散 codec token 分别适合什么？](chapters/10-video-audio-omni.md#omm-008) | 社区线索 |
| OMM-009 | 视频、语音与 Omni | L2 | [VALL-E 式零样本 TTS 为什么能利用短语音提示？](chapters/10-video-audio-omni.md#omm-009) | 社区线索 |
| OMM-010 | 视频、语音与 Omni | L2 | [Qwen2-Audio 这类音频理解模型为什么不等于 ASR？](chapters/10-video-audio-omni.md#omm-010) | 编辑补充 |
| OMM-011 | 视频、语音与 Omni | L2 | [Qwen2.5-Omni 的 Thinker-Talker 如何协作？](chapters/10-video-audio-omni.md#omm-011) | 编辑补充 |
| OMM-012 | 视频、语音与 Omni | L3 | [ASR→LLM→TTS 流水线与端到端语音对话怎样选？](chapters/10-video-audio-omni.md#omm-012) | 社区线索 |
| OMM-013 | 视频、语音与 Omni | L3 | [流式语音系统的端到端延迟应怎样拆解？](chapters/10-video-audio-omni.md#omm-013) | 编辑补充 |
| OMM-014 | 视频、语音与 Omni | L2 | [为什么唇读等视觉信息可以帮助噪声下的 ASR？](chapters/10-video-audio-omni.md#omm-014) | 编辑补充 |
| OMM-015 | 视频、语音与 Omni | L3 | [Omni 模型怎样评估是否真的融合了声音与视觉？](chapters/10-video-audio-omni.md#omm-015) | 编辑补充 |
| DST-001 | 分布式训练与显存工程 | L1 | [数据并行 DDP 每个 step 做了什么？](chapters/11-distributed.md#dst-001) | 社区线索 |
| DST-002 | 分布式训练与显存工程 | L2 | [DDP 梯度累积如何保持与大 batch 等价？](chapters/11-distributed.md#dst-002) | 编辑补充 |
| DST-003 | 分布式训练与显存工程 | L1 | [大模型训练显存怎样估算和优化？以 7B Adam 为例](chapters/11-distributed.md#dst-003) | 社区线索 |
| DST-004 | 分布式训练与显存工程 | L1 | [ZeRO-1/2/3 各切分什么？理想状态显存是多少？](chapters/11-distributed.md#dst-004) | 社区线索 |
| DST-005 | 分布式训练与显存工程 | L2 | [FSDP 与 ZeRO-3 有什么联系和区别？](chapters/11-distributed.md#dst-005) | 社区线索 |
| DST-006 | 分布式训练与显存工程 | L2 | [Megatron 的 MLP 张量并行为什么先列切再行切？](chapters/11-distributed.md#dst-006) | 社区线索 |
| DST-007 | 分布式训练与显存工程 | L2 | [流水线并行的 bubble 从哪里来，怎样降低？](chapters/11-distributed.md#dst-007) | 社区线索 |
| DST-008 | 分布式训练与显存工程 | L2 | [Megatron 的 Sequence Parallelism 与 TP 怎样配合？](chapters/11-distributed.md#dst-008) | 编辑补充 |
| DST-009 | 分布式训练与显存工程 | L3 | [Context Parallelism 如何训练更长上下文？](chapters/11-distributed.md#dst-009) | 编辑补充 |
| DST-010 | 分布式训练与显存工程 | L3 | [MoE 的专家并行有哪些通信与负载问题？](chapters/11-distributed.md#dst-010) | 编辑补充 |
| DST-011 | 分布式训练与显存工程 | L1 | [AllReduce、ReduceScatter、AllGather 怎样对应？](chapters/11-distributed.md#dst-011) | 社区线索 |
| DST-012 | 分布式训练与显存工程 | L2 | [DDP 怎样重叠反向计算与梯度通信？](chapters/11-distributed.md#dst-012) | 社区线索 |
| DST-013 | 分布式训练与显存工程 | L2 | [激活 checkpointing 为什么省显存，有哪些正确性条件？](chapters/11-distributed.md#dst-013) | 社区线索 |
| DST-014 | 分布式训练与显存工程 | L1 | [BF16 与 FP16 混合精度训练为什么表现不同？](chapters/11-distributed.md#dst-014) | 编辑补充 |
| DST-015 | 分布式训练与显存工程 | L2 | [CPU/NVMe Offload 适合哪些场景，为什么可能变慢？](chapters/11-distributed.md#dst-015) | 编辑补充 |
| DST-016 | 分布式训练与显存工程 | L3 | [分布式训练怎样做到可靠断点续训？](chapters/11-distributed.md#dst-016) | 编辑补充 |
| DST-017 | 分布式训练与显存工程 | L2 | [DistributedSampler、set_epoch 和 drop_last 怎么用？](chapters/11-distributed.md#dst-017) | 编辑补充 |
| DST-018 | 分布式训练与显存工程 | L3 | [训练挂在 NCCL collective 上，如何定位？](chapters/11-distributed.md#dst-018) | 社区线索 |
| DST-019 | 分布式训练与显存工程 | L2 | [MFU、HFU 与 GPU utilization 有什么区别？](chapters/11-distributed.md#dst-019) | 社区线索 |
| DST-020 | 分布式训练与显存工程 | L3 | [多模态训练吞吐波动，怎样做 profiling 与负载平衡？](chapters/11-distributed.md#dst-020) | 编辑补充 |
| COD-001 | 手撕代码与算法 | L1 | [手写稳定 Softmax 与交叉熵，为什么要减最大值？](chapters/12-coding.md#cod-001) | 编辑补充 |
| COD-002 | 手撕代码与算法 | L1 | [手撕 Multi-Head Attention：形状、缩放与 mask 怎么写？](chapters/12-coding.md#cod-002) | 社区线索 |
| COD-003 | 手撕代码与算法 | L2 | [KV Cache 增量解码的因果 mask 为什么容易写错？](chapters/12-coding.md#cod-003) | 编辑补充 |
| COD-004 | 手撕代码与算法 | L2 | [手写 RoPE，并证明旋转保持范数与相对位置内积。](chapters/12-coding.md#cod-004) | 编辑补充 |
| COD-005 | 手撕代码与算法 | L2 | [手写 InfoNCE：正样本标签与 in-batch negatives 如何组织？](chapters/12-coding.md#cod-005) | 社区线索 |
| COD-006 | 手撕代码与算法 | L2 | [实现 top-k / top-p 采样，截断边界怎么处理？](chapters/12-coding.md#cod-006) | 编辑补充 |
| COD-007 | 手撕代码与算法 | L2 | [实现 LoRA Linear 并证明 merge 前后输出一致。](chapters/12-coding.md#cod-007) | 编辑补充 |
| COD-008 | 手撕代码与算法 | L2 | [实现 DPO loss，怎样避免符号和序列概率错误？](chapters/12-coding.md#cod-008) | 编辑补充 |
| COD-009 | 手撕代码与算法 | L2 | [实现 GRPO 组内优势，标准差为零时怎么办？](chapters/12-coding.md#cod-009) | 编辑补充 |
| COD-010 | 手撕代码与算法 | L1 | [数组第 k 大：堆与 Quickselect 怎样取舍？](chapters/12-coding.md#cod-010) | 社区线索 |
| COD-011 | 手撕代码与算法 | L1 | [岛屿问题：DFS/BFS 的时间、空间与边界。](chapters/12-coding.md#cod-011) | 社区线索 |
| COD-012 | 手撕代码与算法 | L1 | [手写编辑距离，并压缩到 O(min(m,n)) 空间。](chapters/12-coding.md#cod-012) | 社区线索 |
| COD-013 | 手撕代码与算法 | L1 | [实现 O(1) 的 LRU Cache，更新已有 key 怎么处理？](chapters/12-coding.md#cod-013) | 编辑补充 |
| COD-014 | 手撕代码与算法 | L2 | [手算并编码 MHA/GQA 的 KV Cache 显存。](chapters/12-coding.md#cod-014) | 社区线索 |
| COD-015 | 手撕代码与算法 | L2 | [手撕代码时怎样设计能揭露错误的测试？](chapters/12-coding.md#cod-015) | 编辑补充 |
| SYS-001 | 系统设计与线上故障 | L2 | [设计企业文档问答系统，先明确哪些约束？](chapters/13-system-design.md#sys-001) | 编辑补充 |
| SYS-002 | 系统设计与线上故障 | L2 | [LLM 服务 P95 延迟突然升高，如何定位？](chapters/13-system-design.md#sys-002) | 编辑补充 |
| SYS-003 | 系统设计与线上故障 | L2 | [如何测吞吐、并发、TTFT、TPOT，并避免错误比较？](chapters/13-system-design.md#sys-003) | 编辑补充 |
| SYS-004 | 系统设计与线上故障 | L2 | [Agent 上下文溢出，怎样压缩而保持任务连续性？](chapters/13-system-design.md#sys-004) | 社区线索 |
| SYS-005 | 系统设计与线上故障 | L2 | [工具调用失败后如何重试，怎样避免重复副作用？](chapters/13-system-design.md#sys-005) | 社区线索 |
| SYS-006 | 系统设计与线上故障 | L2 | [设计 Agent 可观测性：日志里应该记录什么？](chapters/13-system-design.md#sys-006) | 编辑补充 |
| SYS-007 | 系统设计与线上故障 | L3 | [多租户 RAG 如何保证权限隔离？](chapters/13-system-design.md#sys-007) | 编辑补充 |
| SYS-008 | 系统设计与线上故障 | L3 | [如何不停服更新知识库与 Embedding 模型？](chapters/13-system-design.md#sys-008) | 社区线索 |
| SYS-009 | 系统设计与线上故障 | L2 | [RAG 回答错了，怎样区分检索错误与生成错误？](chapters/13-system-design.md#sys-009) | 社区线索 |
| SYS-010 | 系统设计与线上故障 | L3 | [高并发 LLM 服务如何限流与降级？](chapters/13-system-design.md#sys-010) | 编辑补充 |
| SYS-011 | 系统设计与线上故障 | L2 | [Prefix Cache 能提高多少性能，怎样设计缓存键？](chapters/13-system-design.md#sys-011) | 编辑补充 |
| SYS-012 | 系统设计与线上故障 | L3 | [微调上线后质量衰减，如何诊断数据漂移？](chapters/13-system-design.md#sys-012) | 社区线索 |
| SYS-013 | 系统设计与线上故障 | L3 | [多模态服务图片/PDF 输入成本失控，怎么优化？](chapters/13-system-design.md#sys-013) | 编辑补充 |
| SYS-014 | 系统设计与线上故障 | L2 | [Prompt 优化“修好一类坏了另一类”，怎样控制回归？](chapters/13-system-design.md#sys-014) | 社区线索 |
| SYS-015 | 系统设计与线上故障 | L3 | [线上模型事故如何止损、回滚与复盘？](chapters/13-system-design.md#sys-015) | 编辑补充 |
| PRJ-001 | 项目、论文与行为追问 | L1 | [两分钟介绍 LLM/MLLM 项目，怎样讲清价值？](chapters/14-project.md#prj-001) | 社区线索 |
| PRJ-002 | 项目、论文与行为追问 | L1 | [面试官追问“哪部分是你做的”，怎样给出可核验回答？](chapters/14-project.md#prj-002) | 社区线索 |
| PRJ-003 | 项目、论文与行为追问 | L2 | [为什么选这个模型，而不是更大的模型？](chapters/14-project.md#prj-003) | 编辑补充 |
| PRJ-004 | 项目、论文与行为追问 | L2 | [如何证明项目提升来自你的改动，而不是数据/算力增加？](chapters/14-project.md#prj-004) | 社区线索 |
| PRJ-005 | 项目、论文与行为追问 | L2 | [项目最困难的 bug，怎样避免答成“调参故事”？](chapters/14-project.md#prj-005) | 编辑补充 |
| PRJ-006 | 项目、论文与行为追问 | L2 | [介绍一篇论文，面试官最可能在哪些地方深挖？](chapters/14-project.md#prj-006) | 编辑补充 |
| PRJ-007 | 项目、论文与行为追问 | L2 | [你会怎样把 badcase 转成下一轮训练数据？](chapters/14-project.md#prj-007) | 编辑补充 |
| PRJ-008 | 项目、论文与行为追问 | L2 | [训练/评测实验怎样做到别人能复现？](chapters/14-project.md#prj-008) | 编辑补充 |
| PRJ-009 | 项目、论文与行为追问 | L1 | [遇到不会的模型或面试追问，怎样回答？](chapters/14-project.md#prj-009) | 编辑补充 |
| PRJ-010 | 项目、论文与行为追问 | L1 | [技术方案有分歧时，如何推进决策？](chapters/14-project.md#prj-010) | 编辑补充 |
