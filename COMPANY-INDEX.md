# 公司线索索引

[返回首页](README.md)

只收录来源中明确提到的公司；这里呈现作者或转载者的说法，不证明雇主曾问过这些题，也不推断公司的固定偏好。

## 商汤

- [TFM-006 · RMSNorm 与 LayerNorm 的公式和性质有什么区别？](chapters/01-transformer.md#tfm-006)
- [TFM-010 · FFN 提供什么作用？SwiGLU 为什么常调整中间维度？](chapters/01-transformer.md#tfm-010)
- [FT-008 · LoRA 与 QLoRA 有何区别，NF4、双重量化与分页优化器做什么？](chapters/03-finetuning.md#ft-008)

## 字节跳动

- [ALN-001 · SFT、RLHF 与 DPO 分别解决什么问题？](chapters/05-alignment.md#aln-001)
- [ALN-003 · PPO 的概率比、clip 和 min 分别起什么作用？](chapters/05-alignment.md#aln-003)
- [ALN-004 · GAE 如何计算，λ 与 γ 如何影响优势估计？](chapters/05-alignment.md#aln-004)
- [ALN-006 · DPO 的损失如何从 KL 正则化 RLHF 目标推出？](chapters/05-alignment.md#aln-006)
- [ALN-008 · PPO 与 DPO 在工程上如何选型？](chapters/05-alignment.md#aln-008)
- [ALN-014 · ORM 与 PRM 的区别和信用分配难点是什么？](chapters/05-alignment.md#aln-014)
- [ALN-018 · IPO 等 DPO 变种主要试图解决什么问题？](chapters/05-alignment.md#aln-018)
- [ALN-019 · 离线偏好优化和在线 RL 的分布差异是什么？](chapters/05-alignment.md#aln-019)
- [ALN-020 · 多目标奖励发生冲突时如何处理？](chapters/05-alignment.md#aln-020)
- [RAG-001 · 一个可落地的 RAG 系统有哪些环节？](chapters/06-rag.md#rag-001)
- [RAG-003 · chunk 大小与 overlap 怎样设置？](chapters/06-rag.md#rag-003)
- [RAG-006 · 混合检索的分数融合与 RRF 有什么区别？](chapters/06-rag.md#rag-006)
- [RAG-007 · reranker 和 embedding 检索模型如何分工？](chapters/06-rag.md#rag-007)
- [RAG-012 · RAG 为什么仍会幻觉，怎样设计引用与拒答？](chapters/06-rag.md#rag-012)
- [AGT-005 · 多 Agent 的通信与共享状态如何设计？](chapters/07-agents.md#agt-005)
- [VLM-014 · Qwen2.5-VL 的视觉编码器和时间建模有哪些变化？](chapters/09-vision-language.md#vlm-014)
- [COD-002 · 手撕 Multi-Head Attention：形状、缩放与 mask 怎么写？](chapters/12-coding.md#cod-002)
- [COD-005 · 手写 InfoNCE：正样本标签与 in-batch negatives 如何组织？](chapters/12-coding.md#cod-005)
- [COD-010 · 数组第 k 大：堆与 Quickselect 怎样取舍？](chapters/12-coding.md#cod-010)
- [COD-011 · 岛屿问题：DFS/BFS 的时间、空间与边界。](chapters/12-coding.md#cod-011)
- [COD-014 · 手算并编码 MHA/GQA 的 KV Cache 显存。](chapters/12-coding.md#cod-014)
- [SYS-004 · Agent 上下文溢出，怎样压缩而保持任务连续性？](chapters/13-system-design.md#sys-004)
- [SYS-014 · Prompt 优化“修好一类坏了另一类”，怎样控制回归？](chapters/13-system-design.md#sys-014)
- [PRJ-001 · 两分钟介绍 LLM/MLLM 项目，怎样讲清价值？](chapters/14-project.md#prj-001)
- [PRJ-002 · 面试官追问“哪部分是你做的”，怎样给出可核验回答？](chapters/14-project.md#prj-002)
- [PRJ-004 · 如何证明项目提升来自你的改动，而不是数据/算力增加？](chapters/14-project.md#prj-004)

## 月之暗面

- [AGT-002 · 如何让 function calling 更可靠？](chapters/07-agents.md#agt-002)
- [AGT-010 · 上下文工程怎样降低长任务成本而保持信息？](chapters/07-agents.md#agt-010)

## 百度

- [COD-012 · 手写编辑距离，并压缩到 O(min(m,n)) 空间。](chapters/12-coding.md#cod-012)
- [SYS-009 · RAG 回答错了，怎样区分检索错误与生成错误？](chapters/13-system-design.md#sys-009)

## 网易

- [TFM-005 · Transformer 为什么常用 LayerNorm，而不是 BatchNorm？](chapters/01-transformer.md#tfm-005)
- [FT-004 · 全参数微调、LoRA、Adapter 和 Prefix-Tuning 怎样选择？](chapters/03-finetuning.md#ft-004)
- [INF-005 · FlashAttention 的核心思想是什么，会改变注意力结果吗？](chapters/04-inference.md#inf-005)

## 腾讯

- [TFM-004 · Encoder-only、Decoder-only 与 Encoder-Decoder 怎样选择？](chapters/01-transformer.md#tfm-004)
- [FT-005 · LoRA 的低秩更新公式及可训练参数量是什么？](chapters/03-finetuning.md#ft-005)
- [INF-009 · PTQ、QAT、W4A16 与 per-group quantization 是什么？](chapters/04-inference.md#inf-009)
- [SYS-005 · 工具调用失败后如何重试，怎样避免重复副作用？](chapters/13-system-design.md#sys-005)
- [SYS-008 · 如何不停服更新知识库与 Embedding 模型？](chapters/13-system-design.md#sys-008)

## 阿里巴巴（淘天）

- [VLM-001 · CLIP 的结构和双向对比损失是什么？](chapters/09-vision-language.md#vlm-001)
- [VLM-005 · BLIP 的 ITC、ITM、LM 三个目标各做什么？](chapters/09-vision-language.md#vlm-005)
- [VLM-006 · BLIP 的 CapFilt 为什么同时需要 captioner 和 filter？](chapters/09-vision-language.md#vlm-006)
- [VLM-007 · BLIP-2 的 Q-Former 为什么能连接冻结的视觉编码器和 LLM？](chapters/09-vision-language.md#vlm-007)
- [VLM-008 · Q-Former 与两层 MLP 连接器怎样选，MLP 已普遍淘汰 Q-Former 吗？](chapters/09-vision-language.md#vlm-008)

## 阿里系

- [TFM-011 · Transformer 一层的时间、空间复杂度如何估算？](chapters/01-transformer.md#tfm-011)
- [INF-010 · GPTQ 为什么利用二阶信息进行逐层量化？](chapters/04-inference.md#inf-010)
- [INF-013 · 投机解码怎样保证目标模型的采样分布？](chapters/04-inference.md#inf-013)
