# 基础、预训练、微调与推理：检索与核验记录

核验日期：2026-10-02（Asia/Shanghai）。范围为 TFM / PRE / FT / INF，各 15 题，共 60 题。

本分片收录 11 个社区来源、50 个原始论文或官方文档。31 题有公开问题、主题或搜索摘要证据，29 题是覆盖知识体系所作的编辑补充。这里的“有证据”不等于公司已确认真题；社区作者及汇编者的公司归属未独立认证。答案全部独立综合，社区答案不作为权威依据。

## 实际执行的社区检索查询

- `大模型 面经 Transformer LoRA 量化 牛客`
- `大模型 面试 八股 预训练 SFT 推理 GitHub`
- `site.zhihu.com 大模型 面经 预训练 token loss Transformer`
- `site.nowcoder.com/discuss 大模型 面经 RoPE RMSNorm FlashAttention 量化`
- `site.nowcoder.com 大模型 面经 数据清洗 tokenizer 预训练`
- `site.xiaohongshu.com 大模型 面试 LoRA Transformer`
- `site.nowcoder.com 大模型 面经 预训练 数据 tokenizer scaling laws`

小红书定向查询没有得到本分片可以打开核验的原帖。搜索中出现的“小红书公司面经”聚合站，标题中的小红书是公司名称，不能视为小红书平台采集结果，本分片未采用。平台限制和其他渠道的完整记录由总仓库汇总。

## 社区访问状态及题目对应

| 来源 | 访问 | 对应题目 | 实际读取与边界 |
| --- | --- | --- | --- |
| [CORE-S001 · 网易大模型应用岗面经（一面、二面）](https://www.nowcoder.com/discuss/909223288612610048?sourceSSR=dynamic) | full | TFM-005、FT-004、INF-005 | 已读公开正文，仅摘 LayerNorm、微调方法对比、FlashAttention 三类题目。公司与面试经历为发帖人自述；正文答案有错误，未用作答案依据。 |
| [CORE-S002 · 腾讯Teg大模型暑期算法面经](https://api-cdn.nowcoder.com/feed/main/detail/f91200d9c116401090432a2d78e5f76d?sourceSSR=users) | full | TFM-004、FT-005、INF-009 | 公开正文可读；仅摘 Transformer 架构、LoRA 原理、量化原理三个主题。公司及轮次未独立认证。 |
| [CORE-S003 · AgentGuide：公司面试案例整理](https://github.com/adongwanai/AgentGuide/blob/main/docs/04-interview/12-company-interview-cases.md) | full | FT-002、FT-006、FT-007、INF-002 | 已读题目列表；二次汇编自称多家公司的案例，缺少逐条原始面经链接，不能视为已核实公司真题。摘 SFT masking、LoRA 初始化/调参、推理显存三个主题。 |
| [CORE-S004 · LLMs_Interview：目录与更新记录](https://github.com/threeneedone/LLMs_Interview) | full | PRE-002、PRE-015 | 已读 README，含分词、预训练、评测等章节。这里只作为公开题库主题证据，未逐条核验其各章节问题，不能归属某家公司。 |
| [CORE-S005 · 大模型算法面经+问题+答案](https://www.nowcoder.com/discuss/891322059656052736?toCommentId=22719031) | snippet | TFM-008、TFM-015、INF-003 | 搜索摘要可读，正文抓取报 Cache miss。摘要明确出现 RoPE、MoE、MHA/MQA/GQA；未采用其答案或频率断言。 |
| [CORE-S006 · 商汤NLP一面](https://www.nowcoder.com/feed/main/detail/fdc049a6b3444f3abee6f22256fc64ac) | snippet | TFM-006、TFM-010、FT-008 | 搜索摘要含 RMSNorm、SwiGLU、QLoRA；打开页面仅返回导航，未读取完整正文。公司名称为帖子自述，证据降为搜索摘要。 |
| [CORE-S007 · 说说现在到底都在问什么（长文，慎入）](https://api-cdn.nowcoder.com/discuss/925432340174606336?sourceSSR=subject) | full | TFM-007、TFM-009、INF-015 | 已读公开正文，摘 Pre/Post-Norm、RoPE 外推、TTFT/吞吐三个主题。原文频率与题型比例属个人有限样本，未用于本仓库统计。 |
| [CORE-S008 · 知乎问答：生成语言模型微调与预训练如何计算 loss](https://www.zhihu.com/en/answer/3335594083) | snippet | PRE-001、FT-002 | 搜索摘要提到 label shift 与 SFT 屏蔽 query；正文打开跳转安全验证并要求登录，未绕过访问限制。仅保留主题证据，不采纳全文或公司归属。 |
| [CORE-S009 · 大厂问什么：2025-26 算法工程师面试常见问题整理（阿里系）](https://www.nowcoder.com/discuss/848942791164981248?sourceSSR=dynamic) | snippet | TFM-011、INF-010、INF-013 | 搜索可读摘录提到 Attention 复杂度、GPTQ、投机解码。其题目次数、P7 概率等未见可复核样本，全部不采纳；公司归属未认证。 |
| [CORE-S010 · LLM-Interview-Code](https://github.com/ckd0817/LLM-Interview-Code) | snippet | TFM-001、TFM-013、FT-001 | 只读取搜索摘要；目录明确列出注意力与 Pretrain/SFT 等损失的手撕实现。作为备考主题证据，不是公司面经。 |
| [CORE-S011 · 大模型常考面试题 100 道（第 51～75 道）](https://www.nowcoder.com/discuss/866256452975943680) | snippet | PRE-003、PRE-004、PRE-007 | 搜索摘要可见 tokenizer、Scaling Law、预训练去重主题，打开后只有导航。作为题库主题证据，不采用其高频或大厂真实性断言。 |

只抽取各页少量问题主题或目录概念，未搬运整页题目和正文答案。`reported_question` 表示公开文中列出问题；`reported_topic` 表示题库、目录或备考仓库覆盖主题；`search_snippet` 表示仅搜索摘要可见。按这些主题写出的规范题目可能比原提问更完整；具体题目不是原文逐字转录。

## 原始资料与核验深度

下面的 full 表示相关正文或官方文档实际可访问，notes 说明实际阅读范围，不能推断每篇论文已逐字通读。arXiv 摘要页面统一标 snippet；算法公式优先使用 PDF 或官方实现核验。

| 来源 | 访问 | 对应题目 | 实际核验 |
| --- | --- | --- | --- |
| [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762) | full | TFM-001、TFM-002、TFM-003、TFM-004、TFM-010、TFM-011、TFM-012、PRE-001、PRE-010、PRE-014、INF-005 | 已读 PDF 第 3 节模型、注意力公式/缩放推导、第 4 节复杂度及训练设置；仅简述方法，不复制全文。 |
| [CORE-S021 · Layer Normalization](https://arxiv.org/pdf/1607.06450) | full | TFM-005 | 已读 PDF 第 2、3 节的统计轴和公式，核验与 BatchNorm 的区别。 |
| [CORE-S022 · Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467) | full | TFM-006 | 已读 PDF 第 3 节定义及不变性说明，核验 RMS 公式。 |
| [CORE-S023 · On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745) | full | TFM-007、TFM-012、PRE-014 | 已读 PDF 的 Pre/Post-LN 初始化梯度分析与 warmup 实验；结论限定在相应设置，不推广为永不需要 warmup。 |
| [CORE-S024 · GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202) | full | TFM-010 | 已读 PDF 定义页，核验 SwiGLU 的三矩阵结构和同参数预算缩小中间维度。 |
| [CORE-S025 · RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864) | full | TFM-008 | 已读 PDF 第 3 节旋转定义及相对位置点积，核验 Q/K 旋转关系。 |
| [CORE-S026 · Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/pdf/2306.15595) | full | TFM-009 | 已读 PDF 方法与图 1，核验位置索引下缩放；扩展效果不作为任意底座的保证。 |
| [CORE-S027 · Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961) | full | TFM-015 | 已读 PDF 路由容量及辅助负载均衡说明；题库工程权衡为独立综合，未复制实验数值。 |
| [CORE-S028 · BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | snippet | TFM-004 | 本轮只核验 arXiv 摘要中的双向上下文与下游适配定位，未通读正文。 |
| [CORE-S029 · Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) | snippet | TFM-004 | 本轮只核验 arXiv 摘要与研究定位，未通读 T5 全文；架构公式另参考 Transformer 原论文。 |
| [CORE-S030 · Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) | snippet | TFM-014、PRE-003 | 本轮只读摘要，核验输入/输出 embedding 绑定建议；参数计数和工程边界为独立推导。 |
| [CORE-S031 · Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | snippet | TFM-012 | 本轮只读摘要核验残差学习背景；题库残差公式同时由 Transformer 原论文支撑，梯度为数学推导。 |
| [CORE-S032 · torch.nn.CrossEntropyLoss — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html) | full | TFM-013、PRE-001、FT-002 | 已读官方文档 loss 定义、输入 logits、ignore_index 与 reduction；固定版本 URL 便于复核。 |
| [CORE-S033 · Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | snippet | PRE-004 | 本轮只读摘要，核验 loss 对模型、数据与计算量的经验幂律定位，未通读全文。 |
| [CORE-S034 · Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556) | full | PRE-004、PRE-005 | 已读 PDF 第 3 节 isoFLOP 方法、参数/数据联合扩展及 loss 拟合，引用 compute-optimal 的条件而非万能 20 倍规则。 |
| [CORE-S035 · SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226) | snippet | PRE-002、PRE-003 | 本轮只读摘要，核验从原始文本训练语言无关子词工具的定位；BPE 过程另由原论文支撑。 |
| [CORE-S036 · The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) | snippet | PRE-006、PRE-007 | 本轮只读摘要核验网页过滤与去重的重要性，具体清洗流程是面试工程建议，非原论文逐条摘录。 |
| [CORE-S037 · Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | snippet | TFM-004、PRE-001、PRE-008 | 本轮只核验摘要及研究定位，未通读全文；CLM 的数学目标用 Transformer、官方 PPL 文档交叉支撑。 |
| [CORE-S038 · Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) | snippet | PRE-007、PRE-008 | 本轮只读摘要，核验去重可降低重复记忆及 train/test 重叠；未照搬具体实验倍数。 |
| [CORE-S039 · DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429) | snippet | PRE-009 | 本轮只读摘要，核验用代理模型学习领域数据权重的流程；其他配比建议为编辑综合。 |
| [CORE-S040 · Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964) | snippet | PRE-011、FT-012 | 本轮只读摘要，核验 DAPT/TAPT 可带来特定域任务收益，未把分类实验当作所有 LLM 的效果保证。 |
| [CORE-S041 · Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | snippet | PRE-012 | 本轮只读摘要，核验 Adam 的 L2 与解耦衰减不等价，未通读正文。 |
| [CORE-S042 · Automatic Mixed Precision package — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/amp.html) | full | PRE-013、PRE-014 | 已读官方文档 autocast、GradScaler 与 dtype 说明及梯度裁剪相关约束。 |
| [CORE-S043 · Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity) | full | TFM-013、PRE-001、PRE-015 | 已读官方文档 PPL 定义、tokenization 注意事项、滑动窗口和计损示例；固定窗口评测结论有条件。 |
| [CORE-S044 · TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) | full | TFM-003、PRE-010、PRE-011、FT-001、FT-002、FT-003、FT-010、FT-015 | 已读官方文档 label shifting/masking、assistant_only_loss、packing 与训练配置；具体 API 默认值可能随版本变化，题库讲原理。 |
| [CORE-S045 · Chat templates](https://huggingface.co/docs/transformers/chat_templating) | full | FT-003、FT-015 | 已读官方模板、add_special_tokens、generation prompt 与模型训练部分。 |
| [CORE-S046 · LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685) | full | FT-004、FT-005、FT-006、FT-007、FT-012 | 已读 PDF 第 4 节公式、A/B 初始化、alpha/r 及权重合并；梯度与参数计数为公式推导。 |
| [CORE-S047 · QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314) | full | FT-007、FT-008、FT-014、INF-009 | 已读 PDF 背景及第 3 节，核验冻结量化底座、NF4、双重量化与分页优化器；未把存储位宽当算术精度。 |
| [CORE-S048 · Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) | snippet | FT-004、FT-009 | 本轮只读摘要，核验 Adapter 冻结底座并增加少量任务模块的定位，未通读正文。 |
| [CORE-S049 · Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) | snippet | FT-004、FT-009 | 本轮只读摘要，核验连续前缀与 virtual tokens 定位；精细层级和成本为编辑讲解，非全文摘录。 |
| [CORE-S050 · LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) | snippet | FT-001、FT-010、FT-015 | 本轮只读摘要，核验少量精选指令集实验；没有把该实验概括成 SFT 数据越少越好的普遍规律。 |
| [CORE-S051 · Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) | snippet | FT-011 | 本轮只读摘要，核验生成指令/回答再过滤无效与相似样本的流程；执行器校验等为编辑工程建议。 |
| [CORE-S052 · Continual Learning Through Synaptic Intelligence](https://arxiv.org/abs/1703.04200) | snippet | FT-012 | 本轮只读摘要，作为持续学习和参数重要性约束的原始研究背景；分类实验不等同于已证明解决 LLM 遗忘。 |
| [CORE-S053 · Performing gradient accumulation with Accelerate](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation) | full | FT-013 | 已读官方文档，特别核验变长 token loss 分母、DDP 同步和最后窗口处理。 |
| [CORE-S054 · torch.utils.checkpoint — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/checkpoint.html) | full | FT-014 | 已读官方 activation checkpointing 原理和状态一致性、随机数与 reentrant 限制。 |
| [CORE-S055 · Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/pdf/1911.02150) | full | INF-001、INF-002、INF-003、INF-004 | 已读 PDF 注意力形状与增量解码代码，核验 MQA、K/V 共享及带宽动机。 |
| [CORE-S056 · GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245) | full | INF-001、INF-002、INF-003、INF-004 | 已读 PDF 方法和 MHA/MQA/GQA 头数关系，核验 mean-pooling 后 uptraining 的步骤。 |
| [CORE-S057 · FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135) | full | TFM-011、INF-004、INF-005、INF-006 | 初版只读摘要；交叉审阅后补读第 3.1 节和 Algorithm 1，核验分块最大值、归一化量的重缩放与反向重算。full 指相关正文可读，未通读或复现实验。 |
| [CORE-S058 · Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | snippet | INF-002、INF-006、INF-007、INF-014 | 本轮只读摘要，核验 KV 分页、碎片及缓存共享的设计目标；具体工程建议独立综合。 |
| [CORE-S059 · Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) | snippet | INF-004、INF-007、INF-015 | 只读会议官方页摘要；按摘要级统一标记，核验 iteration-level scheduling 与 selective batching，未读论文 PDF，未复制吞吐倍数。 |
| [CORE-S060 · The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | snippet | INF-008 | 本轮只读摘要，核验 nucleus sampling 的动态概率质量集合动机；temperature 等为采样基础讲解。 |
| [CORE-S061 · GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/pdf/2210.17323) | full | INF-009、INF-010 | 已读 PDF 第 3 节逐层重构目标与近似二阶方法，核验不是全参数重新训练。 |
| [CORE-S062 · AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) | snippet | INF-009、INF-011 | 已读摘要明确写出用激活统计、等价缩放且避免混合精度。PDF 读取曾报错；不声称读取全文。 |
| [CORE-S063 · SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/pdf/2211.10438) | full | INF-009、INF-012 | 已读 PDF 第 4 节等价通道缩放公式和 W8A8 动机，核验浮点等价不等于量化零误差。 |
| [CORE-S064 · Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192) | full | INF-013 | 已读 PDF 第 2 节算法与第 3 节接受率，核验 min(1,p/q) 与 norm(max(p−q,0)) 修正分布。 |
| [CORE-S065 · Automatic Prefix Caching — vLLM v0.20.1](https://docs.vllm.ai/en/v0.20.1/design/prefix_caching/) | full | INF-014 | 已读官方文档 block hash、父前缀、模态 hash 与 cache_salt；旧 latest 路径重定向到 contributing，已替换为固定版本正确页面。 |
| [CORE-S066 · Metrics — vLLM v0.20.1](https://docs.vllm.ai/en/v0.20.1/design/metrics/) | full | INF-015 | 已读官方指标定义，包括 TTFT、prefill/decode interval 和请求前端计时；题库同时指出客户端计时与服务端口径区别。 |
| [CORE-S067 · Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162.pdf) | full | PRE-002 | 已读 PDF 第 3.2 节 BPE 合并过程与子词动机。 |
| [CORE-S068 · Transformers: Utilities for generation](https://huggingface.co/docs/transformers/internal/generation_utils) | full | INF-008 | 交叉审阅后已读 Temperature/TopK/TopPLogitsWarper 的定义与参数约束；不采用示例中的常用参数范围，未运行模型示例。 |
| [CORE-S069 · PEFT: Prompt tuning](https://huggingface.co/docs/peft/main/en/package_reference/prompt_tuning) | full | FT-009 | 交叉审阅后已读方法说明与 Usage，核验输入层虚拟 token、冻结底座与 Prefix-Tuning 区别；main 文档不保证 API 永久不变，未运行示例。 |

## 原始资料检索及路径修正

- 使用论文标题、已知 arXiv 标识、PyTorch / Hugging Face / vLLM 官方文档地址执行 `open`，并用 `find` 定位 Pre-LN、SwiGLU、relative、load、initialize、NormalFloat、ignore_index、assistant_only_loss、bfloat16、Activation checkpointing、diag、min(1、cache_salt 与 TTFT 等关键片段。
- 查询 `Orca distributed serving system transformer based generative models iteration level scheduling paper`，找到 USENIX 官方会议页和开放 PDF；本分片实际采用会议页完整摘要。
- 查询 `site.docs.vllm.ai automatic prefix caching design hash cache_salt`，找到 v0.20.1 固定版本文档。原 `latest/design/automatic_prefix_caching/` 重定向到 contributing 页面，没有把错误页面当作前缀缓存资料。
- 查询 `site.arxiv.org Don't Stop Pretraining Adapt Language Models Domains Tasks 2004`，核验了 2004.10964 的标题及领域适配定位。
- PyTorch `stable` 页面在访问时先返回版本跳转，继续读取到 2.14 官方页面，最终记录固定版本 URL。题库以原理为主，不将当前 API 默认值写成长期不变事实。
- 两个凭记忆尝试的 arXiv 标识不匹配：1608.07909 实际为 Cosmic Decoherence: Massive Fields，2207.02037 实际为 Trion-phonon interaction in atomically thin semiconductors。已核验标题后弃用，未纳入来源与答案。Orca 改用 USENIX 原始来源。
- AWQ PDF 本轮读取返回错误；摘要明确解释激活感知、等价缩放与避免混合精度，故保留摘要级来源。未虚称核验 PDF 全文。
- `find` 在部分 PDF 上因文字拆分未匹配到字符串，随后使用已知段落 `open` 阅读公式，例如 GPTQ 第 3 节、RoPE 第 3 节、SwiGLU 定义页、RMSNorm 定义及投机解码第 2 节。
- 交叉审阅后补读 FlashAttention PDF 第 3.1 节与 Algorithm 1，实际看到逐行最大值与归一化量合并，以及 backward recomputation 段；相应来源由摘要级更新为明确正文范围。另读官方 generation utilities 与 PEFT Prompt tuning，追加直接出处。Orca 未额外阅读正文，保持摘要级 snippet。

## 纠错与答案编写原则

- 社区原答案把 KV cache 复用与“降低总显存”混淆：本题库明确缓存是显存换重复计算，分页才减少预留浪费和冗余。
- 社区有把 AWQ 写成保留 1% FP16 权重的说法：原摘要明确最终方案通过等价缩放避免硬件不友好的混合精度，本题库纠正该说法。
- 社区有 BN 在单样本推理必然失效的说法：本题库明确 eval 模式可使用 running statistics，Transformer 选 LN 的理由是统计轴和序列/批次特性。
- FlashAttention 只降低中间矩阵存储和 IO，精确稠密 attention 的算术量仍有二次项；没有称其为近似 attention 或保证 bitwise 一致。
- 量化底座与 QLoRA 的存储位宽、计算 dtype、激活和优化器占用分开讨论；LoRA 更新低秩不意味着原权重低秩。
- 投机解码分布正确性用接受率与拒绝后的修正分布解释，避免把同分布误写成固定种子下逐 token 相同。
- 参数计数、KV 容量例子、Jacobian 与归一化关系包含独立数学推导；工程建议是面试回答的编辑综合，不伪装成论文逐条结论或任何公司的标准答案。

## 本地初检

- 60 个问题 ID 唯一，每类连续 001–015。
- 每题至少一个 primary reference，所有引用和社区证据均指向本分片存在的 source ID。
- quick 长度均在 60–140 字符；detail 3–5 条；pitfalls / followups 各 1–2 条。
- 没有社区证据的题目均 `editorial=true`；有主题证据仍不宣称其是真实公司考题。
- JSON 仅保存独立回答、少量题目主题和链接，没有复制文章、整套既有题库或付费内容。

本分片只做公开渠道收集和本地交付，没有外部发布，也没有绕过登录、验证码或付费访问。
