# 多模态与分布式训练题库采集记录

采集日期：2026-10-02（Asia/Shanghai）。范围：VLM 25 题、OMM 15 题、DST 20 题。此分片保留 60 条来源记录；其中 8 条社区来源、52 条论文或官方资料。MM-S008 是未采用的候选来源，保留其访问限制供复查。

## 证据口径

- **题目采集**只从公开社区内容提取少量主题并重新表述；作者自述的经历与公司归因没有经过独立认证。来源标题中的公司名不构成仓库的公司真题认证。
- **答案核验**使用作者论文、项目官方仓库与官方文档；社区附带答案不作为技术正确性的依据。答案的诊断步骤、工程取舍、例子和追问由编辑整理，涉及推导的公式注明适用条件。
- `reported_question`：来源明确列出相应问法或可明确对应的题目；`reported_topic`：来源讨论备考主题，本题做结构化展开；`search_snippet`：只有搜索索引返回的片段，不声称读过全文。
- `editorial=true` 表示本题没有已采用社区证据，是为补齐知识覆盖新增的题目。当前 27 题附社区线索，33 题为编辑补充；这不表示任何面试发生频率。
- `access=full` 表示正文可读且已检查相关内容，不代表每一章节都被逐句审校；`snippet` 包含只读作者摘要和只读搜索片段两种情况，具体限制写在 notes。
- 本分片未新增小红书原帖证据，不使用二手标题冒充已读小红书正文；全仓库的渠道边界由主编合并记录。
- 来源的 `published_date=null` 表示未在本轮可靠提取日期。访问日期与发布日期分开；不继承“高频”“必考”“最新”或计次排行等未经核实的营销标签。

## 搜索查询与访问结果

| 查询 | 结果与采用方式 |
| --- | --- |
| `site.nowcoder.com 多模态 大模型 面经 CLIP LLaVA 分布式` | 发现知乎/社区与 GitHub 题库线索，再分渠道核对。 |
| `site.zhihu.com 多模态 大模型 面试 CLIP LLaVA` | MM-S001 可读英文索引/翻译正文；提取 CLIP、BLIP、BLIP2、连接器主题。MM-S007 仅读搜索片段。 |
| `site.github.com LLM interview questions multimodal Deepspeed FSDP` | MM-S004、MM-S006 可读，取少量主题；题库作者的公司和频率声明未采用。 |
| `site.nowcoder.com/discuss 多模态 面经 CLIP BLIP 视频` | 本轮没有取得可用于本分片视频题的明确面经证据；视频题按编辑补充标记。 |
| `site.zhihu.com 大模型 面试 视频 音频 Whisper` | 未取得可靠的音频/视频公司面经；不强行绑定公司。 |
| `site.nowcoder.com/discuss 大模型 面经 deepspeed zero 分布式` | MM-S002 正文可读；MM-S003 搜索片段可读，直接打开只返回页面壳，因此保留 snippet。 |
| `site.github.com interview questions video audio multimodal Whisper` | MM-S005 可读且有 ASR、codec、TTS、语音对话备考主题。MM-S008 直接打开 cache miss，未用作题目证据。 |
| `site.docs.nvidia.com megatron core context parallelism sequence parallelism` | 找到 MM-S057/058；个别旧 api-guide 路径打不开，改用官方用户指南与仓库文档。 |
| `site.docs.nvidia.com megatron core expert parallelism` | 找到 MM-S058 的 EP 与 TP 组合说明。 |
| `site.docs.nvidia.com nccl ring allreduce reduce scatter` | 找到官方 collective 说明，采用 MM-S059；未将 ring 当作 NCCL 永远使用的算法。 |
| `site.arxiv.org OmniBench audio video text benchmark` | 找到 MM-S043 摘要，用于确认评估动机。 |

原始文献多数用确定的论文地址直接打开。CLIP、ZeRO、Megatron、PaLM 最初读作者摘要，之后追加 HTML 关键段落核验，来源记录已更新为 full。BLIP、BLIP2、LLaVA、LLaVA-1.5、Qwen2/2.5-VL、Whisper、Qwen2.5-Omni、Moshi 检查可读 HTML；其他论文摘要来源保持 snippet。SigLIP HTML 访问失败，保留作者摘要的机制说明，不搬运实验表或未读公式。

PyTorch stable 地址在本轮跳转到 2.14；采用工具实际可读的 2.14 文档地址，不据此承诺用户已有环境兼容。NCCL 与 DeepSpeed 文档也可能随版本更新，使用时应匹配本地安装版本。

## 公式与版本易错点核查

- VLM-001：CLIP 双向交叉熵与训练尺度依据作者 HTML 的训练说明，特征归一化另核对官方示例。
- VLM-009/010：原始 LLaVA 第二阶段仍冻结视觉编码器；loss 只覆盖回答与结束标记，attention mask 与 loss mask 分开。
- VLM-014：Qwen2.5-VL 大多数视觉层用窗口注意力，四层保留全局注意力；不能宣称整个模型严格线性。
- OMM-004：时间 IoU 明确用交集/并集，非定位精度或分类准确率。
- OMM-006：WER 分母是参考词数；corpus 聚合先汇总再相除。字符级指标需统一文字归一化。
- OMM-007：CTC 对齐包括 blank 和重复折叠；目标长度小于输入长度不是所有含重复目标的充分条件。
- DST-003/004：显存公式沿用明确的 16 bytes/parameter 状态假设，分片数是数据并行组大小；不包括激活和临时峰值。
- DST-006：列/行切分采用数学乘法记号，框架权重转置时需映射；第二层输出是求和。
- DST-007：bubble 公式限理想 fill-drain、等时阶段，不能推广为所有 1F1B 调度测量值。
- DST-008/009：Megatron SP 主要切部分操作，CP 切各层序列激活；局部独立 attention 不等价 CP。
- DST-011：ring 公式区分每卡发送和接收，不把二者再与算法流量口径混加。
- DST-019：MFU 不把重计算当有效模型 FLOPs；6NT 对长上下文、多模态和 MoE 有明确限制。

## 来源索引

下表 ID 与 JSON 中引用一一对应。正文/摘要的用途与限制见 notes，题目对应列表由实际 JSON 自动生成。

| ID | 来源 | 渠道 / 类型 | 访问 | 用于题目 |
| --- | --- | --- | --- | --- |
| MM-S001 | [淘天面经：大模型（知乎英语索引页）](https://www.zhihu.com/en/article/26593966342) | 知乎 / community | full | VLM-001, VLM-005, VLM-006, VLM-007, VLM-008 |
| MM-S002 | [大模型基础架构岗面经（二）](https://www.nowcoder.com/discuss/656279057671155712) | 牛客 / community | full | DST-001, DST-004, DST-007 |
| MM-S003 | [NLP 大模型春招记录](https://www.nowcoder.com/discuss/601149086300971008) | 牛客 / community | snippet | DST-003, DST-006, DST-013 |
| MM-S004 | [llm-rl-infra-interview](https://github.com/XFWang522/llm-rl-infra-interview) | GitHub / community | full | DST-005, DST-011, DST-012, DST-018, DST-019 |
| MM-S005 | [AI-Engineer-Interview-Questions / Multimodal](https://github.com/ombharatiya/AI-Engineer-Interview-Questions/blob/main/10-multimodal/README.md) | GitHub / community | full | OMM-005, OMM-006, OMM-008, OMM-009, OMM-012 |
| MM-S006 | [ai-engineering-interview-questions](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions/blob/main/README.md) | GitHub / community | full | VLM-016, VLM-017, VLM-018, VLM-019, VLM-021 |
| MM-S007 | [字节多模态算法面试题详解](https://zhuanlan.zhihu.com/p/2023042484121413435) | 知乎 / community | snippet | VLM-014 |
| MM-S008 | [awesome-ai-interviews / multimodal-ai](https://github.com/JustInCache/awesome-ai-interviews/blob/main/topics/11-multimodal-ai.md) | GitHub / community | snippet | 未采用候选 |
| MM-S010 | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/html/2103.00020v1) | arXiv / primary | full | VLM-001 |
| MM-S011 | [OpenAI CLIP 官方实现](https://github.com/openai/CLIP) | GitHub 官方仓库 / primary | full | VLM-002 |
| MM-S012 | [Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality](https://arxiv.org/abs/2204.03162) | arXiv / primary | snippet | VLM-003 |
| MM-S013 | [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) | arXiv / primary | snippet | VLM-004 |
| MM-S014 | [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/html/2201.12086v2) | arXiv / primary | full | VLM-005, VLM-006 |
| MM-S015 | [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/html/2301.12597v3) | arXiv / primary | full | VLM-007, VLM-008 |
| MM-S016 | [Visual Instruction Tuning](https://arxiv.org/html/2304.08485v2) | arXiv / primary | full | VLM-009, VLM-010 |
| MM-S017 | [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/html/2310.03744v2) | arXiv / primary | full | VLM-008, VLM-011 |
| MM-S018 | [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/html/2409.12191v2) | arXiv / primary | full | VLM-012, VLM-013 |
| MM-S019 | [Qwen2.5-VL Technical Report](https://arxiv.org/html/2502.13923v1) | arXiv / primary | full | VLM-014, OMM-002, DST-020 |
| MM-S020 | [InternVL2 Quick Start 官方文档](https://internvl.readthedocs.io/en/latest/internvl2.0/quick_start.html) | InternVL 官方文档 / primary | full | VLM-015, VLM-022 |
| MM-S021 | [DocVQA: A Dataset for VQA on Document Images](https://arxiv.org/abs/2007.00398) | arXiv / primary | snippet | VLM-016 |
| MM-S022 | [Hierarchical multimodal transformers for Multi-Page DocVQA](https://arxiv.org/abs/2212.05935) | arXiv / primary | snippet | VLM-017 |
| MM-S023 | [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355) | arXiv / primary | snippet | VLM-018 |
| MM-S024 | [Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](https://arxiv.org/abs/2311.16922) | arXiv / primary | snippet | VLM-019 |
| MM-S025 | [MMBench: Is Your Multi-modal Model an All-around Player?](https://arxiv.org/abs/2307.06281) | arXiv / primary | snippet | VLM-020 |
| MM-S026 | [LLaVA 官方仓库](https://github.com/haotian-liu/LLaVA) | GitHub 官方仓库 / primary | full | VLM-019, VLM-021 |
| MM-S027 | [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) | arXiv / primary | snippet | VLM-022 |
| MM-S028 | [Kosmos-2: Grounding Multimodal Large Language Models to the World](https://arxiv.org/abs/2306.14824) | arXiv / primary | snippet | VLM-023 |
| MM-S029 | [Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs](https://arxiv.org/abs/2406.16860) | arXiv / primary | snippet | VLM-024 |
| MM-S030 | [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449) | arXiv / primary | snippet | VLM-025 |
| MM-S031 | [Video-LLaVA: Learning United Visual Representation by Alignment Before Projection](https://arxiv.org/abs/2311.10122) | arXiv / primary | snippet | OMM-001 |
| MM-S032 | [VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs](https://arxiv.org/abs/2406.07476) | arXiv / primary | snippet | OMM-003 |
| MM-S033 | [QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries](https://arxiv.org/abs/2107.09609) | arXiv / primary | snippet | OMM-004 |
| MM-S034 | [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/html/2212.04356v1) | arXiv / primary | full | OMM-005 |
| MM-S035 | [Hugging Face Evaluate WER 官方实现](https://huggingface.co/spaces/evaluate-metric/wer/blob/main/wer.py) | Hugging Face 官方实现 / primary | full | OMM-006 |
| MM-S036 | [PyTorch CTCLoss 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CTCLoss.html) | PyTorch 官方文档 / primary | full | OMM-007 |
| MM-S037 | [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) | arXiv / primary | snippet | OMM-008 |
| MM-S038 | [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111) | arXiv / primary | snippet | OMM-009 |
| MM-S039 | [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759) | arXiv / primary | snippet | OMM-010 |
| MM-S040 | [Qwen2.5-Omni Technical Report](https://arxiv.org/html/2503.20215v1) | arXiv / primary | full | OMM-011 |
| MM-S041 | [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/html/2410.00037v2) | arXiv / primary | full | OMM-012, OMM-013 |
| MM-S042 | [Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction](https://arxiv.org/abs/2201.02184) | arXiv / primary | snippet | OMM-014 |
| MM-S043 | [OmniBench: Towards The Future of Universal Omni-Language Models](https://arxiv.org/abs/2409.15272) | arXiv / primary | snippet | OMM-015 |
| MM-S050 | [PyTorch DistributedDataParallel 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html) | PyTorch 官方文档 / primary | full | DST-001, DST-002 |
| MM-S051 | [PyTorch Distributed Data Parallel 设计说明](https://docs.pytorch.org/docs/2.14/notes/ddp.html) | PyTorch 官方文档 / primary | full | DST-012 |
| MM-S052 | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/html/1910.02054v3) | arXiv / primary | full | DST-003, DST-004 |
| MM-S053 | [DeepSpeed ZeRO 官方文档](https://deepspeed.readthedocs.io/en/latest/zero3.html) | DeepSpeed 官方文档 / primary | full | DST-004, DST-015 |
| MM-S054 | [PyTorch FSDP2 fully_shard 文档](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html) | PyTorch 官方文档 / primary | full | DST-005 |
| MM-S055 | [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/html/1909.08053v4) | arXiv / primary | full | DST-006 |
| MM-S056 | [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) | arXiv / primary | snippet | DST-007 |
| MM-S057 | [Megatron Bridge Parallelisms 官方文档](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/parallelisms.md) | GitHub 官方仓库 / primary | full | DST-008, DST-009 |
| MM-S058 | [Megatron Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) | NVIDIA 官方文档 / primary | full | DST-009, DST-010 |
| MM-S059 | [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) | NVIDIA 官方文档 / primary | full | DST-011 |
| MM-S060 | [PyTorch torch.utils.checkpoint 文档](https://docs.pytorch.org/docs/2.14/checkpoint.html) | PyTorch 官方文档 / primary | full | DST-013 |
| MM-S061 | [PyTorch AMP 文档](https://docs.pytorch.org/docs/2.14/amp.html) | PyTorch 官方文档 / primary | full | DST-014 |
| MM-S062 | [PyTorch Distributed Checkpoint 文档](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html) | PyTorch 官方文档 / primary | full | DST-016 |
| MM-S063 | [PyTorch torch.utils.data 文档](https://docs.pytorch.org/docs/2.14/data.html) | PyTorch 官方文档 / primary | full | DST-017 |
| MM-S064 | [NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html) | NVIDIA 官方文档 / primary | full | DST-018 |
| MM-S065 | [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/html/2204.02311v5) | arXiv / primary | full | DST-019 |
| MM-S066 | [PyTorch Profiler 文档](https://docs.pytorch.org/docs/2.14/profiler.html) | PyTorch 官方文档 / primary | full | DST-020 |
| MM-S067 | [PyTorch Distributed communication 文档](https://docs.pytorch.org/docs/2.14/distributed.html) | PyTorch 官方文档 / primary | full | DST-018 |

## 来源访问备注

- **MM-S001**：已读公开面经正文；页面为英文索引/翻译呈现。仅抽取问题主题；发布主体和面试经历为作者自述，未独立核实公司来源。
- **MM-S002**：已读公开正文；列出并行、显存、ZeRO、长文本主题，公司未指明。
- **MM-S003**：搜索返回正文片段；直接打开仅返回页面壳。只保留片段明确出现的 Adam 显存、ZeRO、重计算、TP 主题；经历为作者自述。
- **MM-S004**：公开题目整理集合，含作者延伸追问；不把其中的频率标签或公司归因继承为事实，仅取少量主题。
- **MM-S005**：公开备考主题集合，并非可核实公司面试记录；仅抽题目主题，不采纳附带答案或延迟数字。
- **MM-S006**：公开题目列表，非公司真题认证；仅抽少量 VQA、文档、幻觉、看图失败主题。
- **MM-S007**：仅读搜索返回片段，含 Qwen2.5-VL 结构问法；题目公司归因未独立核实，附带答案存在泛化表述，不作为答案依据。
- **MM-S008**：搜索摘要有多模态主题，直接打开 cache miss；未用于题目证据。
- **MM-S010**：已读 HTML 的训练伪代码，核对归一化特征、可训练温度、双向交叉熵；零样本接口另有官方实现。
- **MM-S011**：已读 README 的归一化特征、零样本分类与 encode_image/encode_text 示例。
- **MM-S012**：已读论文摘要，用于核对组合性评测动机；诊断步骤为编辑补充。
- **MM-S013**：已读论文摘要；HTML 版本读取失败。核对逐对 sigmoid 与无全局 softmax 归一化，未摘录实验表。
- **MM-S014**：已读公开 HTML，重点为三种目标与 CapFilt；仅概括机制，未搬运论文段落。
- **MM-S015**：已读 HTML 的 Q-Former、attention mask、两阶段和冻结骨干说明。
- **MM-S016**：已读 HTML 4.1/4.2，核对连接器、两阶段冻结策略和只对 assistant token 计算损失。
- **MM-S017**：已读 HTML，核对 LLaVA-1.5 的 MLP、336px 和任务数据/格式改进。
- **MM-S018**：已读 HTML 架构部分，核对动态分辨率、2×2 合并和 M-RoPE。
- **MM-S019**：已读 HTML，核对窗口与四层全局注意力、动态 FPS、绝对时间位置编码和训练负载平衡。
- **MM-S020**：已读 dynamic_preprocess 代码及多图接口；tile 规则是该版本实现，非所有 VLM 通则。
- **MM-S021**：已读论文摘要，核对文档问答需利用文本与布局；工程诊断为编辑补充。
- **MM-S022**：已读摘要，核对多页定位与分层建模；系统设计步骤为编辑整理。
- **MM-S023**：已读 POPE 论文摘要，核对对象存在性问答评测思路；不将其当全部幻觉指标。
- **MM-S024**：已读摘要，核对对比视觉条件降低语言先验影响；故障排查顺序为编辑建议。
- **MM-S025**：已读摘要，用于核对多维评估与选择题评测；推荐评估组合为编辑补充。
- **MM-S026**：已读 README 训练与 LoRA 说明；微调决策为编辑综合，不把原版冻结策略推广到所有版本。
- **MM-S027**：已读论文摘要，核对交错图文/视频输入和 few-shot 目标。
- **MM-S028**：已读摘要，核对语言片段与图像区域关联；坐标变换排查为编辑补充。
- **MM-S029**：已读摘要，核对视觉表征、数据和多模态评估共同影响能力；采样策略为编辑建议。
- **MM-S030**：已读摘要，核对文档视觉多向量检索；与生成式问答的对照为编辑总结。
- **MM-S031**：已读摘要，核对联合图像视频表征；抽帧预算为通用工程推导。
- **MM-S032**：已读摘要，核对 STC 连接器与音频分支；长视频方案为编辑比较。
- **MM-S033**：已读摘要，核对 Moment-DETR 输出片段坐标与显著性；tIoU 为区间交并比定义。
- **MM-S034**：已读 Whisper HTML 架构/预处理/长音频窗口部分。
- **MM-S035**：已读公式、动态字符串对齐说明及 corpus 聚合代码。
- **MM-S036**：已读 CTC 对齐假设、输入形状、blank 和长度约束。
- **MM-S037**：已读 EnCodec 论文摘要，核对神经音频压缩/RVQ 目标。
- **MM-S038**：已读 VALL-E 摘要，核对声学 token 建模与短音频条件。
- **MM-S039**：已读摘要，核对语音交互、音频分析和指令模式；未摘录详细训练超参数。
- **MM-S040**：已读 Thinker-Talker、TMRoPE、音频编码及流式解码章节。
- **MM-S041**：已读多流音频/文本和延迟章节；区分理论编解码延迟与端到端用户延迟。
- **MM-S042**：已读 AV-HuBERT 摘要，核对音视频自监督表征与唇读用途。
- **MM-S043**：已读摘要，核对图像、声音和文本联合理解评测；冲突与同步切片为编辑扩展。
- **MM-S050**：已读输入切分、梯度同步与 no_sync 说明。
- **MM-S051**：已读 reducer/bucket/autograd hook 和 collective 顺序说明。
- **MM-S052**：已读 HTML 状态显存分析；16 字节/参数基于 FP16 参数/梯度、FP32 主权重及 Adam 双状态的假设。
- **MM-S053**：已读 Stage 1/2/3、CPU/NVMe offload 和逐模块收集参数说明。
- **MM-S054**：已读按参数 DTensor 分片、reshard、all-gather 与梯度同步说明；不混用 FSDP1 包装 API。
- **MM-S055**：已读 HTML 的 MLP/attention 张量切分，核对列并行与行并行的通信位置。
- **MM-S056**：已读摘要，核对张量/流水线/数据并行联合扩展；bubble 公式仅为理想调度近似。
- **MM-S057**：已读 SP/CP 的作用范围和 sequence_parallel 对 TP 的要求。
- **MM-S058**：已读 DP/TP/PP/CP/EP 的作用维度，MoE 与通信配置；不将指南示例当性能保证。
- **MM-S059**：已读 all-reduce、reduce-scatter、all-gather 及等价关系；ring 通信量由算法步骤自行推导。
- **MM-S060**：已读重计算/RNG/可重入与非可重入行为。
- **MM-S061**：已读 autocast 和 GradScaler；dtype 数值范围按 IEEE 表示解释。
- **MM-S062**：已读分片保存/加载和 state_dict 说明；完整恢复状态列表为编辑综合。
- **MM-S063**：已读 DistributedSampler、set_epoch、drop_last 与 worker 行为。
- **MM-S064**：已读故障排查入口；进程组一致性另参照 PyTorch distributed 文档。
- **MM-S065**：已读 HTML 训练效率与 MFU 定义；6NT 只作为忽略额外项的稠密 Transformer 近似。
- **MM-S066**：已读 CPU/CUDA 活动、shape、memory 和 trace 能力；瓶颈定位建议为编辑整理。
- **MM-S067**：已读 collective 一致性、调试和 monitored_barrier；用于 NCCL 挂起诊断。

## 分片自检

已验证：60 道题 ID 唯一，分类数量为 25/15/20；每题有至少一个 primary 引用；所有引用和社区证据 ID 都存在；quick 为 60–140 字符；detail 为 3–5 条；pitfalls 与 followups 各为 1–2 条；editorial 与有无社区证据一致；JSON 可以严格解析。跨分片技术复审由仓库集成阶段进行。

