# 本轮补充的独立同伴审阅

日期：2026-10-02。审阅范围：`tools/supplement.py`、`tools/build.py`、`tools/validate.py`、`tools/run_checks.py`、`research/supplement-request-2026-10-02.json`、`data/user_requested_sources.json`，并读 `tools/common.py` 的来源限制。初审只做只读检查与内存测试，未改其他文件。

最终状态：下述工具问题与新增答案问题均已修复并复核，无剩余待修项。225题全库的结构校验和补充映射校验均返回空错误列表，U00—U14共15项已实际关联；本轮对14道新增/修改Core题、3道多模态题及新增关联FT-005逐题审阅。指定知乎文章的标题与“核心30问”目录仍未取得，准确完成度为0/30，不能把这15项补题等同于原文章30问已完成。

## 工具初审结论

构建和独立校验均调用 `validate_supplement`，缺少用户条目会阻止构建。新增来源分片采用标准 `{sources, questions}`，能被全仓库加载。生成清单链接指向既有章节锚点，不复制答案，保留重复主题到多个题号的映射。

请求记录有 U00—U14 共 15 项，分别覆盖 Q-Former 总问题、MLP/淘汰说法、交叉注意力、MQA/GQA、packing、CE/KL、熵、秩/特征值、LLaVA/InternVL、SFT loss/mask、LoRA、QLoRA、位置编码、RoPE、mask attention。初审时 core/mm 正在写入映射，当前 15 项均未关联；校验正确报告缺项。这不是最终答案覆盖审阅结果。

SUP-C001 明确为 `blocked`，文章状态 `pending_source_text`，没有题目目录；notes 明确标题和30问未读取。当前未以该链接提供 `community_evidence` 或技术依据。`common.validate` 也禁止 blocked 来源作为这些引用，访问边界诚实。

## S-01 [P2，已修复] blocked 文章的禁止提取依赖 status 字符串

位置：`tools/supplement.py` 的 `validate_supplement` 文章检查。

初审版本只在 `status == pending_source_text` 时拒绝非空 `article.items`。内存复现：保持 `access=blocked`，改 `status=completed` 并填一个合成 item，15 项用户映射齐全时返回空错误列表。`build_supplement` 的 else 分支随即展示这些未经读取的条目，绕过“受阻来源不得推断题目”的规则；`common.validate` 无法捕获 request 文件自己的 article.items。

修法：验证 status 枚举及状态/访问一致性；`access=blocked` 无条件禁止声称抽取条目。若未来目录来自可读二手转载，应另记录可读的证据 source_id 和对应 item provenance，保留原知乎 blocked 状态。现有 request 文件本身没有触发该问题。

## S-02 [P2，已修复] 错误类型在结构检查前抛异常

位置：`tools/supplement.py` 的 `mapped_questions` / `validate_supplement`。

`validate_supplement` 在确认 `user_supplement_keys` 类型前调用 `mapped_questions`。内存测试把一个字段改成 `None`，得到 `TypeError: 'NoneType' object is not iterable`，而非可读校验错误；包含 list/dict 键时也会因不可哈希而异常。

修法：先要求 list[str]，检查去重及允许键，再构造映射；映射辅助函数遇无效字段可跳过，由校验统一报告。有效15项/漏一项的内存测试均符合预期。

## 初审检查记录

- 完整的15项内存映射：无错误。
- 删除最后一项：正确报告 U14 缺 canonical answer。
- `user_supplement_keys=None`：复现 S-02。
- blocked+completed+非空 article.items：复现 S-01。
- 没有运行会重写章节、索引或 validation.json 的 build/run_checks；完整构建由根任务负责。

## 新增答案事实与数学审阅

初审阶段core/mm尚在写入，最终逐题审阅结果见后续各节。

## 工具修复复核

根任务已修 S-01/S-02，2026-10-02 重新运行内存测试确认：合法15项映射通过；None 和含 dict 的键列表均得到明确 list-of-strings 错误而非异常；blocked+merged+条目被拒绝；未知 completed 状态被拒绝。当前 pending_source_text 且目录缺失的实际记录仍保持诚实。两个工具问题标记为已修复。

## 多模态答案复核

已逐题审阅 VLM-007、VLM-008、VLM-026，并再次打开 [BLIP-2 第3节正文](https://arxiv.org/html/2301.12597v3)、[MiniCPM-V4.5 第2.1.1节](https://arxiv.org/html/2509.18154v1)、[InternVL2.5 quickstart](https://internvl.readthedocs.io/en/latest/internvl2.5/quick_start.html)、InternVL2_5-8B [配置](https://huggingface.co/OpenGVLab/InternVL2_5-8B/blob/main/config.json)和[建模源码](https://huggingface.co/OpenGVLab/InternVL2_5-8B/blob/main/modeling_internvl_chat.py)，及 [LLaVA-1.5-7B 配置](https://huggingface.co/liuhaotian/llava-v1.5-7b/blob/main/config.json)。没有运行这些模型。

- U00→VLM-007/008：查询来源、输出形状、32为实验配置、冻结参数但保留输入梯度、两阶段训练与适用场景均已覆盖。
- U01→VLM-008：MLP逐token映射不主动减数量；Q-Former与更广义Resampler区分正确。已用明确实例反驳无证据的普遍淘汰说法，没有据此宣称商业份额。
- U08→VLM-026：固定两个checkpoint再比较，涵盖切图、缩略图、连接器、token预算、序列布局和多图/视频接口；没有将后续LLaVA版本能力倒填到1.5。
- 独立计算确认576、256/tile及12局部块+thumbnail=3328。InternVL源码ratio=0.5为空间缩小、通道重排，CLS移除和MLP前LayerNorm与说明一致。quickstart在多块时额外追加缩略图，局部块max_num与总块数区分正确。
- Sources scopes 限定论文版本、checkpoint、main读取日和静态实现；没有冒称运行验证或证明行业普遍趋势。所有技术引用均为primary，未使用SUP-C001。

M-01 可选精度建议已采纳：VLM-007最初用“因果掩码”概括ITG，作者已改为混合掩码的完整可见关系（query互看但不看文本、文本看全部query和历史文本），与BLIP-2正文一致。本组无待修事实/数学问题。

## Core答案与数学复核

已读最终TFM-003/008/013/016/017/018/019、PRE-010、FT-001/002/006/007/008、INF-003及新增关联FT-005。独立重新读取官方[SDPA文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)、[ForCausalLMLoss源码](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/loss/loss_utils.py)、[TRL packing文档](https://huggingface.co/docs/trl/main/en/reducing_memory_usage)、[PEFT LoRA源码](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/layer.py)、[LoRA论文第4节](https://arxiv.org/pdf/2106.09685)、[QLoRA论文第3节](https://arxiv.org/pdf/2305.14314)、[MIT信息论第1章](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/2243edffb30f57181ed97dcb77691580_MIT6_441S16_chapter_1.pdf)、[Stanford特征值讲义](https://ee263.stanford.edu/lectures/eig.pdf)、[LAPACK Schur算法概述](https://www.netlib.org/lapack/lug/node50.html)、[Llama固定版本源码](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/llama/modeling_llama.py)和[KLDivLoss文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.KLDivLoss.html)的相关函数/章节。没有运行GPU训练或模型推理。

- 注意力：cross-attention输出长度由Q决定、投影后的Q/K维度约束正确。GQA保持query头数，缓存字节公式按总token数计，物理repeat和TP复制风险已说明。cached causal采用真实历史偏移；SDPA与MHA的bool语义区别与2.14文档一致。
- Packing/SFT：独立段同时隔离attention及shifted标签。短序列`[P0,P1,A0,A1,EOS]`的监督对齐、首answer由末prompt位置预测、EOS和padding按位置区分均正确；冻结/不计直接loss与计算图梯度区分合理。没有声称EOS或position重置自动隔离任意后端。
- LoRA：独立推导`dA=sB^T G`与`dB=sGA^T`，初始B为零的梯度结论正确。原论文高斯A、PEFT普通Linear的Kaiming A、Embedding例外及变体已区分；标准和rsLoRA缩放、dropout输入位置、rank容量与尺度混杂均正确。
- QLoRA：NF4存储与BF16等计算精度分开；double quantization压缩量化常数，paged optimizer管理状态峰值，均与正文一致。效果/速度及合并量化风险没有无条件泛化。
- 熵/位置：fixed-p的CE/KL等价前提、条件熵平均范围、模型输出熵与真实数据熵区分正确。RoPE的相对差符号、正交性、已旋转K写入缓存顺序与固定版本实现一致；位置编码比较没有把可计算新位置当成可靠外推保证。

### C-01 [P2，已修复] CE梯度变量混淆

TFM-013原detail将`q−p`写成对预测概率`q`的梯度。已反馈作者并改成：固定归一化p且`q=softmax(z)`时，对logits z的梯度是`q−p`，对正概率`q_i`的梯度是`−p_i/q_i`。独立解析推导及三分类软目标的中心差分复核通过，logits梯度最大差约`1.24e−10`。

### C-02 [覆盖建议，已修复] 特征值计算方法需直接回答

U07问“分别如何计算”。TFM-018原版解释定义和关系，但未写求特征值的操作步骤；作者已补特征多项式、解特征向量、手算例、通用数值Hessenberg/Schur流程，并补SUP-P001直接依据。独立检验`[[2,1],[0,3]]`的λ=2、3及向量(1,0)、(1,1)，代入残差均为零。LAPACK所述阶段及实Schur的2×2块处理与答案一致。

### C-03 [精度建议，已修复] 秩与谱关系的前提

新quick曾用“可对角化才能”将充分条件表达为必要条件；现为“可对角化时”充分条件。最后detail曾把包括零值的σ平方一概称为非零特征值；现准确区分非零谱和矩形情形需补的零特征值。不可对角化nilpotent反例、满秩判定、数值容差均正确。最终无剩余数学待修项。

## 15项实际映射与来源边界终检

| 请求 | 最终题号 | 审阅结果 |
|---|---|---|
| U00 Q-Former原理/比较/应用 | VLM-007、VLM-008 | 覆盖 |
| U01 两层MLP及淘汰说法 | VLM-008 | 覆盖，并限制普遍趋势推断 |
| U02 Cross-attention | TFM-016 | 覆盖 |
| U03 MQA/GQA | INF-003 | 覆盖 |
| U04 SFT packing | PRE-010 | 覆盖 |
| U05 CE与KL | TFM-013 | 覆盖，梯度变量已修 |
| U06 熵 | TFM-017 | 覆盖 |
| U07 秩/特征值及计算 | TFM-018 | 覆盖，计算步骤与前提已补 |
| U08 LLaVA/InternVL | VLM-026 | 覆盖，固定版本比较 |
| U09 预训练/SFT loss及mask | FT-001、FT-002 | 覆盖 |
| U10 LoRA原理/初始化/调参 | FT-005、FT-006、FT-007 | 覆盖 |
| U11 LoRA/QLoRA | FT-008 | 覆盖 |
| U12 位置编码 | TFM-019 | 覆盖 |
| U13 RoPE | TFM-008 | 覆盖 |
| U14 Mask attention | TFM-003 | 覆盖 |

最终只读加载：262来源、225题；`common.validate`与`validate_supplement`均返回空错误列表。扫描全库blocked来源与`reference_ids`/`community_evidence`的交集，两种引用均为零。数学答案技术引用为primary；源码固定版本与main读取日期清楚，摘要来源未被用来证明具体实现公式或行业普遍结论。

SUP-C001保留`blocked`与`pending_source_text`，request无题目目录；没有以通用题库补出文章“30问”，没有声称已读正文，也没有冒称标题已验证。该原文章内容准确状态为0/30未取得，15项已处理指用户独立补充问题。完整构建、生成目录与最终run_checks由根任务运行，本审阅只写此报告。
