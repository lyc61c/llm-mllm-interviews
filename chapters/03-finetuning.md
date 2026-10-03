# SFT、LoRA 与参数高效微调

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 题目

- [FT-001 · 预训练与 SFT 的 loss 有何差异，label shift 如何对齐？](#ft-001)
- [FT-002 · SFT 怎样只对回答部分计算 loss？](#ft-002)
- [FT-003 · chat template 错配会导致哪些问题？](#ft-003)
- [FT-004 · 全参数微调、LoRA、Adapter 和 Prefix-Tuning 怎样选择？](#ft-004)
- [FT-005 · LoRA 的低秩更新公式及可训练参数量是什么？](#ft-005)
- [FT-006 · LoRA 怎样初始化，r、alpha 与 dropout 各控制什么？](#ft-006)
- [FT-007 · LoRA 的 rank、alpha、dropout 和 target_modules 应怎么调？](#ft-007)
- [FT-008 · LoRA 与 QLoRA 有何区别，NF4、双重量化与分页优化器做什么？](#ft-008)
- [FT-009 · Adapter、Prompt-Tuning 与 Prefix-Tuning 有何差别？](#ft-009)
- [FT-010 · SFT 数据量越大越好吗，怎样构建高质量指令集？](#ft-010)
- [FT-011 · 用模型生成 SFT 数据怎样避免错误与同质化？](#ft-011)
- [FT-012 · 微调后的灾难性遗忘怎样发现和缓解？](#ft-012)
- [FT-013 · 梯度累积等价于大 batch 吗？变长样本怎么归一化？](#ft-013)
- [FT-014 · gradient checkpointing 节省什么，为何会变慢？](#ft-014)
- [FT-015 · SFT loss 下降但任务效果变差，应该怎样排查？](#ft-015)

<a id="ft-001"></a>
## FT-001 · 预训练与 SFT 的 loss 有何差异，label shift 如何对齐？

**L1**

### 答案

预训练与常规 SFT 都可使用 next-token 交叉熵，主要差异是数据分布与监督范围。序列 $x_0,\ldots,x_{T-1}$ 的 logits 第 $t$ 行基于 $x_{\le t}$，预测 $x_{t+1}$；可对齐 `logits[:,:-1]` 和 `labels[:,1:]`，或用尾部补 ignore 的等价实现，shift 只能由一处负责。

例如输入 `[P0,P1,A0,A1,EOS]`，未 shift 的标签为 `[-100,-100,A0,A1,EOS]`。末 prompt 的 P1 logits 监督首回答 A0，A0 监督 A1，A1 监督 EOS；P0 到 P1 的目标忽略，末 EOS logits 没有下一标签。若额外屏蔽 P1 logits，就会丢掉首回答监督。

预训练通常监督有效文本，排除 padding、无后续目标和配方指定边界；SFT 可以只监督回答、所有 assistant 回合、最后回合或全序列，由训练配方决定，也能学习知识和内容。Loss mask 决定目标贡献，attention mask 决定可读上下文；回答梯度仍可经 prompt 表示回传到共享参数。两种训练都需 causal 约束与正确 packing 隔离。

常规损失按有效 token 平均；若改为每样本等权或加入辅助项，目标也随之改变。比较效果时应核对模板、监督范围、数据和任务指标，不能直接用不同 mask 下的 loss 排名。

$$
\begin{aligned}z_t&\longrightarrow x_{t+1}\\\mathcal L&=-\frac{\sum_{t=0}^{T-2}m_{t+1}\log\operatorname{softmax}(z_t)_{x_{t+1}}}{\sum_{t=0}^{T-2}m_{t+1}}\end{aligned}
$$

### 易错点

- 把 SFT 写成另一种分类 loss，或断言 SFT 永远只能计回答 loss。
- 按预测 logits 所在位置屏蔽 prompt，错误丢掉 prompt→首 answer 的有效预测。

### 追问

- 多轮对话只监督最后一轮时，前几轮回答是否仍提供上下文？
- 预训练的全 token loss 和 SFT 的回答 loss 能直接比较吗？

<a id="ft-002"></a>
## FT-002 · SFT 怎样只对回答部分计算 loss？

**L1**

### 答案

只监督回答时，在最终 token 序列中把 prompt、padding 和其他不监督位置的 labels 设为 `ignore_index`，例如 `-100`，回答标签保留真实 ID。应先应用模型 chat template 再标注，多轮可选择所有 assistant 回合或最后回合，并明确正文、角色结束及 EOS 的监督范围，避免混用字符和 token 边界。

若首 answer 在 $j$，保留 `labels[j]=input_ids[j]`，prompt 的 `labels[:j]=-100`；shift 后 `logits[j-1]` 预测首回答。因此只忽略 prompt 目标，不删除末 prompt 的 logits，模型内部 shift 时 collator 仍保持标签与输入同位置。合法 prompt 的 attention mask 继续有效，回答 loss 的梯度也能经过它的上下文路径。

Padding 与真实 EOS 即使共用 ID，也必须按位置区分，不能按 EOS ID 全部忽略；EOS 可以是有效停止监督。独立 packing 段首目标不能由上一段末 logits 预测，还需块状可见性或后端段边界。最后用小序列逐项检查 logits 索引、目标、mask、有效数量和 loss。

$$
\mathcal L=-\frac{\sum_{t=0}^{T-2}m_{t+1}\log p_\theta(x_{t+1}\mid x_{\le t})}{\sum_{t=0}^{T-2}m_{t+1}}
$$

### 易错点

- 把 prompt 的 `attention_mask` 设为 0，或把末 prompt logit 的首回答监督删掉。
- collator 和模型内部各 shift 一次，或把 `pad=EOS` 当作所有 EOS 均为 padding。

### 追问

- 如何验证多轮 assistant-only mask 与 chat template 的 generation 区间？
- 没有任何有效回答目标的样本应怎样过滤或处理？

<a id="ft-003"></a>
## FT-003 · chat template 错配会导致哪些问题？

**L2**

### 答案

Chat template 将角色与消息转成模型训练过的 token 序列，错配可能导致续写用户、角色混乱或无法停止。Tokenizer、`chat_template` 和特殊 token 配置应与模型、适配器一起版本化，训练和部署保持一致。

训练通常不额外添加新 assistant 回复的起始提示，生成时按模板需要添加 generation prompt；先渲染文本再编码时检查 `add_special_tokens`，避免重复 BOS/EOS。用短对话核对 token ID、角色边界、loss mask 与停止行为，再覆盖多轮对话。

### 易错点

- 只检查字符串外观，忽略底层 token ID。
- 把所有模型都换成同一套 ChatML 而无需训练。

### 追问

- 为什么重复 EOS 可能损害生成？
- 模型已有模板时还需要 clone 新模板吗？

<a id="ft-004"></a>
## FT-004 · 全参数微调、LoRA、Adapter 和 Prefix-Tuning 怎样选择？

**L1**

### 答案

全参数微调更新自由度大，但梯度和优化器状态开销更高；PEFT 冻结大部分底座，训练少量增量，适合资源受限或多任务切换。选型要结合任务差异、数据量、显存与部署，PEFT 不保证总能达到全参数效果。

LoRA 给线性层加低秩更新，合适时可合并回权重；Adapter 加任务独立模块，但可能增加算子和时延；Prefix-Tuning 学习连续前缀，通常提供额外 attention 键值条件。冻结底座仍需前向和用于反传的激活，参数减少比例不能直接当作显存减少比例。

### 易错点

- 训练参数减少 99% 就声称训练显存减少 99%。
- 仅看单次训练成本，忽略多适配器服务和合并成本。

### 追问

- 领域迁移很大时为什么全参数值得尝试？
- 多任务适配器如何与 prefix cache 协同？

<a id="ft-005"></a>
## FT-005 · LoRA 的低秩更新公式及可训练参数量是什么？

**L1**

### 答案

LoRA 冻结 $W_0\in\mathbb R^{d_{\rm out}\times d_{\rm in}}$，学习 $\Delta W=(\alpha/r)BA$，其中 $A\in\mathbb R^{r\times d_{\rm in}}$、$B\in\mathbb R^{d_{\rm out}\times r}$。更新秩至多为 $r$，可训练参数为 $r(d_{\rm in}+d_{\rm out})$，不要求底座或整个模型低秩。

前向为 $W_0x+(\alpha/r)BAx$，梯度能经过冻结底座传播到需要训练的模块。标准浮点部署可合并增量，免去额外小投影；需要多适配器切换时则未必合并。LoRA 主要节省权重梯度与优化器状态，长序列激活仍可能主导显存。

$$
\begin{aligned}\Delta W&=\frac{\alpha}{r}BA\\N_{\rm train}&=r(d_{\rm in}+d_{\rm out})\\\operatorname{rank}(\Delta W)&\le r\end{aligned}
$$

### 易错点

- 把 LoRA 说成对原模型权重做截断 SVD。
- 漏掉不同目标模块的数量，误报总训练参数量。

### 追问

- 如何估算同时对 Q/K/V/O 与 FFN 加 LoRA 的参数量？
- 量化底座合并 LoRA 有什么精度风险？

<a id="ft-006"></a>
## FT-006 · LoRA 怎样初始化，r、alpha 与 dropout 各控制什么？

**L2**

### 答案

原始 LoRA 使用随机高斯 $A$、零 $B$，使初始 $\Delta W=sBA=0$。PEFT 普通 Linear 默认采用 Kaiming-uniform A、零 B，Gaussian 选项也使用零 B；Embedding 等模块可能有不同约定，其他初始化则应按方法确认。

令 $G=\partial L/\partial\Delta W$，有 $\partial L/\partial A=sB^\top G$、$\partial L/\partial B=sGA^\top$。初始 B 为零时，纯低秩分支的 A 梯度为零，B 通常先更新，随后 A 获得梯度；A/B 同时为零则纯 BA 路径无法启动，零 A 加随机 B 是可启动但轨迹不同的方案。PiSSA、EVA、LoftQ 等使用权重、激活或量化信息，要核对是否改动底座及是否保持初始函数；`init_lora_weights=False` 通常随机初始化 A/B，不再是 no-op。

$r$ 控制秩上界与参数量，标准 $s=\alpha/r$ 控制分支尺度，rsLoRA 使用 $\alpha/\sqrt r$。固定 alpha 改 rank 同时改变容量和缩放，alpha 也不等于学习率。普通 Linear 的 dropout 施加在低秩分支输入，训练时正则、eval 时关闭，底座分支保留。应联合消融 rank、alpha、dropout、目标模块与学习率，没有通用最佳 rank 或 alpha:rank 比例。

$$
\begin{aligned}\Delta W&=sBA,\quad\operatorname{rank}(\Delta W)\le r\\s_{\rm standard}&=\frac{\alpha}{r},\quad s_{\rm rsLoRA}=\frac{\alpha}{\sqrt r}\\G&=\frac{\partial L}{\partial\Delta W}\\\frac{\partial L}{\partial A}&=sB^\top G,\quad\frac{\partial L}{\partial B}=sGA^\top\end{aligned}
$$

### 易错点

- 把原论文高斯初始化、PEFT Linear 默认和所有新初始化变体说成完全相同。
- 双零 A/B、默认 no-op 与训练阶段梯度为零混为一谈，或把 $\alpha$ 当作优化器学习率。

### 追问

- 如何检查添加 adapter 前后初始输出一致，以及第一步 A/B 的梯度？
- 增大 r 时怎样控制缩放和优化超参数以比较容量收益？

<a id="ft-007"></a>
## FT-007 · LoRA 的 rank、alpha、dropout 和 target_modules 应怎么调？

**L2**

### 答案

LoRA 调参需要一起考虑更新容量、分支尺度、正则与适配位置。Rank 的参数预算线性增长，更新秩仅满足 $\operatorname{rank}(\Delta W)\le r$，实际容量收益可能饱和或过拟合；比较时控制数据、有效 token、目标模块和学习率，并看验证指标、参数量与吞吐。

标准缩放为 $s=\alpha/r$，`use_rslora=True` 为 $\alpha/\sqrt r$。固定 alpha 增大 rank 会减小标准分支尺度，固定 alpha/r 也不能保证优化轨迹相同，因此可在同一缩放家族内分别消融容量与尺度。普通 Linear 分支是 $sBA\operatorname{Dropout}(x)$，它不删除底座权重，eval 时关闭；小数据可检查正则收益，过大 dropout 也会欠拟合，且它与全模型 attention/hidden dropout 配置不同。

`target_modules` 决定更新位置，可比较 attention 投影、MLP 或 `all-linear`，并核对输出层排除、embedding/LM head 保存及额外 trainable bias。常见 Q/V 实验不意味着 LoRA 只支持 Q/V。记录 PEFT 版本、初始化、缩放变体、优化器和成本；库默认不是数据相关实验中的最优值。

$$
\begin{aligned}h&=W_0x+sBA\operatorname{Dropout}(x)\\N_{\rm train}&\approx r(d_{\rm in}+d_{\rm out})\\s_{\rm standard}&=\alpha/r,\quad s_{\rm rsLoRA}=\alpha/\sqrt r\end{aligned}
$$

### 易错点

- 给所有模型套 $r=8$、$\alpha=2r$ 等万能值，或声称 dropout 在推理时继续增强多样性。
- 改变 rank/模块/缩放后仍把消融结果归因于单个因素。

### 追问

- 扩展 `target_modules` 和增大 r，怎样在相同训练参数预算下比较？
- 为什么相同 $\alpha/r$ 仍不能保证两个 rank 的有效更新相同？

<a id="ft-008"></a>
## FT-008 · LoRA 与 QLoRA 有何区别，NF4、双重量化与分页优化器做什么？

**L2**

### 答案

LoRA 本身不要求量化，通常在 FP16/BF16 底座上训练低秩分支；QLoRA 将冻结底座以 4-bit 形式存储，计算时按块反量化到较高精度，再与可训练 adapter 组合。梯度通过底座计算传到 adapter，但不更新量化底座；权重存储位宽与 GEMM 的 compute dtype 必须区分。

NF4 为近似正态权重设计非均匀码本和块 scale，是有损量化，不等同于线性 INT4。Double Quantization 压缩第一层量化的 scale 等常数，不是再次量化同一权重以提高精度；加载 4-bit 权重也不自动意味着复现完整 QLoRA 配方。

Paged optimizer 借助分页或统一内存管理优化器状态峰值，缓解瞬时显存压力，也可能产生 CPU/GPU 迁移开销；它不压缩激活。长序列的 attention 激活、反量化工作区、adapter 梯度和状态、batch 与 checkpoint 策略仍可决定是否 OOM。

QLoRA 主要改善底座存储和训练可行性，速度与精度要按硬件、内核、权重分布和任务实测。合并部署可先反量化底座、加 $sBA$ 再量化，但会引入新误差，不能直接把浮点增量加到整数码；应比较未合并、合并与重新量化的输出。

$$
h=\operatorname{Dequantize}(Q(W_0))x+sBAx
$$

### 易错点

- QLoRA 训练全部 INT4 权重，或把 4-bit 权重容量当作训练总显存。
- LoRA 与 QLoRA 是互斥算法，或 QLoRA 在所有任务都与浮点 LoRA 等精度且更快。

### 追问

- LoRA/QLoRA 的公平比较要统一哪些模块、数据、初始化及评测条件？
- 为什么长上下文训练的显存瓶颈可能从权重转移到激活？

<a id="ft-009"></a>
## FT-009 · Adapter、Prompt-Tuning 与 Prefix-Tuning 有何差别？

**L2**

### 答案

Adapter 在网络中增加可训练模块，常由降维、非线性、升维和残差组成，参数量由瓶颈维度决定。Prompt-Tuning 学习输入层软提示，虚拟 token 不必对应可读词，也不等同于手写文本 prompt。Prefix-Tuning 给 attention 增加连续前缀，可经重参数化产生多层 K/V，具体实现依架构而异。

这些方法都可冻结底座，但插入位置、容量与开销不同：软前缀占序列或 KV 预算，Adapter 的额外算子可能影响时延与内核融合。

### 易错点

- 把所有 soft prompt 方法都叫输入 embedding 调参。
- 说 Prefix-Tuning 完全没有服务时开销。

### 追问

- 参数量相同时哪些实验能比较表达能力？
- 离散 prompt 与软提示怎样保存和部署？

<a id="ft-010"></a>
## FT-010 · SFT 数据量越大越好吗，怎样构建高质量指令集？

**L2**

### 答案

SFT 数据是否有效取决于答案正确性、任务覆盖与真实请求分布，增加数量不一定改善效果。可先用高质量种子构建任务、难度、语言、格式和拒答边界矩阵，再按实际分布补齐，清理重复和冲突；少样本论文结果不能推广为所有场景都只需少量数据。

可验证任务用规则或执行器核验，开放回答用明确标准和抽审。按来源、模板或任务组切分，避免近重复跨训练与验证；在固定 token 预算下比较质量和配比，同时记录通用回归及边缘案例。

### 易错点

- LIMA 使用少量优质数据，就断言规模不重要。
- 单纯把相同模板扩写数万条当成丰富覆盖。

### 追问

- 如何区分模型底座能力不足与 SFT 覆盖不足？
- 同一 prompt 多个冲突答案怎么处理？

<a id="ft-011"></a>
## FT-011 · 用模型生成 SFT 数据怎样避免错误与同质化？

**L2**

### 答案

模型合成 SFT 数据能降低标注成本，也会把教师的错误、风格偏差和有限覆盖传给学生。应先确定能力缺口，再生成多样候选、验证正确性与去重，保留真实或人工数据对照；模型自评高分不能当作事实正确的证据。

Self-Instruct 以种子任务生成指令和回答后过滤无效、相似样本。代码、数学和结构化任务优先用执行器、测试或确定性规则，多提示、来源与难度组合还需统计模板和语义重复。用独立真实验证集比较真实/合成配比，并追踪教师的系统性错误。

### 易错点

- 只用同一个模型生成并自评分，认为已独立验证。
- 把生成条数当作有效独立样本数。

### 追问

- 教师强于学生时怎样控制任务难度？
- 蒸馏风格和蒸馏能力有什么区别？

<a id="ft-012"></a>
## FT-012 · 微调后的灾难性遗忘怎样发现和缓解？

**L2**

### 答案

灾难性遗忘表现为新任务改善、旧能力明显退化。用固定通用与领域评测分别检查新旧任务、指令格式和边界行为，先排除模板错配等实现问题，再考虑混入代表性旧样本、降低学习率和更新范围、早停或正则约束。

Replay 的配比会影响领域收益，参数重要性正则等持续学习方法也需按任务和成本验证。PEFT 可以减少部分风险，但冻结底座不保证带适配器的输出不退化。保留底座和适配器版本可回滚，仍应完成上线前回归。

### 易错点

- LoRA 不修改底座，因此部署输出永远不可能遗忘。
- 只用新的 validation set 做早停。

### 追问

- 混入通用数据为何可能削弱领域适配？
- 怎样区分真实遗忘与生成参数变化？

<a id="ft-013"></a>
## FT-013 · 梯度累积等价于大 batch 吗？变长样本怎么归一化？

**L2**

### 答案

梯度累积汇总多个 micro-batch 后再更新，减少单次激活显存；有效 batch 约为 micro-batch、累积次数和数据并行数的乘积，最后不足的窗口要单独处理。正确归一化时可近似大 batch，dropout 与浮点累加顺序等仍会造成差异。

对 token 平均目标，应累加窗口内有效 token 的 NLL，再除以整个窗口的有效 token 总数，不能等权平均长度不同的 micro-batch 的 token 平均 loss。DDP 的梯度平均和框架自动缩放会影响系数，不能重复除世界大小。Optimizer 和 scheduler 按真实更新推进，累积中可用 `no_sync` 减少通信。

$$
\begin{aligned}B_{\rm effective}&=B_{\rm micro}K_{\rm accum}N_{\rm DP}\\\mathcal L_{\rm window}&=\frac{\sum_{k=1}^{K_{\rm accum}}\sum_{t\in\mathcal T_k}\operatorname{NLL}_{k,t}}{\sum_{k=1}^{K_{\rm accum}}|\mathcal T_k|}\end{aligned}
$$

### 易错点

- 每个 batch loss 除累积次数，就认为变长序列始终等价。
- 累积时忘记清零梯度或每个 micro-step 都更新 scheduler。

### 追问

- 如何验证累积与一次大 batch 的梯度接近？
- 为什么梯度累积降低显存却可能降低吞吐？

<a id="ft-014"></a>
## FT-014 · gradient checkpointing 节省什么，为何会变慢？

**L2**

### 答案

Gradient checkpointing 只保留若干边界激活，在反向需要时重算内部前向片段，以额外计算换激活显存。节省程度取决于切分粒度，它不减少参数或优化器状态，主要用于需要反向的训练；冻结底座的 LoRA 也需传播到适配器，长序列场景同样可能受益。

重算要与原前向一致，随机 dropout、状态修改、分支和设备迁移都应遵守框架约束。应确认当前版本的 reentrant/non-reentrant 行为，并比较峰值显存与有效 token/s，不能默认所有函数的重算天然等价。

### 易错点

- 把 gradient checkpointing 当作保存 checkpoint 文件。
- 无反向的常规推理用它来节省 KV cache。

### 追问

- 为什么 LoRA 的输入梯度配置会影响 checkpointing？
- 如何选择重算粒度与激活卸载？

<a id="ft-015"></a>
## FT-015 · SFT loss 下降但任务效果变差，应该怎样排查？

**L2**

### 答案

SFT loss 降低只表示更贴合训练目标，不能证明事实性或真实任务改善。先抽查完整样本的最终 token、监督区间、截断和 EOS，排除模板、mask、样本失效或验证泄漏，再固定生成配置比较 train/validation loss 与任务指标。

按长度、领域和格式分析失败案例，检查高重复、错误答案、教师风格偏差及训练/部署 prompt 差异。确认过拟合或分布错配后，再以单变量实验尝试降低学习率、早停、减 rank、正则或调整配比。

### 易错点

- 一看到效果差就继续加 epoch。
- 只观察总体 loss，不看大量被 mask 或截断的样本。

### 追问

- 训练 loss 接近 0 是否一定是好事？
- 如何区分过拟合与生成参数造成的差异？

## 参考资料

- [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [torch.nn.CrossEntropyLoss — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [TRL SFT Trainer — loss, labels and packing](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Transformers v4.57.1 official ForCausalLMLoss](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/loss/loss_utils.py)
- [Chat templates](https://huggingface.co/docs/transformers/chat_templating)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- [PEFT LoRA — initialization and LoraConfig](https://huggingface.co/docs/peft/main/en/package_reference/lora)
- [PEFT official implementation: LoRA layer](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/layer.py)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)
- [PEFT: Prompt tuning](https://huggingface.co/docs/peft/main/en/package_reference/prompt_tuning)
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
- [Continual Learning Through Synaptic Intelligence](https://arxiv.org/abs/1703.04200)
- [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)
- [Performing gradient accumulation with Accelerate](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)
- [torch.utils.checkpoint — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/checkpoint.html)
