# SFT、LoRA 与参数高效微调

[返回仓库首页](../README.md) · [来源目录](../SOURCES.md) · [阅读全部题目](../exports/questions.jsonl)

L1 基础 · L2 进阶 · L3 深入。难度为编辑判断；社区线索不等于雇主确认真题。所有答案为独立整理。

## 题目导航

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

**L1 · 社区题目线索** · 标签：SFT / 指令微调 / Loss / LabelShift / PromptMask / 有效Token

**30 秒回答**

预训练与常规 SFT 都可用 next-token 交叉熵，差别主要在数据分布和监督范围。预训练常监督有效文本，SFT 常只监督回答。logit_t 预测下一 token，首回答由末 prompt 位置预测；prompt 不计直接 loss 仍参与上下文 attention。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 设最终序列 x_0,…,x_{T−1}，logits 的第 t 行基于 x_0…x_t，预测目标 x_{t+1}。直接实现可对齐 logits[:,:−1] 与 labels[:,1:]；框架也可在尾部补 ignore 后保持全部 logits。二者有效目标一致，shift 只由一处负责。
- 假设 [P0,P1,A0,A1,EOS]，未 shift 的 labels=[−100,−100,A0,A1,EOS]。shift 后 P1 位置 logits 预测 A0，A0 位置预测 A1，A1 位置预测 EOS；P0→P1 的目标忽略，末 EOS logit 无下一标签。不能额外屏蔽 P1 的 logits 从而丢掉首回答监督。
- 预训练常用全有效文本标签，排除 padding、无后续目标及配方指定的文档边界；SFT 可仅回答、所有 assistant 回合、最后回合或全序列计损，并非模型名称决定。SFT 仍可学习任务内容/知识，不能一概说只学格式。
- loss mask 决定目标贡献，attention mask 决定可访问的条件；回答 loss 的梯度可经 attention 回到 prompt 表示和共享参数，prompt 没有直接 CE 项不等于 prompt 路径没有梯度。所有生成式目标仍需正确 causal 约束和 packing 隔离。
- 用 L=Σ有效目标 NLL/N_valid 表示常规 token 平均；若改为每样本等权或加入辅助损失，目标会不同。比较训练效果应核对模板、监督范围、数据分布与任务评测，不能把不同 mask 下的 loss 数值直接排名。

### 公式

```text
L=−Σ_{t=0}^{T−2} m_{t+1} log softmax(z_t)[x_{t+1}] / Σ_{t=0}^{T−2} m_{t+1}; z_t predicts x_{t+1}
```

### 易错点

- 把 SFT 写成另一种分类 loss，或断言 SFT 永远只能计回答 loss。
- 按预测 logits 所在位置屏蔽 prompt，错误丢掉 prompt→首 answer 的有效预测。

### 面试官可能追问

- 多轮对话只监督最后一轮时，前几轮回答是否仍提供上下文？
- 预训练的全 token loss 和 SFT 的回答 loss 能直接比较吗？

</details>

**技术依据**

- [CORE-S044 · TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [CORE-S050 · LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [CORE-S032 · torch.nn.CrossEntropyLoss — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [CORE-S072 · TRL SFT Trainer — loss, labels and packing](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [CORE-S081 · Transformers v4.57.1 official ForCausalLMLoss](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/loss/loss_utils.py)

**题目出处线索**

- [CORE-S010 · LLM-Interview-Code](https://github.com/ckd0817/LLM-Interview-Code) · `search_snippet`：代码备考仓库摘要列出注意力和 Pretrain/SFT 训练损失；题库按主题整理，不是公司真题证据。

<a id="ft-002"></a>
## FT-002 · SFT 怎样只对回答部分计算 loss？

**L1 · 社区题目线索** · 标签：LossMask / LabelShift / Padding

**30 秒回答**

将 prompt、padding 及不监督的角色位置标签设为 ignore_index，例如 −100，回答 token 保留真实 ID。条件 token 仍参与前向和 attention，mask loss 并非删去输入；还应核对 label shift、模板边界和 EOS。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 先应用模型 chat template，再在最终 token 序列上标记监督目标。多轮对话可选所有 assistant 回合或仅最后回合，需确认 generation 区间包含哪些正文/结束 token；字符切分与 token 切分不能混用。
- 若首 answer 位于 j，保留 labels[j]=x_j，prompt 的 labels[:j]=−100；shift 后由 logits[j−1] 预测首 answer。只屏蔽 prompt 的目标 label，不把 logits[j−1] 一起删掉；若模型内部处理 shift，collator 保持 labels 与 input_ids 同位置。
- 合法 prompt 的 attention_mask 仍为有效，因果注意力允许回答关注先前问题和历史。ignore labels 只去掉该目标直接 CE 项，回答梯度仍能经 prompt 的上下文路径传播；prompt mask 不等于 attention 屏蔽。
- padding 与真实 EOS 即使共享 token ID，也应按实际位置区分；不要按 EOS ID 全部置 −100。EOS 可作为停止生成的有效监督目标，pad query 与无后续目标另行忽略，归一化按真正保留的目标数。
- packing 时在每段边界执行同一监督规则，独立段首 label 不应由前一段末 logit 预测；同时提供块状可见性或后端显式段边界。用小序列逐项打印 logits 索引、目标 ID、mask 与最终 loss 验证。

### 公式

```text
L=−Σ_{t=0}^{T−2} m_{t+1} log p_θ(x_{t+1}|x_≤t)/Σ_{t=0}^{T−2}m_{t+1}
```

### 易错点

- 把 prompt 的 attention_mask 设为 0，或把末 prompt logit 的首回答监督删掉。
- collator 和模型内部各 shift 一次，或把 pad=EOS 当作所有 EOS 均为 padding。

### 面试官可能追问

- 如何验证多轮 assistant-only mask 与 chat template 的 generation 区间？
- 没有任何有效回答目标的样本应怎样过滤或处理？

</details>

**技术依据**

- [CORE-S044 · TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [CORE-S032 · torch.nn.CrossEntropyLoss — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [CORE-S072 · TRL SFT Trainer — loss, labels and packing](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [CORE-S081 · Transformers v4.57.1 official ForCausalLMLoss](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/loss/loss_utils.py)

**题目出处线索**

- [CORE-S003 · AgentGuide：公司面试案例整理](https://github.com/adongwanai/AgentGuide/blob/main/docs/04-interview/12-company-interview-cases.md) · `reported_question`：二次汇编题目列表出现相应提问；原始公司面经未逐条核验。
- [CORE-S008 · 知乎问答：生成语言模型微调与预训练如何计算 loss](https://www.zhihu.com/en/answer/3335594083) · `search_snippet`：知乎搜索摘要讨论预训练与微调 loss/label shift；属于技术问答主题，非公司面经，正文安全验证受阻。

<a id="ft-003"></a>
## FT-003 · chat template 错配会导致哪些问题？

**L2 · 编辑补充题** · 标签：ChatTemplate / EOS / Tokenizer

**30 秒回答**

chat template 把角色与消息变成模型训练过的 token 序列，错配可能导致续写用户、角色混乱或无法停止。训练和部署应统一模板、特殊 token 与 EOS；模板已添加特殊 token 时，再次 tokenization 要避免重复添加。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 保存 tokenizer、chat_template 和 special token 配置，与模型及适配器一起版本化。
- 训练通常不额外插入新 assistant 回复起始提示；生成时按模板需要添加 generation prompt。
- 先渲染文本再编码时检查 add_special_tokens，避免重复 BOS/EOS。
- 用短对话检查序列 ID、角色边界、loss mask 和终止行为，多轮再做回归。

### 易错点

- 只检查字符串外观，忽略底层 token ID。
- 把所有模型都换成同一套 ChatML 而无需训练。

### 面试官可能追问

- 为什么重复 EOS 可能损害生成？
- 模型已有模板时还需要 clone 新模板吗？

</details>

**技术依据**

- [CORE-S045 · Chat templates](https://huggingface.co/docs/transformers/chat_templating)
- [CORE-S044 · TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="ft-004"></a>
## FT-004 · 全参数微调、LoRA、Adapter 和 Prefix-Tuning 怎样选择？

**L1 · 社区题目线索** · 标签：PEFT / FullFT / 选型

**30 秒回答**

全参数微调自由度更高，但需要更多梯度和优化器状态；PEFT 冻结大部分底座，训练少量增量参数，适合资源受限或多任务切换。选择应结合任务差异、数据量、显存和部署方式，不能保证 PEFT 总与全参数效果相同。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- LoRA 对线性层加入低秩更新，可在适合的部署设置中合并回权重。
- Adapter 引入额外模块，任务参数独立，但可能增加推理算子和时延。
- Prefix-Tuning 学习连续前缀，通常给 attention 提供额外键值条件。
- PEFT 仍需要底座前向及反向激活，参数减少比例不等于显存减少比例。

### 易错点

- 训练参数减少 99% 就声称训练显存减少 99%。
- 仅看单次训练成本，忽略多适配器服务和合并成本。

### 面试官可能追问

- 领域迁移很大时为什么全参数值得尝试？
- 多任务适配器如何与 prefix cache 协同？

</details>

**技术依据**

- [CORE-S046 · LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [CORE-S048 · Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- [CORE-S049 · Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)

**题目出处线索**

- [CORE-S001 · 网易大模型应用岗面经（一面、二面）](https://www.nowcoder.com/discuss/909223288612610048?sourceSSR=dynamic) · `reported_question`：公开面经题目列表直接出现这一主题；题目已改写，答案独立整理，公司归属未独立认证。

<a id="ft-005"></a>
## FT-005 · LoRA 的低秩更新公式及可训练参数量是什么？

**L1 · 社区题目线索** · 标签：LoRA / 低秩 / 公式

**30 秒回答**

LoRA 冻结底座矩阵 W，学习 ΔW=(α/r)BA，其中 A、B 经较小秩 r 分解更新。参数量从 d_out×d_in 降为 r(d_in+d_out)，限制的是更新矩阵的秩，不要求原始权重或整个模型是低秩。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- W∈R^{d_out×d_in}，A∈R^{r×d_in}，B∈R^{d_out×r}，rank(BA)≤r。
- 前向为 Wx+(α/r)BAx；梯度经过冻结底座传播至需要训练的模块。
- 标准浮点部署可将增量合入 W，免去额外两次小投影；多适配器切换未必合并。
- 省的是权重梯度及优化器状态，长序列激活仍可主导显存。

### 公式

```text
ΔW=(α/r)BA; N_train=r(d_in+d_out)
```

### 易错点

- 把 LoRA 说成对原模型权重做截断 SVD。
- 漏掉不同目标模块的数量，误报总训练参数量。

### 面试官可能追问

- 如何估算同时对 Q/K/V/O 与 FFN 加 LoRA 的参数量？
- 量化底座合并 LoRA 有什么精度风险？

</details>

**技术依据**

- [CORE-S046 · LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)

**题目出处线索**

- [CORE-S002 · 腾讯Teg大模型暑期算法面经](https://api-cdn.nowcoder.com/feed/main/detail/f91200d9c116401090432a2d78e5f76d?sourceSSR=users) · `reported_question`：公开帖子列出架构、LoRA、量化原理；仅保留问题主题，未采纳社区答案。

<a id="ft-006"></a>
## FT-006 · LoRA 怎样初始化，r、alpha 与 dropout 各控制什么？

**L2 · 社区题目线索** · 标签：LoRA / 初始化 / 梯度 / Dropout / PEFTDefaults / rsLoRA

**30 秒回答**

原始 LoRA 用随机 A、零 B，使初始增量为零；PEFT 普通线性层默认常用 Kaiming-uniform A、零 B，也支持其他初始化。r 控制容量，标准 alpha/r 调节分支尺度，dropout 用于训练正则。应区分论文、库默认和可选变体，再做消融。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 采用 W_eff=W_0+sBA，A∈R^(r×d_in)、B∈R^(d_out×r)。原论文使用随机高斯 A、零 B；本轮读取的 PEFT main 普通 Linear 默认为 Kaiming-uniform A、零 B，Gaussian 选项仍为零 B。Embedding 等层可有不同约定，不能推广为所有模块统一默认。
- 设 G=∂L/∂ΔW，∂L/∂A=sB^T G、∂L/∂B=sGA^T。B=0 时纯分支初始 A 梯度为 0，B 通常可更新；随后 A 可获得梯度。A/B 都为零会让纯 BA 路径无法启动，随机 B+零 A 则是另一种可启动但轨迹不同的方案。
- PiSSA/EVA/LoftQ 等可选方法使用不同权重、激活或量化相关信息，是否改动底座及如何保持初始函数须按方法核验。PEFT init_lora_weights=False 随机初始化 A/B，初始 adapter 一般不是 no-op；不能把它与默认的零增量混同。
- r 决定更新秩上界与 r(d_in+d_out) 参数量；标准 s=alpha/r，rsLoRA 可用 alpha/√r。固定 alpha 改 r 会同时改变容量和缩放，不是纯容量对比；alpha 也不等于学习率，其作用和优化器/初始化共同影响更新。
- 普通 Linear LoRA 的 dropout 施加在低秩分支输入，训练时正则，eval 时关闭，底座分支仍保留。r、alpha、dropout、目标模块与学习率应联合记录，并用验证指标/成本消融；没有对所有模型都最佳的 r 或固定 alpha:r 比值。

### 公式

```text
ΔW=sBA; rank(ΔW)≤r; s=alpha/r（标准）或 alpha/√r（rsLoRA）; ∂L/∂A=sB^T G, ∂L/∂B=sGA^T
```

### 易错点

- 把原论文高斯初始化、PEFT Linear 默认和所有新初始化变体说成完全相同。
- 双零 A/B、默认 no-op 与训练阶段梯度为零混为一谈，或把 alpha 当作优化器学习率。

### 面试官可能追问

- 如何检查添加 adapter 前后初始输出一致，以及第一步 A/B 的梯度？
- 增大 r 时怎样控制缩放和优化超参数以比较容量收益？

</details>

**技术依据**

- [CORE-S046 · LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [CORE-S070 · PEFT LoRA — initialization and LoraConfig](https://huggingface.co/docs/peft/main/en/package_reference/lora)
- [CORE-S071 · PEFT official implementation: LoRA layer](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/layer.py)

**题目出处线索**

- [CORE-S003 · AgentGuide：公司面试案例整理](https://github.com/adongwanai/AgentGuide/blob/main/docs/04-interview/12-company-interview-cases.md) · `reported_question`：二次汇编题目列表出现相应提问；原始公司面经未逐条核验。

<a id="ft-007"></a>
## FT-007 · LoRA 的 rank、alpha、dropout 和 target_modules 应怎么调？

**L2 · 社区题目线索** · 标签：LoRA / 超参数 / Ablation / Dropout / rsLoRA

**30 秒回答**

rank 控制更新容量，alpha/r 控制标准 LoRA 分支尺度，dropout 影响训练正则，target_modules 决定适配位置。应比较验证指标、参数和吞吐；固定 alpha 改 rank 同时改变缩放，不能把收益只归于容量。默认配置也不是通用最佳值。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 参数预算对 r 线性增长，更新秩只满足 ≤r，实际学得秩还取决于数据/优化；扩大 rank 可能收益饱和或过拟合。比较时控制目标模块、数据、有效 token 数与学习率，不要只看训练 loss。
- 标准 LoRA s=alpha/r；PEFT use_rslora=True 使用 alpha/√r。固定 alpha 增大 r 会减小标准分支尺度，固定 alpha/r 又不保证优化轨迹完全相同；可在选定缩放家族内分别消融容量与尺度。
- PEFT 普通 Linear 的训练前向为 base(x)+sB A dropout(x)，dropout 不直接删除底座权重，eval 时关闭。小数据可检验其正则收益，过大概率也可能欠拟合；它与全模型 attention/hidden dropout 不是同一个配置。
- target_modules 决定更新位置。原论文常见 Q/V 实验不代表库和所有模型只支持这两类层；可对比 attention 投影、MLP 与 all-linear，同时核对输出层排除规则、保存 embedding/LM head 及额外 trainable bias 的设置。
- 记录具体 PEFT 版本、init_lora_weights、r、alpha、dropout、目标模块、缩放变体、优化器和验证成本。default r/alpha/dropout 属于库配置，推荐值属于数据相关实验结论，两者不能冒充公司标准答案或通用最佳值。

### 公式

```text
h=W_0x+sB A Dropout(x); trainable≈r(d_in+d_out); s_standard=alpha/r, s_rsLoRA=alpha/√r
```

### 易错点

- 给所有模型套 r=8、alpha=2r 等万能值，或声称 dropout 在推理时继续增强多样性。
- 改变 rank/模块/缩放后仍把消融结果归因于单个因素。

### 面试官可能追问

- 扩展 target_modules 和增大 r，怎样在相同训练参数预算下比较？
- 为什么相同 alpha/r 仍不能保证两个 rank 的有效更新相同？

</details>

**技术依据**

- [CORE-S046 · LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [CORE-S047 · QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)
- [CORE-S070 · PEFT LoRA — initialization and LoraConfig](https://huggingface.co/docs/peft/main/en/package_reference/lora)
- [CORE-S071 · PEFT official implementation: LoRA layer](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/layer.py)

**题目出处线索**

- [CORE-S003 · AgentGuide：公司面试案例整理](https://github.com/adongwanai/AgentGuide/blob/main/docs/04-interview/12-company-interview-cases.md) · `reported_question`：二次汇编题目列表出现相应提问；原始公司面经未逐条核验。

<a id="ft-008"></a>
## FT-008 · LoRA 与 QLoRA 有何区别，NF4、双重量化与分页优化器做什么？

**L2 · 社区题目线索** · 标签：QLoRA / NF4 / 量化训练 / LoRAComparison / StorageVsCompute

**30 秒回答**

LoRA 通过低秩增量适配冻结底座；QLoRA 在此基础上以 4-bit 量化形式存储冻结底座，再用较高精度计算并训练 adapter。它用 NF4、双重量化和分页优化器节省内存，但激活、梯度及工作区仍占显存，也不保证比普通 LoRA 更快或完全同精度。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 普通 LoRA 本身不要求量化，常用 FP16/BF16 底座；QLoRA 原方案冻结 4-bit 底座，计算时按块解量化，再与可训练 LoRA 分支组合。梯度穿过底座计算传到 adapter，但不更新量化底座参数；存储 bit 数与 GEMM compute dtype 必须分开说明。
- NF4 使用适合近似正态权重分布的非均匀量化码本及块 scale，4-bit 值不是一般意义上的线性 INT4。它是有损量化；权重分布、校准/初始化、实现与目标模块都会影响效果，不能称对任意张量无损。
- Double Quantization 进一步量化用于第一层量化的常数，主要压缩 scale 等元数据，不是把同一权重再量化一次来提高精度。原始 QLoRA 的组件定义与库里可选开关/默认值应分开，启用某个 4-bit 加载器不自动表示完整复现论文配方。
- Paged optimizer 借助统一内存/分页管理优化器状态峰值，缓解瞬时显存压力；它不压缩 activation，也可能引入 CPU/GPU 迁移开销。长序列下 attention 激活、解量化工作区、adapter 梯度/状态、batch 和 checkpoint 策略仍可决定是否 OOM。
- 收益重点是底座存储与训练可行性；速度须实测解量化开销、内核、GPU 和序列长度。部署合并时可将解量化底座加上 sBA 后再量化，通常引入新的误差，不能直接把浮点增量加到 4-bit 整数码上；应比较未合并、合并和重新量化的输出。

### 公式

```text
h=Dequantize(Q(W_0))x+sBAx; trainable parameters belong to A/B（及配方显式启用的额外模块），quantized W_0 frozen
```

### 易错点

- QLoRA 训练全部 INT4 权重，或把 4-bit 权重容量当作训练总显存。
- LoRA 与 QLoRA 是互斥算法，或 QLoRA 在所有任务都与浮点 LoRA 等精度且更快。

### 面试官可能追问

- LoRA/QLoRA 的公平比较要统一哪些模块、数据、初始化及评测条件？
- 为什么长上下文训练的显存瓶颈可能从权重转移到激活？

</details>

**技术依据**

- [CORE-S047 · QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)
- [CORE-S046 · LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [CORE-S070 · PEFT LoRA — initialization and LoraConfig](https://huggingface.co/docs/peft/main/en/package_reference/lora)

**题目出处线索**

- [CORE-S006 · 商汤NLP一面](https://www.nowcoder.com/feed/main/detail/fdc049a6b3444f3abee6f22256fc64ac) · `search_snippet`：仅搜索摘要可见 RMSNorm/SwiGLU/QLoRA；正文未读到，不能核验提问完整上下文。

<a id="ft-009"></a>
## FT-009 · Adapter、Prompt-Tuning 与 Prefix-Tuning 有何差别？

**L2 · 编辑补充题** · 标签：Adapter / PromptTuning / PrefixTuning

**30 秒回答**

Adapter 在网络中加入可训练模块；Prompt-Tuning 通常学习输入层的软提示；Prefix-Tuning 则给 attention 提供可训练连续前缀，常影响多层键值。它们都可冻结底座，但加入位置、表达容量和推理开销不同。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- Adapter 常使用降维、非线性、升维及残差结构，额外参数由瓶颈维度决定。
- Prompt-Tuning 的虚拟 token 不必对应可读词；可用于引导模型，但不能等同于人工文本 prompt。
- Prefix-Tuning 的前缀可通过重参数化生成不同层的 K/V，原始实现细节依架构而异。
- 软前缀会占一定序列或 KV 预算；Adapter 的额外模块可能影响内核融合。

### 易错点

- 把所有 soft prompt 方法都叫输入 embedding 调参。
- 说 Prefix-Tuning 完全没有服务时开销。

### 面试官可能追问

- 参数量相同时哪些实验能比较表达能力？
- 离散 prompt 与软提示怎样保存和部署？

</details>

**技术依据**

- [CORE-S048 · Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- [CORE-S049 · Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- [CORE-S069 · PEFT: Prompt tuning](https://huggingface.co/docs/peft/main/en/package_reference/prompt_tuning)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="ft-010"></a>
## FT-010 · SFT 数据量越大越好吗，怎样构建高质量指令集？

**L2 · 编辑补充题** · 标签：数据质量 / SFT / LIMA

**30 秒回答**

SFT 效果取决于回答正确性、任务覆盖与分布匹配，数据越多并非一定越好。可从高质量种子集出发分层补齐真实需求，清洗重复与冲突，独立保留验证集。小数据论文的结果也不能推成所有场景只需少量样本。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 先列任务、难度、语言、格式和拒答边界矩阵，再按真实请求分布采样。
- 对答案可验证任务使用规则或执行器核验，开放回答使用明确标注标准和抽审。
- 按来源、模板或任务组切分，防止近重复跨 train/validation。
- 比较固定 token 预算下的不同质量及配比，记录通用回归和边缘 case。

### 易错点

- LIMA 使用少量优质数据，就断言规模不重要。
- 单纯把相同模板扩写数万条当成丰富覆盖。

### 面试官可能追问

- 如何区分模型底座能力不足与 SFT 覆盖不足？
- 同一 prompt 多个冲突答案怎么处理？

</details>

**技术依据**

- [CORE-S050 · LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [CORE-S044 · TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="ft-011"></a>
## FT-011 · 用模型生成 SFT 数据怎样避免错误与同质化？

**L2 · 编辑补充题** · 标签：SyntheticData / SelfInstruct / 数据验证

**30 秒回答**

合成数据可降低标注成本，但教师模型会把错误、风格偏差及有限任务覆盖传给学生。应先明确能力缺口，再生成多样候选、验证正确性与去重，保留人工或真实数据对照，不能把模型自评高分视作事实正确。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- Self-Instruct 使用种子任务生成指令与回答，再过滤无效及相似样本。
- 代码、数学、结构化任务优先用执行器、单元测试或确定性规则验证结果。
- 多种提示、来源和难度组合有助覆盖，但需统计模板和语义重复。
- 固定一份独立真实验证集，对比真实/合成配比，追踪教师系统性错误。

### 易错点

- 只用同一个模型生成并自评分，认为已独立验证。
- 把生成条数当作有效独立样本数。

### 面试官可能追问

- 教师强于学生时怎样控制任务难度？
- 蒸馏风格和蒸馏能力有什么区别？

</details>

**技术依据**

- [CORE-S051 · Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="ft-012"></a>
## FT-012 · 微调后的灾难性遗忘怎样发现和缓解？

**L2 · 编辑补充题** · 标签：CatastrophicForgetting / ContinualLearning / 回归

**30 秒回答**

灾难性遗忘表现为新任务改善而旧任务能力明显退化。先用固定通用与领域评测量化，再尝试混入旧数据、减小学习率和更新范围、早停或正则约束。PEFT 有助降低部分风险，但冻结底座也不能保证适配后输出不退化。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 分别跟踪新任务、旧任务、指令格式和安全边界，避免把模板错配当成知识遗忘。
- replay 使用代表性旧样本，与新数据共同训练，但配比会影响领域收益。
- 参数重要性正则等持续学习方法提供思路，效果取决于任务和计算成本。
- 保留底座及适配器版本可回滚，不能用可回滚替代上线前回归。

### 易错点

- LoRA 不修改底座，因此部署输出永远不可能遗忘。
- 只用新的 validation set 做早停。

### 面试官可能追问

- 混入通用数据为何可能削弱领域适配？
- 怎样区分真实遗忘与生成参数变化？

</details>

**技术依据**

- [CORE-S052 · Continual Learning Through Synaptic Intelligence](https://arxiv.org/abs/1703.04200)
- [CORE-S040 · Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)
- [CORE-S046 · LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="ft-013"></a>
## FT-013 · 梯度累积等价于大 batch 吗？变长样本怎么归一化？

**L2 · 编辑补充题** · 标签：GradientAccumulation / TokenNormalization / DDP

**30 秒回答**

梯度累积让多个 micro-batch 的梯度汇总后更新，减少单次激活显存。正确归一化时可近似大 batch，但 dropout、数值顺序等会有差异；变长样本不能简单等权平均各 micro-batch 的 token 平均 loss。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 有效 batch 约为 micro-batch×累积次数×数据并行数，最后不满的累积窗口要单独处理。
- 对 token 平均目标，应累加各窗口有效 token 的 NLL，再按整个窗口有效 token 总数缩放。
- DDP 的梯度平均及框架自动缩放会改变实现系数，应核对而非重复除以世界大小。
- optimizer.step 与 scheduler.step 按真实参数更新推进，可用 no_sync 减少累积期间通信。

### 公式

```text
B_effective=B_micro·K_accum·N_DP；token loss 用整个累积窗口的有效 token 数归一化
```

### 易错点

- 每个 batch loss 除累积次数，就认为变长序列始终等价。
- 累积时忘记清零梯度或每个 micro-step 都更新 scheduler。

### 面试官可能追问

- 如何验证累积与一次大 batch 的梯度接近？
- 为什么梯度累积降低显存却可能降低吞吐？

</details>

**技术依据**

- [CORE-S053 · Performing gradient accumulation with Accelerate](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="ft-014"></a>
## FT-014 · gradient checkpointing 节省什么，为何会变慢？

**L2 · 编辑补充题** · 标签：ActivationCheckpointing / 显存 / 反向传播

**30 秒回答**

gradient checkpointing 少保存中间激活，在反向时重算前向片段，以计算换显存。它不减少底座参数或优化器状态，主要用于训练反向。实现需保证重算与原前向一致，随机数状态、分支或状态修改可能影响梯度正确性。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 保留若干边界激活，其余在 backward 需要时重新计算，节省量取决于切分粒度。
- 冻结底座的 LoRA 训练仍需传播到适配器，长序列场景可受益。
- 状态化函数、随机 dropout、设备迁移要遵守框架约束，不能假设重算天然等价。
- 查看版本文档中的 reentrant/non-reentrant 行为，基准比较峰值显存和有效 token/s。

### 易错点

- 把 gradient checkpointing 当作保存 checkpoint 文件。
- 无反向的常规推理用它来节省 KV cache。

### 面试官可能追问

- 为什么 LoRA 的输入梯度配置会影响 checkpointing？
- 如何选择重算粒度与激活卸载？

</details>

**技术依据**

- [CORE-S054 · torch.utils.checkpoint — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/checkpoint.html)
- [CORE-S047 · QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="ft-015"></a>
## FT-015 · SFT loss 下降但任务效果变差，应该怎样排查？

**L2 · 编辑补充题** · 标签：Debugging / Overfit / SFT

**30 秒回答**

先排除数据与实现问题，再判断过拟合或分布错配：核对模板、标签 mask、有效样本、EOS 和验证泄漏，固定生成配置比较任务指标。loss 下降只说明模型更贴合训练目标，不能直接证明事实性或真实任务收益。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 抽取完整样本打印最终 token、监督区间、截断与终止标记，确认模型实际学到的是回答。
- 比较 train/validation loss 及真实任务指标，按长度、领域和格式分析 bad case。
- 检查高重复、错误答案、教师风格偏差和训练/部署 prompt 差异。
- 再考虑减小学习率、早停、降 rank、正则或改数据配比，并保留单变量消融。

### 易错点

- 一看到效果差就继续加 epoch。
- 只观察总体 loss，不看大量被 mask 或截断的样本。

### 面试官可能追问

- 训练 loss 接近 0 是否一定是好事？
- 如何区分过拟合与生成参数造成的差异？

</details>

**技术依据**

- [CORE-S044 · TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [CORE-S045 · Chat templates](https://huggingface.co/docs/transformers/chat_templating)
- [CORE-S050 · LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。
