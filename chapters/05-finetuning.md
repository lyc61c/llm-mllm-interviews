# SFT、PEFT、蒸馏与模型编辑

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 目录

- [指令数据与训练目标](#topic-1)
  - [FT-001 · 预训练与 SFT 的 loss 有何差异，label shift 如何对齐？](#ft-001)
  - [FT-002 · SFT 怎样只对回答部分计算 loss？](#ft-002)
  - [FT-003 · chat template 错配会导致哪些问题？](#ft-003)
  - [FT-010 · SFT 数据量越大越好吗，怎样构建高质量指令集？](#ft-010)
  - [FT-011 · 用模型生成 SFT 数据怎样避免错误与同质化？](#ft-011)
  - [FT-016 · 领域 SFT 应从 Base 还是 Instruct/Chat 模型开始？](#ft-016)
  - [PRE-010 · SFT 数据 packing 怎样提高效率并保持跨样本隔离？](#pre-010)
- [LoRA 与其他 PEFT](#topic-2)
  - [FT-004 · 全参数微调、LoRA、Adapter 和 Prefix-Tuning 怎样选择？](#ft-004)
  - [FT-005 · LoRA 的低秩更新公式及可训练参数量是什么？](#ft-005)
  - [FT-006 · LoRA 怎样初始化，r、alpha 与 dropout 各控制什么？](#ft-006)
  - [FT-007 · LoRA 的 rank、alpha、dropout 和 target_modules 应怎么调？](#ft-007)
  - [FT-008 · LoRA 与 QLoRA 有何区别，NF4、双重量化与分页优化器做什么？](#ft-008)
  - [FT-009 · Adapter、Prompt-Tuning 与 Prefix-Tuning 有何差别？](#ft-009)
  - [FT-017 · BitFit 为什么只调 bias，适用条件和局限是什么？](#ft-017)
  - [FT-018 · P-Tuning v1、P-Tuning v2 与 Prompt/Prefix Tuning 怎样区分？](#ft-018)
  - [FT-019 · AdaLoRA 如何自适应分配低秩预算，与固定 rank LoRA 有何区别？](#ft-019)
  - [FT-020 · AdapterFusion 和 AdapterDrop 如何组合知识或降低 adapter 开销？](#ft-020)
  - [FT-021 · MAM Adapter、UniPELT 等组合 PEFT 方法为什么要混合不同模块？](#ft-021)
  - [FT-023 · Prompt learning 中 template、verbalizer 与连续提示分别是什么？](#ft-023)
- [训练技巧与排错](#topic-3)
  - [FT-012 · 微调后的灾难性遗忘怎样发现和缓解？](#ft-012)
  - [FT-013 · 梯度累积等价于大 batch 吗？变长样本怎么归一化？](#ft-013)
  - [FT-014 · gradient checkpointing 节省什么，为何会变慢？](#ft-014)
  - [FT-015 · SFT loss 下降但任务效果变差，应该怎样排查？](#ft-015)
- [知识蒸馏与模型编辑](#topic-4)
  - [FT-022 · 知识蒸馏中的 logits、隐藏层和生成答案监督各有什么作用？](#ft-022)
  - [FT-024 · 模型编辑与继续训练、RAG 有何区别？ROME、MEMIT 与 MEND 如何修改知识？](#ft-024)
  - [FT-025 · OPD 的原理和优化目标是什么，与 SFT、RL 怎样选择？](#ft-025)
  - [FT-026 · 拿不到教师 logits 时还能做 OPD 吗，只有文本反馈有哪些限制？](#ft-026)
  - [FT-027 · 教师与学生词表不同，跨 tokenizer 的 OPD 怎样定义对齐和损失？](#ft-027)

<a id="topic-1"></a>
## 指令数据与训练目标

<a id="ft-001"></a>
### FT-001 · 预训练与 SFT 的 loss 有何差异，label shift 如何对齐？

**L1** · 小红书

#### 答案

预训练和常规监督微调 SFT 都可以用下一个 token 的交叉熵，主要区别是模型读什么数据、哪些位置需要算损失。预训练通常学习有效文本的各个位置，SFT 用指令和答案教任务行为，常只监督回答，也可以监督所有助手回合或全序列，这要按训练目标确定。

标签移动是为了让当前位置预测下一个位置。公式中 $`z_t`$ 是第 $`t`$ 个位置的 logits，也就是各候选 token 的未归一化分数，$`m_{t+1}`$ 表示下一个目标是否参与损失。例如输入是 `[P0,P1,A0,A1,EOS]`，P 表示提示词，A 表示答案，EOS 是结束符。标签仍与输入同位置存放为 `[-100,-100,A0,A1,EOS]`；内部 shift 后，P1 的输出预测 A0，A0 预测 A1，A1 预测 EOS。这里 -100 是忽略标签的常用约定，末 EOS 没有后续目标。

所以不能因为 P1 属于提示词，就把它预测首个答案的那一项也删掉。数据处理和模型内部只能有一处负责 shift，否则会错位两次。损失 mask 决定哪些目标计分，注意力 mask 决定能读哪些历史，两者用途不同：提示词标签不计损，模型仍要读取它，回答梯度也可以通过其表示传播。有效损失通常按参与监督的 token 总数平均；不同监督范围下的训练 loss 不能直接比较高低，任务效果还要单独验证。

```math
\begin{aligned}z_t&\longrightarrow x_{t+1}\\\mathcal L&=-\frac{\sum_{t=0}^{T-2}m_{t+1}\log\mathrm{softmax}(z_t)_{x_{t+1}}}{\sum_{t=0}^{T-2}m_{t+1}}\end{aligned}
```

#### 易错点

- 常规 SFT 可以沿用交叉熵，监督范围也不一定永远只含最后一轮回答。
- 屏蔽提示目标时，不能删掉末提示位置对首回答的预测。

#### 追问

- 多轮对话只监督最后一轮时，前几轮回答是否仍提供上下文？
- 预训练的全 token loss 和 SFT 的回答 loss 能直接比较吗？

<a id="ft-002"></a>
### FT-002 · SFT 怎样只对回答部分计算 loss？

**L1**

#### 答案

只对回答计算 SFT 损失，就是保留回答 token 的真实标签，把提示词和填充位置的标签设成忽略值，同时让模型仍然能读到提示词。通常用 -100 作为交叉熵的 ignore_index；它表示这个目标不计分，并不是把对应输入删掉。

应先按模型的聊天模板渲染并分词，再确定每轮助手回答的 token 区间。假设首个回答 token 在位置 $`j`$，那么 `labels[j]` 必须保留，而前面的提示词标签可以忽略。经过正常 shift，是位置 $`j-1`$ 的输出预测这个首回答。公式中的 $`m_{t+1}`$ 就是目标是否监督的标记，分母是有效目标数。多轮对话可以监督全部助手回合，也可以只监督最后一轮，但前几轮通常仍提供上下文。

细节上，要明确角色结束符和 EOS 是否作为停止目标。若填充符恰好与 EOS 共用 token ID，要按位置区分，不能把全部 EOS 标签都忽略，否则模型可能学不好停止。打包独立样本时还要屏蔽跨段目标，并隔离注意力。最后最好手工检查一条很短的样本，逐项打印输入、标签、预测位置和有效数；一个没有任何有效回答目标的样本应被过滤或明确处理，不能悄悄参与平均。

```math
\mathcal L=-\frac{\sum_{t=0}^{T-2}m_{t+1}\log p_\theta(x_{t+1}\mid x_{\le t})}{\sum_{t=0}^{T-2}m_{t+1}}
```

#### 易错点

- 把提示词的注意力置零，会使答案失去需要的条件。
- 重复 shift，或因 pad 与 EOS 共用 ID 而忽略所有真实结束符。

#### 追问

- 如何验证多轮 assistant-only mask 与 chat template 的 generation 区间？
- 没有任何有效回答目标的样本应怎样过滤或处理？

<a id="ft-003"></a>
### FT-003 · chat template 错配会导致哪些问题？

**L2**

#### 答案

Chat template 是把“谁说了什么”变成模型熟悉的 token 格式，模板错配相当于换了对话协议，模型可能分不清角色、接着写用户的话，或不知道该什么时候停。模板里的角色标记、开始结束符和空白形式，都是模型训练分布的一部分。

例如某个模型用专门的助手起始标记表示“接下来该回答”，部署时如果漏掉它，模型可能只继续上一段文本；如果重复添加结束符，也可能过早结束。训练样本通常已经包含答案，不应额外追加一个用于新回答的 generation prompt；推理时是否追加，要按该模板的约定处理。若先渲染字符串再分词，还要检查分词器是否又自动添加 BOS 或 EOS，避免重复特殊 token。

排查时不能只看字符串是否漂亮，应核对真实 token ID、角色边界、回答 loss mask 和停止行为。先验证单轮，再覆盖多轮、工具消息和空回答。分词器、模板、特殊 token 配置应与底座和适配器一起保存、版本化，训练和部署保持一致。把所有模型都统一成某种常见对话格式可能方便工程，但除非经过对应训练或验证，不能假设原模型会自然理解新协议。

#### 易错点

- 只检查字符串外观，忽略底层 token ID。
- 把所有模型都换成同一套 ChatML 而无需训练。

#### 追问

- 为什么重复 EOS 可能损害生成？
- 模型已有模板时还需要 clone 新模板吗？

<a id="ft-010"></a>
### FT-010 · SFT 数据量越大越好吗，怎样构建高质量指令集？

**L2** · 字节跳动 / 阿里巴巴 / 深势科技

#### 答案

SFT 数据不是越多越好，关键是正确答案覆盖了需要的任务和真实用户分布，重复或错误样本再多也可能降低效果。高质量指令集要同时考虑内容正确、指令清楚、输出格式一致和任务多样，而不是只追求样本总数。

我会先列出任务、语言、难度、格式和拒答边界，再用优质种子补齐覆盖。例如一个抽取系统不仅要有“信息齐全”的例子，也需要缺字段、无结果、歧义输入和格式错误的例子。代码可以执行测试，数学可以核验答案，开放问答则用明确标准并抽审。对相同问题的冲突答案，先确定是否只是合理的多解，还是事实或格式标准冲突，再决定保留和标注方式。

切分数据时按来源、模板或任务组隔离，防止近重复让验证分数虚高；实验最好固定训练 token 预算，对比质量和配比，而不只比较条数。少量优质数据有效，说明底座已有能力时高质量示范很重要，不代表所有任务都只需要少量样本。如果底座缺少领域能力或任务复杂，仍可能需要更广的数据覆盖。最后要测真实请求和通用回归，区分底座能力不足、数据没覆盖和训练格式实现错误。

#### 易错点

- LIMA 使用少量优质数据，就断言规模不重要。
- 单纯把相同模板扩写数万条当成丰富覆盖。

#### 追问

- 如何区分模型底座能力不足与 SFT 覆盖不足？
- 同一 prompt 多个冲突答案怎么处理？

<a id="ft-011"></a>
### FT-011 · 用模型生成 SFT 数据怎样避免错误与同质化？

**L2**

#### 答案

模型合成 SFT 数据能降低标注成本，但教师给出的答案仍可能错误、风格单一或覆盖不足，必须把“生成”与“验证”分开。首先确定学生缺什么，再生成不同类型和难度的候选，过滤错误、重复和不符合任务的样本，而不是让教师随意扩写大量相似问答。

例如做代码任务，可以让模型生成题目和解答，再用执行器检查语法、测试用例和边界条件；做开放问答则需要依据核对或人工抽审。Self-Instruct 的思路是从种子任务扩展指令和回答，再去掉无效和相似内容。使用多种提示、来源和难度能增加覆盖，但仍要统计模板重复和语义重复，因为换几个人名不等于增加独立知识。

教师自评高分不能当作独立正确性证据。可以保留真实数据对照，在独立真实验证集上比较不同真实与合成配比，并分析教师反复犯的系统性错误。题目太难，学生可能学到大量不可靠解释；太容易，又只模仿表达。蒸馏也可能传递教师的措辞和拒答偏差，而不是真实能力，因此验收要看任务正确率、泛化和多样性，不只看合成数据上的训练损失。

#### 易错点

- 只用同一个模型生成并自评分，认为已独立验证。
- 把生成条数当作有效独立样本数。

#### 追问

- 教师强于学生时怎样控制任务难度？
- 蒸馏风格和蒸馏能力有什么区别？

<a id="ft-016"></a>
### FT-016 · 领域 SFT 应从 Base 还是 Instruct/Chat 模型开始？

**L2**

#### 答案

领域 SFT 从 Base 还是 Instruct 模型开始，取决于已有行为是否接近目标，以及手头数据能否教会完整的交互协议。Base 主要通过语言建模训练，便于自行塑造任务行为；Instruct 或 Chat 已有指令和对话习惯，少数据时往往更容易适配，但也可能带有不符合业务的格式或偏好。

例如已有 Chat 模型能稳定对话，只需要学一个领域的结构化输出，可以沿用其模板做小规模微调。若希望严格重定义输出协议，且有充分高质量示范，Base 也值得比较。实验要固定目标评测、数据、训练成本和生成设置，还要检查语言能力、许可证、上下文窗口和特殊 token。Chat 的消息格式、助手监督范围和结束规则应沿用或有意识地重新训练；Base 则需要在训练与部署中一致定义它们。

领域知识不足和接口行为不合格是不同问题。原始领域文本较多时，可以考虑继续预训练；明确问答或抽取行为则用 SFT，两者也都可能改变知识。快速变化、需要证据或权限约束的信息，还可以通过检索提供。最终要比较领域收益、通用回归和实际成本，不能认定 Chat 总优于 Base，也不能认为 SFT 只改变说话方式而绝不会学习或损坏内容能力。

#### 易错点

- 把Base/Chat选择当作无需任务验证的固定规则。
- 给Chat模型更换模板却未同步训练和推理。

#### 追问

- 数据很少时怎样比较PEFT与直接prompt的收益？
- 如何防止领域SFT损坏原先工具调用格式？

<a id="pre-010"></a>
### PRE-010 · SFT 数据 packing 怎样提高效率并保持跨样本隔离？

**L2**

#### 答案

Packing 是把多个短训练样本装进一条长序列，减少填充位置，让计算更多地花在真实内容上。比如最大长度是 2048，而多数对话只有两三百 token，逐条补齐会浪费很多计算；把短对话合理装箱通常能提高吞吐。但独立对话既不能读到彼此内容，也不能跨边界产生预测目标。

处理时先为每条对话应用聊天模板、分词并标好回答标签，再一起打包输入、标签、样本编号和边界。公式里 $`s_i`$ 是位置 $`i`$ 所属的样本，$`v_j`$ 表示位置 $`j`$ 是否有效。位置 $`i`$ 能读取 $`j`$，必须同时满足同一样本、$`j\le i`$ 的因果条件，以及不是填充。可以用块对角因果 mask，或把段边界交给支持变长注意力的后端。结束符和重置位置编号都不会自动建立这种隔离。

还要处理标签移动：普通整行 shift 会让第一段末尾的预测去监督第二段开头，这是不合法的跨样本目标。可以把后一段的首标签忽略，或逐段完成 shift；真实的结束符是否计损则单独定义。回答的首 token 仍应由本段提示词最后一个位置预测。公式里的 $`m_t`$ 选择需要监督的 token，$`\mathcal H_t`$ 是同一样本的历史，损失按有效目标平均。

验证时可以把另一个打包样本替换掉，检查当前样本的 logits 是否不变，再比较未打包与打包的有效损失和梯度。长对话若被拆开，还要明确是否继承历史上下文，并统计丢弃的回答内容。连续原始文本有时有意允许跨文档上下文，那是另一种目标，不能和独立 SFT 的隔离要求混用。

```math
\begin{aligned}\mathrm{allowed}(i,j)&=\mathbf1[s_i=s_j]\,\mathbf1[j\le i]\,\mathbf1[v_j=1]\\\mathcal L&=-\frac{\sum_t m_t\log p_\theta(x_t\mid x_{\mathcal H_t})}{\sum_t m_t}\\\mathcal H_t&=\{j:j\lt t,\ s_j=s_t,\ v_j=1\}\end{aligned}
```

![普通因果 mask 与样本隔离 packing mask](../assets/causal-packing.svg)

两段样本各有 3 个 token；隔离 attention 时，还要处理 shift 后的跨样本标签。

#### 易错点

- EOS、回答 loss mask 或位置编号归零，都不能单独隔离注意力。
- 注意力隔离后仍要排除跨段的预测标签，也不能重复 shift。

#### 追问

- 如何处理长对话拆分时需要继承的历史 prompt？
- 为何 packing 开启后吞吐提高却可能改变梯度加权？

<a id="topic-2"></a>
## LoRA 与其他 PEFT

<a id="ft-004"></a>
### FT-004 · 全参数微调、LoRA、Adapter 和 Prefix-Tuning 怎样选择？

**L1** · 深势科技

#### 答案

选择全参数微调还是参数高效微调，要看任务需要改动多大、数据够不够，以及训练和部署能承担什么成本。全参数微调允许所有权重改变，适应能力更充分，但权重梯度和优化器状态很大；LoRA、Adapter 和 Prefix-Tuning 冻结大部分底座，只训练少量额外参数，更适合资源受限或需要保存多个任务版本的场景。

LoRA 在线性层旁边加低秩更新，浮点部署时通常可以合并回原权重；Adapter 在网络中插入小模块，常用降维、非线性、升维的结构，任务切换方便，但会增加算子；Prefix-Tuning 学习连续前缀，常把额外的键和值加入注意力，为生成提供任务条件，也会占用注意力和缓存预算。比如多个业务共用一个底座，分别保存小适配器，存储可能比保存多个完整模型划算。

我会先用相同数据做小规模基线，比较目标任务、通用回归、峰值显存和真实服务延迟。领域差异很大、数据充分时，全参数值得尝试；数据少时，更多自由度也可能更容易过拟合。冻结底座并不意味着跳过它的计算：梯度为了到达前面的可训练模块，仍可能经过冻结层并需要保存或重算激活。因此可训练参数减少 99%，不能推导训练显存也减少 99%，也不能保证参数高效方法总能达到全参数的效果。

#### 易错点

- 可训练参数减少比例不等于训练显存减少比例。
- 只看训练文件大小，忽略多适配器服务、额外算子和合并成本。

#### 追问

- 领域迁移很大时为什么全参数值得尝试？
- 多任务适配器如何与 prefix cache 协同？

<a id="ft-005"></a>
### FT-005 · LoRA 的低秩更新公式及可训练参数量是什么？

**L1** · 腾讯 / 百度

#### 答案

LoRA 是把需要学习的大矩阵更新拆成两个小矩阵相乘，用较少参数控制模型变化。它冻结原权重 $`W_0`$，学习 $`\Delta W=(\alpha/r)BA`$；这里输入维度是 $`d_{\rm in}`$，输出维度是 $`d_{\rm out}`$，$`A`$ 把输入压到 $`r`$ 维，$`B`$ 再升到输出维度，$`\alpha/r`$ 控制分支尺度。

一个投影的可训练参数是 $`r(d_{\rm in}+d_{\rm out})`$。例如原矩阵是 $`4096\times4096`$，约 1678 万参数，rank 取 8 时两个小矩阵一共 65536 参数。多个目标层要逐层累加，额外训练的偏置或输出头也要另算。低秩约束作用于增量，不作用于整个底座：$`BA`$ 的秩至多为 $`r`$，但 $`W_0+BA`$ 仍可能满秩。一个 $`3\times3`$ 矩阵究竟几秩，要看数值和独立行列，不能只凭形状判断；零 B 初始化时增量的初始秩是零。

前向仍计算 $`W_0x`$ 和低秩分支，冻结只是不给 $`W_0`$ 算权重更新，梯度仍能通过它的输入路径到达其他适配器。LoRA 主要节省可训练权重的梯度和优化器状态，长序列激活可能仍是瓶颈。浮点模型可以把增量合并到权重，减少额外分支的推理开销，但底座大小不会因此变成适配器大小。

合并前要确认底座版本、目标层、缩放和 dtype 一致，并关闭 dropout。低精度舍入可能使合并前后结果不是逐位相同，应检查 logits 和任务效果。量化权重不能直接加浮点增量到整数编码，可能需要反量化、合并后再量化。多任务共享底座时，保留独立适配器更方便切换和回滚，是否合并取决于服务方式。

```math
\begin{aligned}\Delta W&=\frac{\alpha}{r}BA\\N_{\rm train}&=r(d_{\rm in}+d_{\rm out})\\\mathrm{rank}(\Delta W)&\le r\end{aligned}
```

![LoRA 冻结主权重与低秩增量分支](../assets/lora.svg)

常用初始化使增量起始为零；图中采用列向量约定。

#### 易错点

- LoRA 约束增量的秩，不是把底座做截断奇异值分解。
- 算总参数时漏掉目标模块数量、额外偏置或输出头。
- 只有合并或相应融合实现，才可减少额外分支开销；未合并不天然零延迟。

#### 追问

- 如何估算同时对 Q/K/V/O 与 FFN 加 LoRA 的参数量？
- 量化底座合并 LoRA 有什么精度风险？
- 多租户共享一个底座时，为什么离线合并每个 adapter 可能增加总权重内存？
- 3×3的底座权重、初始增量和训练后增量，秩分别有哪些上界？

<a id="ft-006"></a>
### FT-006 · LoRA 怎样初始化，r、alpha 与 dropout 各控制什么？

**L2** · 腾讯 / 快手 / 百度

#### 答案

LoRA 的常见初始化让低秩分支一开始输出零，这样刚插入适配器时模型行为保持原样，再从训练中逐渐学会更新。最初的 LoRA 方案采用随机高斯 A、零 B；PEFT 普通 Linear 的默认实现采用 Kaiming 均匀 A、零 B，不能把论文和所有库模块的初始化说成完全相同。

从 $`\Delta W=sBA`$ 看梯度，$`s`$ 是缩放，$`G`$ 是损失对增量矩阵的梯度，则 A 的梯度为 $`sB^\top G`$，B 的梯度为 $`sGA^\top`$。初始化 B 为零时，来自这条低秩路径的 A 梯度起初为零，B 通常先更新，然后 A 才获得梯度。若 A 和 B 同时为零，两边梯度都为零，单靠这个乘积路径就无法启动。反过来零 A、随机 B 可以启动，但训练轨迹不同。

Rank $`r`$ 控制参数量和增量秩上界，alpha 控制分支缩放，普通 LoRA 使用 $`s=\alpha/r`$，rsLoRA 使用 $`\alpha/\sqrt r`$。Alpha 不是优化器学习率；固定 alpha 增大 rank，会同时改变容量和尺度。普通线性 LoRA 的 dropout 施加在分支输入上，训练时正则，推理时关闭，底座分支仍保留。

默认初始化中的 `kaiming_uniform_(A, a=sqrt(5))` 会得到约 $`1/\sqrt{d_{\rm in}}`$ 的均匀边界，是沿用普通 Linear 的约定，并不等于面向 ReLU 的方差 $`2/d_{\rm in}`$ 配方。其他权重或激活驱动的初始化可能改变底座或初始函数，Embedding 也可能采用不同约定。实践要核对具体模块和版本，并检查初始输出及首步梯度，不能仅凭方法名称推断。

```math
\begin{aligned}\Delta W&=sBA,\quad\mathrm{rank}(\Delta W)\le r\\s_{\rm standard}&=\frac{\alpha}{r},\quad s_{\rm rsLoRA}=\frac{\alpha}{\sqrt r}\\G&=\frac{\partial L}{\partial\Delta W}\\\frac{\partial L}{\partial A}&=sB^\top G,\quad\frac{\partial L}{\partial B}=sGA^\top\end{aligned}
```

#### 易错点

- 最初 LoRA 与 PEFT 默认、不同模块及新初始化变体，不能混作一个配方。
- 双零初始化与单零初始化不同，alpha 也不是优化器学习率。

#### 追问

- 如何检查添加 adapter 前后初始输出一致，以及第一步 A/B 的梯度？
- 增大 r 时怎样控制缩放和优化超参数以比较容量收益？

<a id="ft-007"></a>
### FT-007 · LoRA 的 rank、alpha、dropout 和 target_modules 应怎么调？

**L2** · 快手 / 百度 / 深势科技

#### 答案

LoRA 调参是在有限预算内决定“改哪些层、给多少容量、改动多强、加多少正则”，几个参数需要一起看。Rank 是增量的秩上界，target_modules 决定适配位置，alpha 决定缩放，dropout 用于训练正则；没有适合所有任务的固定 rank 或 alpha 比例。

可以先确定目标层，再扫描少量容量。例如只适配注意力的查询和值投影，和同时适配注意力及前馈层，是两种容量分布。固定参数预算时，扩大模块覆盖与提高每层 rank 可以做对照。公式中的 $`h`$ 是层输出，$`x`$ 是输入，底座分支为 $`W_0x`$，低秩分支为 $`sBA\mathrm{Dropout}(x)`$。普通缩放 $`s=\alpha/r`$ 与 rsLoRA 的 $`\alpha/\sqrt r`$ 不同；固定 alpha 改 rank 后，结果不能只归因于容量。即使固定 alpha/r，两个 rank 的优化轨迹也不必相同。

小数据可以试验 dropout 是否缓解过拟合，但过大也会欠拟合，推理时应关闭。还要核对输出头、embedding 和偏置是否额外训练，打印实际可训练参数，而不是只相信模块名称。比较时固定数据、有效 token、学习率、初始化和缩放家族，同时看任务效果、显存和吞吐。

LoRA 不会省掉底座前向，为把梯度传到更早的分支，也通常仍要计算冻结层的输入梯度。额外的小矩阵乘法、内核调度、量化解码和小 batch 利用率都影响速度，所以训练参数少几十倍，不代表训练快几十倍。应实测相同硬件和长度下的有效 token/s、更新耗时与峰值显存，并区分训练速度和合并后的推理延迟。

```math
\begin{aligned}h&=W_0x+sBA\mathrm{Dropout}(x)\\N_{\rm train}&\approx r(d_{\rm in}+d_{\rm out})\\s_{\rm standard}&=\alpha/r,\quad s_{\rm rsLoRA}=\alpha/\sqrt r\end{aligned}
```

#### 易错点

- 没有适用所有任务的 rank、alpha 比例或 dropout，推理时普通 dropout 应关闭。
- 一次改变 rank、模块和缩放后，不能把结果只归因于 rank。

#### 追问

- 扩展 `target_modules` 和增大 r，怎样在相同训练参数预算下比较？
- 为什么相同 $`\alpha/r`$ 仍不能保证两个 rank 的有效更新相同？

<a id="ft-008"></a>
### FT-008 · LoRA 与 QLoRA 有何区别，NF4、双重量化与分页优化器做什么？

**L2** · 字节跳动

#### 答案

QLoRA 是在 LoRA 的基础上，把冻结底座压成 4 位存储，再用较高精度计算来训练低秩适配器，主要目的是让大模型微调更容易装进显存。普通 LoRA 不要求量化，底座可以是 BF16 或 FP16；QLoRA 仍然训练 LoRA 分支，而不是直接更新全部 4 位权重。

公式中的 $`Q(W_0)`$ 是量化编码，Dequantize 表示按块恢复到计算所需的精度，再与 $`sBAx`$ 相加。NF4 使用适合近似正态权重的非均匀码本，属于有损量化，与均匀 INT4 不同。双重量化进一步压缩第一层量化所需的 scale 等常数，不是把同一个权重再量化一次以提高精度。分页优化器利用统一内存等机制缓解优化器状态的显存峰值，可能发生 CPU 与 GPU 迁移，它不负责压缩激活。

举例来说，底座有 $`P`$ 个参数，16 位权重理想存储约 $`2P`$ 字节，4 位编码约 $`P/2`$ 字节，但还要加 scale、未量化层和工作区。适配器参数量 $`P_a`$ 在两种方法中可以相同；若其训练状态采用低精度参数和梯度、FP32 主权重及两个 Adam 状态，粗估约为 $`16P_a`$ 字节，实际应按实现逐项核对。长上下文下激活很大，冻结底座并不会删除适配器反传需要的这些内容。

所以“4 位存储加 BF16 计算”不等于所有运算都是 4 位，也不意味着必然更快或与浮点 LoRA 完全等精度。公平比较要统一数据、模块和训练设置，并测质量和吞吐。量化底座合并适配器时，需确认后端支持，必要时反量化合并并重新量化，再检查新增误差，不能把浮点增量直接加到整数码上。

```math
h=\mathrm{Dequantize}(Q(W_0))x+sBAx
```

#### 易错点

- 4 位底座存储不等于全部算子是 4 位，也不等于训练所有量化权重。
- QLoRA 结合 LoRA 与量化，速度和精度均需实测，4 位权重账单不是总显存。

#### 追问

- LoRA/QLoRA 的公平比较要统一哪些模块、数据、初始化及评测条件？
- 为什么长上下文训练的显存瓶颈可能从权重转移到激活？
- 长序列时，为什么 QLoRA 相对浮点 LoRA 的节省比例可能变小？

<a id="ft-009"></a>
### FT-009 · Adapter、Prompt-Tuning 与 Prefix-Tuning 有何差别？

**L2** · 腾讯

#### 答案

Adapter、Prompt-Tuning 和 Prefix-Tuning 都是用少量参数适配任务，但它们把可训练部分放在不同位置。Adapter 改模型内部的隐藏表示，Prompt-Tuning 在输入端加入可学习的软提示，Prefix-Tuning 则向注意力提供可学习的前缀条件，常体现为多层额外的键和值。

Adapter 常把隐藏维度先压到较小的瓶颈，经过非线性再升回来，并与原表示相加。例如隐藏维度 4096、瓶颈 64，就用两个较小投影学习任务变化。Prompt-Tuning 学习一串向量，像是在输入前增加虚拟 token，但这些向量不一定能解码成可读词。Prefix-Tuning 的前缀可直接或通过重参数化网络产生各层 K/V，因此不能把所有连续提示都当成只调整输入 embedding。

它们通常能冻结底座，但仍要经过底座计算并反传到可训练部分。软提示会占用输入或缓存预算，前缀增加注意力读取范围，Adapter 则增加算子，可能影响融合和时延。选型要在相近参数和训练预算下比较任务质量，并看真实部署开销；仅比较保存文件大小不够。手写离散提示不训练参数，与这些需要优化可学习向量的方法也要区分。

#### 易错点

- 把所有 soft prompt 方法都叫输入 embedding 调参。
- 说 Prefix-Tuning 完全没有服务时开销。

#### 追问

- 参数量相同时哪些实验能比较表达能力？
- 离散 prompt 与软提示怎样保存和部署？

<a id="ft-017"></a>
### FT-017 · BitFit 为什么只调 bias，适用条件和局限是什么？

**L2**

#### 答案

BitFit 是冻结主要权重，只训练偏置项，让模型用极少参数调整已有特征的偏移和激活阈值。在线性层 $`Wx+b`$ 里，$`W`$ 决定输入如何组合，$`b`$ 改变输出偏移；BitFit 主要调整 $`b`$，通常还需要训练任务输出头，并把这部分计入参数预算。

例如一个预训练特征在新任务里需要更容易被激活，改变偏置就可能改善分类，不必重学整个投影。公式中的 $`W_0`$ 是冻结底座，$`b`$ 是可训练偏置，$`\theta_{\rm head}`$ 是输出头参数。它没有新增 LoRA 的低秩乘积，也不额外插入 Adapter 分支，所以存储和任务切换开销较小。

限制在于它只能调整模型已有偏置，表达能力有限。原方法主要在 BERT 类模型的小到中等任务数据上验证，不能无条件推广到任意大模型生成任务。现代解码器有些线性层不带偏置，RMSNorm 也未必有可调 bias，可能几乎没有可用参数。实践中要打印实际训练参数，检查没有误解冻整层，并评估任务效果。冻结权重仍可能参与输入梯度传播，因此参数极少也不意味着全部中间激活都可以省掉。

```math
\min_{b,\theta_{\rm head}}\mathcal L\big(f_{W_0,b}(x),y\big),\qquad W_0\ \text{fixed}
```

#### 易错点

- 对bias-free模型宣称与BERT一样能直接BitFit。
- 认为可训练参数少就意味着全部激活都不需要反向保存。

#### 追问

- 哪些bias最可能改变attention或FFN行为？
- 训练输出头应不应该计入可训练参数比例？

<a id="ft-018"></a>
### FT-018 · P-Tuning v1、P-Tuning v2 与 Prompt/Prefix Tuning 怎样区分？

**L2**

#### 答案

这些连续提示方法的共同点，是学习一组向量来引导模型，区别在于向量放在哪里、如何产生，以及是否还更新底座。Prompt Tuning 通常只在输入 embedding 前加软提示；Prefix Tuning 常为多个注意力层加入前缀键和值，而 P-Tuning v1、v2 还涉及不同的提示组织和训练设计。

P-Tuning v1 在模板里插入连续提示，常通过提示编码器，如双向循环网络或多层感知机，生成这些向量，并与离散文字提示结合。原方法有冻结和联合微调设置，所以不能只听名称就断言底座一定冻结。P-Tuning v2 把连续提示放进多个层，给各层更直接的任务条件，与深层 Prefix 方法联系紧密，也扩展到序列标注等任务，并不只是把 v1 的提示编码器做大。

例如输入端软提示必须通过整个网络影响最后输出，多层提示则能直接改变中间层的注意力条件。具体实现究竟增加输入 token、K/V，是否采用重参数化，输出头是否训练，都要看配置。公式中 $`\theta_0`$ 是冻结底座，$`\theta_p`$ 是提示参数，$`\theta_h`$ 是任务头，只描述冻结设置。虚拟 token 不一定对应可读词，较长提示也会增加上下文或缓存成本。比较应控制参数、长度和训练预算，不能把某个实验里接近全参数的效果理解成普遍等价。

```math
\theta^*=\arg\min_{\theta_p,\theta_h}\mathcal L\big(f_{\theta_0,\theta_p,\theta_h}(x),y\big),\qquad\theta_0\ \text{fixed in the frozen setting}
```

#### 易错点

- 虚拟 token 不一定能解码成普通文本。
- P-Tuning v2 的主要区别包括多层提示，而不只是更大提示编码器。

#### 追问

- 层级前缀为什么比只放输入端有更短的梯度作用路径？
- deep prompt对KV cache和上下文预算有哪些影响？

<a id="ft-019"></a>
### FT-019 · AdaLoRA 如何自适应分配低秩预算，与固定 rank LoRA 有何区别？

**L2**

#### 答案

AdaLoRA 会根据任务中的重要性，把有限的低秩容量分给不同矩阵，避免每层都用同样 rank 而浪费预算。固定 LoRA 先人为指定每个目标矩阵的秩；AdaLoRA 则边训练边判断哪些方向更值得保留，逐步把不重要的方向裁掉。

它把增量写成 $`\Delta W=P\Lambda Q`$，$`P`$ 和 $`Q`$ 表示左右方向，$`\Lambda`$ 是各方向的可学习尺度，形式类似奇异值分解，但不表示每一步都精确分解完整底座。训练会跟踪方向的梯度敏感性和重要性，在全局预算下调整不同层的有效 rank。比如注意力某层更需要适配，就可能保留更多方向，而另一层被裁掉一部分。正交正则让这些方向少重叠，公式中的 $`I`$ 是单位矩阵，$`\lVert\cdot\rVert_F`$ 是矩阵元素平方和开方的范数。

通常从较大预算逐步降到目标预算，所以预算调度、重要性平滑和裁剪时机都影响结果。过早裁剪可能损失后面才显得重要的容量，也不能只凭权重大小判断任务重要性。它增加了状态和调度复杂度，未必总比固定 LoRA 更稳定或更快。公平比较要固定总预算、目标模块和数据；部署可合并浮点增量，但量化底座的合并与再次量化仍需单独验证。

```math
\Delta W=P\Lambda Q,\qquad\mathcal L_{\rm orth}=\lVert P^\top P-I\rVert_F^2+\lVert QQ^\top-I\rVert_F^2
```

#### 易错点

- 类似奇异值分解的增量参数化，不是每步精确分解底座权重。
- 动态预算和裁剪时机也是训练设置，比较时要控制总预算。

#### 追问

- 何时裁剪方向会造成不可逆的容量损失？
- 正交约束为何有助于比较不同奇异方向的重要性？

<a id="ft-020"></a>
### FT-020 · AdapterFusion 和 AdapterDrop 如何组合知识或降低 adapter 开销？

**L2**

#### 答案

AdapterFusion 是学习怎样组合已有任务适配器的知识，AdapterDrop 是减少执行哪些适配器的开销，二者分别关注复用和效率。普通 Adapter 在主干旁边增加小模块，不同任务可以各自保存一份；Fusion 不直接平均它们的权重，而是学习组合它们的输出。

Fusion 通常先分别训练各任务 Adapter，再冻结它们和底座，训练融合模块，让当前 token 按需要读取不同适配器的表示。例如处理一个同时涉及问答和实体识别的任务，可以让融合层选择两类适配器提供的特征。公式里的 $`\alpha_a(h)`$ 是输入表示 $`h`$ 对第 $`a`$ 个适配器的组合权重，权重和为一；这个式子表达组合直觉，具体查询、键和值的结构依实现而定。

AdapterDrop 则在部分层跳过适配器分支，训练中让模型适应这种保留策略，推理时减少小模块的执行。它跳过的是 Adapter，不是删除整个 Transformer 层。融合多个 Adapter 增加存储和前向计算，Drop 可能损失精度，因此都要测任务效果和真实延迟。若主干计算才是主要瓶颈，少执行几个适配器的收益也可能有限。多任务服务还需考虑批处理、切换频率和融合数量，不能只看单个任务文件大小。

```math
h'=h+\sum_{a=1}^{A}\alpha_a(h)\,\mathrm{Adapter}_a(h),\qquad\sum_a\alpha_a(h)=1
```

#### 易错点

- 把多个adapter融合误称为权重平均。
- 把AdapterDrop说成删除Transformer主体层。

#### 追问

- 冻结多个adapter后融合层需要哪些训练数据？
- 多个任务adapter在一个batch内如何调度？

<a id="ft-021"></a>
### FT-021 · MAM Adapter、UniPELT 等组合 PEFT 方法为什么要混合不同模块？

**L2**

#### 答案

组合参数高效微调，是希望让不同的小模块各自补足模型的一种适配能力，在有限预算下更灵活地改变行为。Adapter 改隐藏表示，Prefix 改注意力条件，LoRA 改线性投影，它们的位置和作用方式不同，所以组合可能有互补收益，但并不是越叠越好。

MAM，也就是 Mix-And-Match Adapter，结合注意力侧的 Prefix 和前馈侧的并行 Adapter；UniPELT 把 Adapter、Prefix、LoRA 等放进统一框架，再用门控调整分支的参与程度。公式里 $`\mathcal M`$ 是模块集合，$`\Delta_m(h)`$ 是第 $`m`$ 个模块的变化，$`g_m(h)`$ 是门控权重，表达“原表示加上若干受控变化”的直觉。一个是选择固定搭配，一个强调门控组合，不能把它们当作完全相同的方法。

例如某个任务主要需要改变注意力选取信息，Prefix 可能更有效；另一任务更需要特征变换，Adapter 或 LoRA 可能更合适。要证明组合有用，需控制总可训练参数和预算，再做逐模块消融，排除只是参数更多。模块叠加也会增加算子、上下文或 KV 成本，门控可能只激活少数分支。保存和部署要记录每个分支的配置；即使 LoRA 合并了，仍要执行的 Prefix 和 Adapter 不会自动消失。

```math
h'=h+\sum_{m\in\mathcal M}g_m(h)\,\Delta_m(h)
```

#### 易错点

- 把UniPELT门控混合与MAM固定搭配说成同一方法。
- 认为LoRA可merge就意味着所有组合PEFT模块都无推理开销。

#### 追问

- 如何设计消融证明组合收益不是单纯参数更多？
- 输入相关门控与任务固定门控有什么区别？

<a id="ft-023"></a>
### FT-023 · Prompt learning 中 template、verbalizer 与连续提示分别是什么？

**L2**

#### 答案

Prompt learning 里，template 是把输入改写成模型熟悉的任务格式，verbalizer 是把模型输出的词映射回类别，连续提示则是通过训练得到的引导向量。比如情感分类，把评论放进“这段评论是 MASK 的”这个模板，再规定“好”映射正面、“差”映射负面，就是模板加标签词映射。

公式中 $`\mathcal V_c`$ 是类别 $`c`$ 的标签词集合，分子把这些词的概率相加，分母在全部类别标签词里归一化；这种形式主要适用于明确的共同预测位置。模板措辞、标签词的常见程度和分词方式都会影响结果，所以归一化不自动保证类别概率已校准。若生成式标签有多个 token，还要定义序列得分与长度处理，不能直接把不同长度的原始概率无条件比较。

离散提示可以人工写，也可以搜索，例如 AutoPrompt 利用梯度提出离散候选后再筛选；连续提示直接优化 embedding，不要求对应人类可读词。KPT 用知识扩展标签词并筛选、校准，PPT 则关注软提示的预训练初始化。它们分别改模板、标签空间和参数初始化，不能混为一个对象。

实践要在验证集上选择模板与标签词，测试集保持独立，并检查随机种子、样本顺序和措辞扰动的稳定性。只写文本、不更新参数是提示工程；训练软提示向量是 prompt tuning。后者还要保存向量和对应底座配置，部署时加载，不能只复制一段文字来替代。

```math
p(c\mid x)=\frac{\sum_{w\in\mathcal V_c}p_\theta(w\mid\mathrm{template}(x))}{\sum_{c'}\sum_{w\in\mathcal V_{c'}}p_\theta(w\mid\mathrm{template}(x))}
```

#### 易错点

- 把prompt engineering与训练soft prompt混为一谈。
- 类别标签长短不同却直接比较未经处理的序列分数。

#### 追问

- verbalizer校准怎样减小标签词频偏差？
- 连续提示需要怎样保存并与底座配套加载？

<a id="topic-3"></a>
## 训练技巧与排错

<a id="ft-012"></a>
### FT-012 · 微调后的灾难性遗忘怎样发现和缓解？

**L2** · 阿里巴巴 / 腾讯

#### 答案

灾难性遗忘是模型学好新任务时，把原来能做的事情明显做差了，需要用新旧能力同时评估。比如领域问答改善，但通用写作、工具调用或其他语言显著退化，才有证据讨论遗忘；先要排除聊天模板、生成参数或部署加载错配造成的假象。

我会在微调前保存一套固定通用和领域评测，训练中同时看新任务收益和旧任务变化。确认遗忘后，可以混入代表性的旧数据做回放，降低学习率和更新范围，或根据联合验证早停。也可用参数重要性正则等持续学习方法，让重要参数不容易偏离，但这些方法的适用性和成本需要具体实验。

例如领域训练中加入少量通用示范，可能保护对话格式，却也占用领域学习预算，比例并非越大越好。LoRA 冻结原权重，方便移除适配器或回滚，但加上适配器后的函数已经改变，仍然可能出现旧能力退化。实际部署要保存底座、适配器及配置版本，并按目标任务和通用能力完成回归；仅凭新验证集 loss 下降就早停，无法判断有没有遗忘。

#### 易错点

- LoRA 不修改底座，因此部署输出永远不可能遗忘。
- 只用新的 validation set 做早停。

#### 追问

- 混入通用数据为何可能削弱领域适配？
- 怎样区分真实遗忘与生成参数变化？

<a id="ft-013"></a>
### FT-013 · 梯度累积等价于大 batch 吗？变长样本怎么归一化？

**L2**

#### 答案

梯度累积把多个小 batch 的梯度先加起来，最后才更新一次参数，目的是用较少的单次激活显存实现较大的有效 batch。公式中的 $`B_{\rm micro}`$ 是每卡每次处理的样本数，$`K_{\rm accum}`$ 是累积次数，$`N_{\rm DP}`$ 是数据并行卡数，三者相乘给出常见的有效 batch 估计，最后不足一个累积窗口时需要单独处理。

能否等价于一次大 batch，关键是目标归一化和更新时机。对于语言模型的 token 平均损失，应把整个累积窗口的有效 token 损失相加，再除以总有效 token 数。例如两个小 batch 分别有 100 和 900 个有效 token，把各自平均 loss 再各占一半，会让前者的单个 token 权重变成后者九倍。公式里的 $`\mathcal T_k`$ 是第 $`k`$ 个小 batch 的有效目标集合。

累积期间参数保持不变，优化器和学习率调度只在真正更新时推进，并在更新后清零梯度。数据并行可能已自动平均梯度，训练框架也可能自动除累积次数，需避免重复缩放；中间 micro-step 可使用 no_sync 减少通信。Dropout 的随机结果、浮点累加顺序，以及按 batch 统计的层会让数值或语义出现差别，所以一般说在相应条件下接近或等价。最直接的验证是在小样本上比较两种方式得到的梯度，而不是只看有效 batch 公式。

```math
\begin{aligned}B_{\rm effective}&=B_{\rm micro}K_{\rm accum}N_{\rm DP}\\\mathcal L_{\rm window}&=\frac{\sum_{k=1}^{K_{\rm accum}}\sum_{t\in\mathcal T_k}\mathrm{NLL}_{k,t}}{\sum_{k=1}^{K_{\rm accum}}|\mathcal T_k|}\end{aligned}
```

#### 易错点

- 每个 batch loss 除累积次数，就认为变长序列始终等价。
- 累积时忘记清零梯度或每个 micro-step 都更新 scheduler。

#### 追问

- 如何验证累积与一次大 batch 的梯度接近？
- 为什么梯度累积降低显存却可能降低吞吐？

<a id="ft-014"></a>
### FT-014 · gradient checkpointing 节省什么，为何会变慢？

**L2**

#### 答案

激活检查点是训练时少保存一些中间结果，反向传播需要时再重算，以额外计算换显存。它和把模型存到磁盘的训练断点是两件事，也不减少权重和优化器状态。

普通反向要用到很多前向激活，检查点只保留某些边界输入。例如一组连续 Transformer 层，中间激活暂不保存，反向来到这组层时，再用边界输入恢复所需结果。检查点划分越细或策略不同，显存和重算开销就不同；长序列的激活很大时收益通常更明显。LoRA 虽然冻结底座，仍要把梯度传到可训练分支，所以同样可能需要这些激活。

重算必须与原前向保持一致。Dropout 的随机状态、函数里的状态修改、设备迁移和数据依赖分支，都可能影响正确性。PyTorch 的可重入与非可重入实现，对梯度条件和行为有差异，应按实际版本核对；尤其要确认冻结输入情况下适配器梯度没有丢失。开启后用任务 loss、梯度、峰值显存和有效 token/s 做检查，不能假设总能固定节省某个比例。无反向的普通推理通常用不到这种机制，它也不会直接减少生成时的历史 KV 缓存。

#### 易错点

- 激活重算与保存训练断点文件不同。
- 普通无反向推理不能靠激活 checkpointing 减少 KV 缓存。

#### 追问

- 为什么 LoRA 的输入梯度配置会影响 checkpointing？
- 如何选择重算粒度与激活卸载？

<a id="ft-015"></a>
### FT-015 · SFT loss 下降但任务效果变差，应该怎样排查？

**L2**

#### 答案

SFT loss 下降只说明模型越来越会模仿训练目标，任务效果变差时，应先核对数据与实现，再判断是否过拟合。真实任务考的是答案正确、格式合规或业务成功，训练交叉熵并不直接优化这些全部要求。

我会先抽查完整样本，确认聊天模板、回答标签、结束符和截断都正确。例如长样本如果只留下提示词、答案被截掉，loss 统计就可能没有业务意义；如果部署模板不同，也可能让训练正常的模型表现异常。接着固定生成设置，同时看训练损失、独立验证损失和任务指标，按领域、长度、格式及难度分析失败案例。

如果训练损失持续降低、验证和真实任务变差，才进一步考虑过拟合、重复数据或错误教师答案。可以逐个实验减少训练轮数、降低学习率、缩小适配器容量、增加正则或调整数据配比。若两种损失都下降而业务指标下降，还要检查模型是否过度学习冗长风格、类别先验或与实际需求不符的目标。每次改动应有独立验证集和失败样例支持，不能看到效果差就继续加 epoch，也不能只凭训练 loss 接近零判断训练成功。

#### 易错点

- 一看到效果差就继续加 epoch。
- 只观察总体 loss，不看大量被 mask 或截断的样本。

#### 追问

- 训练 loss 接近 0 是否一定是好事？
- 如何区分过拟合与生成参数造成的差异？

<a id="topic-4"></a>
## 知识蒸馏与模型编辑

<a id="ft-022"></a>
### FT-022 · 知识蒸馏中的 logits、隐藏层和生成答案监督各有什么作用？

**L2** · 字节跳动 / 深势科技

#### 答案

知识蒸馏是让较小的学生模型学习教师的行为，不同监督信号分别教“下一步概率”“中间表示”和“整段答案”。Logits 蒸馏保留教师对其他候选的判断，隐藏层蒸馏约束内部特征，答案蒸馏则把教师生成的文本当作学生的训练示范。

Logits 是候选输出的原始分数，除以温度 $`T`$ 再做 softmax 得到软概率。较高温度能显露非最大候选之间的关系，例如两个相近答案的概率，而不仅告诉学生正确项是哪个。公式中的 $`p_T`$ 和 $`p_S`$ 是教师和学生分布，$`\lambda`$ 控制真实硬标签与软监督的混合；乘 $`T^2`$ 用于近似补偿大温度下梯度缩小，并不是保证各种温度完全等价。教师分布固定时，最小化正向 KL 与软目标交叉熵只差一个不影响学生梯度的教师熵项。

隐藏层监督要用层映射 $`m(\ell)`$ 和投影 $`P_\ell`$ 对齐不同深度、宽度的表示，不能要求每层天然一一相等。教师生成答案做 SFT，跨分词器比较容易，但丢失了完整候选分布，也会继承教师错误和风格。逐 token 概率蒸馏必须保证前缀与词表可对齐，不能对不同分词器直接按下标算 KL。

蒸馏可以发生在基础训练、领域、指令或对齐阶段；学生自己生成前缀，再由教师反馈，是在线策略蒸馏的一种方式。选信号要看教师接口和成本，混合真实标签、过滤错误，并独立评估幻觉、质量和压缩收益。学生容量有限，蒸馏不保证完整保留教师知识，也不保证学生必然超过教师。

```math
\begin{aligned}p_T&=\mathrm{softmax}(z_T/T),\quad p_S=\mathrm{softmax}(z_S/T)\\\mathcal L&=(1-\lambda)\mathcal L_{\rm hard}+\lambda T^2D_{\rm KL}(p_T\Vert p_S)\\\mathcal L_{\rm hidden}&=\sum_\ell\lVert H_S^{(\ell)}P_\ell-H_T^{(m(\ell))}\rVert_F^2\end{aligned}
```

#### 易错点

- 不同词表和前缀未经对齐，不能直接逐维算 KL。
- 教师软监督和生成答案都可能有错，不能替代独立验证。

#### 追问

- 为什么学生自己生成的前缀可能需要教师再标注？
- 温度变大时T²补偿的近似前提是什么？

<a id="ft-024"></a>
### FT-024 · 模型编辑与继续训练、RAG 有何区别？ROME、MEMIT 与 MEND 如何修改知识？

**L2** · 腾讯

#### 答案

模型编辑是用少量事实更正，定向改变模型的回答，同时尽量保护无关能力；继续训练通常调整更广的分布，RAG 则在推理时提供外部证据而不必改底座。比如要把某个实体的属性改正，编辑希望只影响相关问题，不让其他知识一起变乱。

ROME 把某些前馈层的事实关联近似看成“键到值”的映射，用带保留约束的秩一权重更新写入目标。公式里 $`\Delta W=uv^\top`$ 是秩一增量，$`k_*`$ 是待修改关联的键表示，$`v_*`$ 是目标值表示；两处 v 角色不同，实际实现需明确。MEMIT 将批量事实更新分配到多个前馈层，MEND 则预先训练小编辑网络，把局部微调梯度转换成受控更新。它们的低秩对象和训练流程不同，不能因为都有小矩阵就等同于 LoRA。

编辑验收不仅看原提示能否答对，还要测同义改写、邻近但不该改变的事实、反向关系、多跳推理和连续批量编辑。单个事实写入成功，不保证所有相关知识已一致更新。例如正向属性改变，模型对反向问法仍可能使用旧关联。要保留可回滚权重，比较简单微调和 RAG，并检查通用能力及困惑度。编辑适合定向纠错探索，不能视为已经保证无副作用维护完整知识库的方法。

```math
\Delta W=u v^{\top}\quad(\text{rank-one update}),\qquad (W+\Delta W)k_*\approx v_*
```

#### 易错点

- 单一提示修改成功，不等于所有改写、反向关联和多跳推理都一致。
- ROME、MEMIT、MEND 和 LoRA 的更新对象与训练流程不同。

#### 追问

- 怎样构造 locality 测试而不只测编辑样例？
- 为什么批量编辑需要关注累计遗忘和更新顺序？

<a id="ft-025"></a>
### FT-025 · OPD 的原理和优化目标是什么，与 SFT、RL 怎样选择？

**L3** · 字节跳动

#### 答案

On-policy distillation，简称 OPD，是让学生先自己生成，再让教师指导学生实际走到的前缀，减少“训练时只见标准路径、上线时却接着自己错误往下写”的差距。On-policy 描述反馈来自学生当前策略访问的状态，不唯一规定使用哪种 KL 方向或哪种优化器。

例如学生解题写错了第一步，教师仍要在这个已出现的前缀上给出下一步分布或纠正，训练才能覆盖学生常犯的情况。GKD 一类方法固定本轮采样文本，在同一状态 $`s_t=(x,y_{<t})`$ 上比较教师 $`p_T`$ 和学生 $`p_S`$ 的完整下一 token 分布；$`x`$ 是提示词，$`y_{<t}`$ 是学生前缀，$`\pi_{\rm roll}`$ 是生成这些轨迹的策略。更新学生时不通过离散采样本身反传，而是对这些固定状态上的散度求梯度，可用正向 KL、反向 KL 或 Jensen–Shannon 散度，并混合已有数据。

正向 KL 通常更强调覆盖教师支持的模式，反向 KL 更倾向集中到教师高概率模式，但实际取决于容量和任务。还有策略梯度式的反向 KL 实现，用学生所选 token 的教师 log-prob 构造稠密信号，它与完整分布匹配的梯度路径不同，不能混写。

SFT 用给定答案建立任务基础，教师固定答案的 SFT 是序列蒸馏，但不因教师存在就自动变成 OPD。RL 则按环境、验证器或偏好奖励优化结果。教师可靠且接口可用时，OPD 有较细的监督；可验证任务需要超越模仿目标时，RL 也值得考虑，三者可以组合。要看教师质量、查询成本和独立任务表现，不能保证 OPD 总比 RL 好或学生必然超过教师。

```math
\begin{aligned}s_t&=(x,y_{\lt t}),\quad y\sim\pi_{\mathrm{roll}}(\cdot\mid x)\\ \mathcal L_{\mathrm{OPD}}(\theta)&=\mathbb E_{x,y}\left[\frac1{|y|}\sum_t D\big(p_T(\cdot\mid s_t),p_S^\theta(\cdot\mid s_t)\big)\right]\\ D_{\mathrm{F}}&=D_{\mathrm{KL}}(p_T\Vert p_S),\quad D_{\mathrm{R}}=D_{\mathrm{KL}}(p_S\Vert p_T)\end{aligned}
```

#### 易错点

- 固定 rollout 的分布匹配，不应对离散采样过程直接反传。
- 学生轨迹提供状态，教师提供监督；直接把自己输出当正例不自动成为蒸馏。

#### 追问

- 学生与教师初始差距很大时，为什么可能先做SFT再提高on-policy比例？
- 为什么OPD也应测幻觉、任务正确率和输出多样性，而不只看蒸馏loss？

<a id="ft-026"></a>
### FT-026 · 拿不到教师 logits 时还能做 OPD 吗，只有文本反馈有哪些限制？

**L3** · 字节跳动

#### 答案

拿不到完整教师 logits 仍可能做在线蒸馏，但要先分清是否能拿到所需 token 的概率：只有文本、只有教师自己生成的概率、以及能给学生任意续写打分，是三种不同接口。只有最后一种概率接口，才可能直接评价学生实际选择的动作。

同词表、同前缀下，若动作 $`a_t`$ 真从学生分布 $`p_S`$ 采样，$`\log p_S(a_t\mid s_t)-\log p_T(a_t\mid s_t)`$ 的期望是反向 KL。这里 $`s_t`$ 是当前提示和历史，$`p_T`$ 是教师概率。单个差值可以是负数，KL 的非负性约束期望。这个采样项能估计 KL 数值，但固定 token 后直接对差值反传，通常遗漏采样分布随学生参数变化的导数，不能当作正确的 KL 梯度；需要有依据的策略梯度或重要性采样处理。温度或 top-p 改变采样分布时，也要重新说明目标。

如果完全只有文本，可以让学生先生成，在它实际访问的前缀上请教师提供下一步或下一段正确续写，再用这些标签做交叉熵，迭代刷新并聚合数据。公式中的 $`d_{\rm roll}`$ 是学生访问的状态分布，$`q_T`$ 是教师文本标签的分布。教师贪心的一条答案是硬标签；若能按教师条件分布反复采样，才可看作对应交叉熵的采样估计。教师若重写整个历史，就不再是相同状态的续写监督。

教师也可以评分或比较完整答案，再训练奖励或偏好目标，但评分不是生成概率，更不等于逐 token KL。仅收集教师对原问题的完整答案属于序列蒸馏；把学生自己的输出直接当正例则没有新增教师信息。无论接口如何，都应过滤教师错误并核算在线查询成本。

```math
\begin{aligned}\hat d_t&=\log p_S(a_t\mid s_t)-\log p_T(a_t\mid s_t),\quad a_t\sim p_S(\cdot\mid s_t)\\ \mathbb E_{a_t}[\hat d_t]&=D_{\mathrm{KL}}(p_S\Vert p_T)\\ \mathcal L_{\mathrm{label}}&=\mathbb E_{s\sim d_{\mathrm{roll}},\ z\sim q_T(\cdot\mid s)}[-\log p_S(z\mid s)]\end{aligned}
```

#### 易错点

- 单个反向 KL 采样差可以为负，且无偏数值估计不等于直接反传得到无偏梯度。
- 文本评分不是教师生成概率；只有文本通常无法精确恢复完整词表分布。

#### 追问

- 教师不能续写assistant前缀时，应该如何调整任务接口和监督目标？
- 教师仅给top-k或生成样本的log-prob，怎样判断它是否足够评价学生轨迹？

<a id="ft-027"></a>
### FT-027 · 教师与学生词表不同，跨 tokenizer 的 OPD 怎样定义对齐和损失？

**L3** · 字节跳动

#### 答案

教师和学生使用不同分词器时，必须先定义共同的文本位置和概率事件，不能把双方第几个 token 或词表第几个元素直接放在一起算 KL。同一句中文、空格或 emoji，可能被切成不同数量的 token，位置相同不代表语义相同。

第一步是把学生轨迹还原成共同文本，再用各自分词器处理，按原文的 UTF-8 字节偏移合并覆盖同一段的 token，检查开头、结束和零宽特殊符号。这样能对齐文本段，却还没解决概率空间。沿一条 token 路径相加 log-prob，只得到这条路径的概率，不一定是所有能输出同一文本的路径总概率。

一种明确但粗粒度的目标，是在双方都具备完整下一 token 分布、且候选能唯一确定下个输出字节的共同边界，把概率汇总到“下一字节、EOS、其他特殊输出”等公共类别。公式中的 $`m_i`$ 把模型 $`i`$ 的每个 token 映射到一个类别，$`q_i`$ 汇总这些概率；类别必须互斥而且完整，因此概率和为一，才可计算普通 KL。它只匹配第一字节等粗信息，不等价于完整文本分布匹配。

GOLD 等方法还可做文本段和词汇匹配，对匹配部分比较概率，对不匹配部分用其他排序目标；但某些合并向量并非归一化概率，不能直接冒充严格 KL。只有 top-k 时，要说明剩余质量和共同的 OTHER 事件；两边不同尾部集合不能仅靠同名桶视作同一事件。若拿不到概率，可退回教师文本纠正或序列蒸馏。实现要用中文、前导空格、emoji 和不同 BOS 设置测试对齐覆盖率、边界与概率和。

```math
\begin{aligned}q_i(b\mid u)&=\sum_{v\in V_i:\,m_i(v;u)=b}p_i(v\mid u),\quad i\in\{T,S\}\\ \sum_{b\in\mathcal B}q_i(b\mid u)&=1\\ \mathcal L_{\mathrm{coarse}}&=\mathbb E_{u\sim d_{\mathrm{roll}}}\left[\sum_{b\in\mathcal B}q_T(b\mid u)\log\frac{q_T(b\mid u)}{q_S(b\mid u)}\right]\end{aligned}
```

#### 易错点

- 首字节汇总只定义粗粒度公共事件，不能宣称等价完整文本 KL。
- 两边不同 top-k 的剩余集合不一定是相同事件，未知尾部也不能冒充完整概率。
- 单一分词路径得分不一定等于所有同文输出路径的总概率。

#### 追问

- 为什么UTF-8字节对齐能修正位置问题，却仍不能自动解决词表概率比较？
- 怎样用中文、emoji、前导空格和不同BOS设置验证对齐覆盖率及概率和？

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
- [PEFT: LoRA](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora)
- [PEFT: Quantization](https://huggingface.co/docs/peft/developer_guides/quantization)
- [PEFT LoRA — initialization and LoraConfig](https://huggingface.co/docs/peft/main/en/package_reference/lora)
- [PEFT official implementation: LoRA layer](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/layer.py)
- [PEFT LoRA Linear 初始化官方实现](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [PEFT: Prompt tuning](https://huggingface.co/docs/peft/main/en/package_reference/prompt_tuning)
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
- [Continual Learning Through Synaptic Intelligence](https://arxiv.org/abs/1703.04200)
- [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)
- [Performing gradient accumulation with Accelerate](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)
- [torch.utils.checkpoint — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/checkpoint.html)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [BitFit: Simple Parameter-efficient Fine-tuning](https://arxiv.org/abs/2106.10199)
- [GPT Understands, Too](https://arxiv.org/abs/2103.10385)
- [P-Tuning v2](https://arxiv.org/abs/2110.07602)
- [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)
- [AdapterFusion: Non-Destructive Task Composition](https://arxiv.org/abs/2005.00247)
- [AdapterDrop: On the Efficiency of Adapters in Transformers](https://arxiv.org/abs/2010.11918)
- [Towards a Unified View of Parameter-Efficient Transfer Learning](https://arxiv.org/abs/2110.04366)
- [UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning](https://arxiv.org/abs/2110.07577)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)
- [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/abs/1908.09355)
- [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)
- [Pre-train, Prompt, and Predict](https://arxiv.org/abs/2107.13586)
- [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://arxiv.org/abs/2010.15980)
- [Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification](https://arxiv.org/abs/2108.02035)
- [PPT: Pre-trained Prompt Tuning for Few-shot Learning](https://arxiv.org/abs/2109.04332)
- [Locating and Editing Factual Associations in GPT](https://rome.baulab.info/)
- [Mass Editing Memory in a Transformer](https://memit.baulab.info/)
- [Fast Model Editing at Scale](https://arxiv.org/abs/2110.11309)
- [On-policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/html/2306.13649)
- [TRL Distillation Trainer](https://huggingface.co/docs/trl/distillation_trainer)
- [Thinking Machines Lab: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html)
- [Sequence-Level Knowledge Distillation](https://aclanthology.org/D16-1139/)
- [TRL General Online Logit Distillation (GOLD) Trainer](https://huggingface.co/docs/trl/gold_trainer)
- [Unlocking On-Policy Distillation for Any Model Family](https://huggingfaceh4-on-policy-distillation.hf.space/)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [TRL Reducing Memory Usage — packing and padding-free](https://huggingface.co/docs/trl/main/en/reducing_memory_usage)
- [PyTorch SDPA — masks, shapes and GQA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
