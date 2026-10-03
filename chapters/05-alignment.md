# RLHF、DPO、PPO 与 GRPO

[返回仓库首页](../README.md) · [来源目录](../SOURCES.md) · [阅读全部题目](../exports/questions.jsonl)

L1 基础 · L2 进阶 · L3 深入。难度为编辑判断；社区线索不等于雇主确认真题。所有答案为独立整理。

## 题目导航

- [ALN-001 · SFT、RLHF 与 DPO 分别解决什么问题？](#aln-001)
- [ALN-002 · 奖励模型如何用成对偏好训练？](#aln-002)
- [ALN-003 · PPO 的概率比、clip 和 min 分别起什么作用？](#aln-003)
- [ALN-004 · GAE 如何计算，λ 与 γ 如何影响优势估计？](#aln-004)
- [ALN-005 · RLHF 的 KL 惩罚与 PPO 新旧策略约束有什么区别？](#aln-005)
- [ALN-006 · DPO 的损失如何从 KL 正则化 RLHF 目标推出？](#aln-006)
- [ALN-007 · DPO 的 β 和参考模型如何理解与调参？](#aln-007)
- [ALN-008 · PPO 与 DPO 在工程上如何选型？](#aln-008)
- [ALN-009 · 如何把点赞、点踩和日志变成高质量偏好数据？](#aln-009)
- [ALN-010 · GRPO 为什么不需要独立价值模型？](#aln-010)
- [ALN-011 · GRPO 组内标准差归一化带来哪些问题？](#aln-011)
- [ALN-012 · GRPO 的长度偏差与 Dr. GRPO 有什么关系？](#aln-012)
- [ALN-013 · RLVR 的可验证奖励如何设计？](#aln-013)
- [ALN-014 · ORM 与 PRM 的区别和信用分配难点是什么？](#aln-014)
- [ALN-015 · 如何识别和缓解 reward hacking？](#aln-015)
- [ALN-016 · 后训练为什么会出现对齐税或遗忘？](#aln-016)
- [ALN-017 · RLAIF 和 Constitutional AI 如何工作？](#aln-017)
- [ALN-018 · IPO 等 DPO 变种主要试图解决什么问题？](#aln-018)
- [ALN-019 · 离线偏好优化和在线 RL 的分布差异是什么？](#aln-019)
- [ALN-020 · 多目标奖励发生冲突时如何处理？](#aln-020)

<a id="aln-001"></a>
## ALN-001 · SFT、RLHF 与 DPO 分别解决什么问题？

**L1 · 社区题目线索** · 标签：SFT / RLHF / DPO

**30 秒回答**

SFT 用优质示范教模型按指令作答；典型 RLHF 用偏好训练奖励模型，再以 RL 优化策略；DPO 用偏好对直接更新策略。三者都依赖数据质量，不能仅凭是否出现某个 bad case 判断算法优劣。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- SFT 通常最小化目标回答的负对数似然，重点是行为和任务示范。
- 典型 RLHF 为 SFT→偏好/RM→策略优化；现实流水线可迭代或混合。
- DPO 仍表达偏好奖励，只省去独立 RM 拟合与在线 RL 更新环节。

### 易错点

- 不要说 SFT 只学知识、RL 只学推理；数据与任务同样重要。

### 面试官可能追问

- 同一 bad case 何时补 SFT 示范，何时补偏好对？

</details>

**技术依据**

- [APP-S101 · Training language models to follow instructions with human feedback](https://arxiv.org/html/2203.02155v1)
- [APP-S104 · Direct Preference Optimization](https://arxiv.org/html/2305.18290v3)

**题目出处线索**

- [APP-S009 · 字节大模型算法实习生：电商业务（已 oc）](https://www.nowcoder.com/discuss/724319940982898688) · `reported_question`：正文问过 SFT/DPO 的选择及 SFT 无法解决的 bad case。

<a id="aln-002"></a>
## ALN-002 · 奖励模型如何用成对偏好训练？

**L2 · 编辑补充题** · 标签：Reward Model / Bradley-Terry

**30 秒回答**

常见奖励模型给完整回答输出标量，使用同一 prompt 下赢家与输家的分差拟合偏好概率。Bradley–Terry 是建模假设，不是所有人类偏好的真实规律；奖励绝对零点通常不能由成对比较唯一确定。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 输入是 (x,y_w,y_l)，把 r_w−r_l 送入 sigmoid 后做二元负对数似然。
- 对同一 x 给所有回答加相同常数不改变比较，所以 RM 分数需要校准。
- 按领域、长度、标注者分层检查，防止奖励只学到长回答或固定套话。

### 公式

```text
\mathcal L_{\rm RM}=-\mathbb E\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
```

### 易错点

- RM 高准确率不代表策略优化后分布外仍可靠。

### 面试官可能追问

- 如何处理平局与偏好不传递？

</details>

**技术依据**

- [APP-S160 · Reward Modeling — TRL](https://huggingface.co/docs/trl/reward_trainer)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="aln-003"></a>
## ALN-003 · PPO 的概率比、clip 和 min 分别起什么作用？

**L2 · 社区题目线索** · 标签：PPO / importance ratio

**30 秒回答**

PPO 用新旧策略概率比修正同批旧策略样本，以 advantage 判断增减概率方向。clip 加 min 截断有利方向上的过度收益，减少大幅更新动机；它不是参数裁剪，也不保证更新后的所有概率比都在区间内。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 定义 ρ_t=π_θ(a_t|s_t)/π_old(a_t|s_t)，old 是本轮采样策略。
- A>0 时限制继续增加概率的收益；A<0 时限制过度减少概率的收益。
- 多轮小批更新仍需监测 KL、clip fraction、熵和 value loss。

### 公式

```text
L^{\rm clip}=\mathbb E[\min(\rho_t A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)]
```

### 易错点

- clip 不是严格信赖域约束，异常 KL 可触发早停。

### 面试官可能追问

- 去掉外面的 min 后，负 advantage 会有什么错误？

</details>

**技术依据**

- [APP-S105 · PPO — Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [APP-S102 · Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

**题目出处线索**

- [APP-S007 · 大模型算法岗面试题复盘：RAG、Agent、评测](https://nanhubrain.csdn.net/6a3ce7a8662f9a54cb841a94.html) · `reported_question`：正文问 PPO 为何比普通 policy gradient 稳定。
- [APP-S008 · 字节面经：大模型算法岗面经 04](https://www.nowcoder.com/discuss/922308546966847488) · `secondary_report`：公开汇总问 clip 后为何再取 min；属于二手题目。

<a id="aln-004"></a>
## ALN-004 · GAE 如何计算，λ 与 γ 如何影响优势估计？

**L2 · 社区题目线索** · 标签：GAE / Critic

**30 秒回答**

GAE 将多个 TD 残差按 γλ 衰减求和，利用价值模型在偏差和方差间折中。λ 越小越依赖局部 bootstrap，λ 接近一越接近回报减 baseline；γ 决定折扣目标，不能把二者混为同一个超参数。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- δ_t=r_t+γV(s_{t+1})−V(s_t)，从轨迹末尾递推优势。
- λ<1 在价值估计不准时可引入偏差；λ=1 通常方差更高。
- 终止、超时截断、padding 的 bootstrap/mask 必须与任务定义一致。

### 公式

```text
\hat A_t^{\rm GAE}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l},\quad \delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
```

### 易错点

- 标准 outcome GRPO 用组内相对奖励，通常没有 GAE/value critic。

### 面试官可能追问

- 生成被 max_tokens 截断时怎样处理末状态价值？

</details>

**技术依据**

- [APP-S103 · Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438)

**题目出处线索**

- [APP-S008 · 字节面经：大模型算法岗面经 04](https://www.nowcoder.com/discuss/922308546966847488) · `secondary_report`：汇总出现 advantage 计算及“GRPO 的 GAE”措辞；答案明确纠正概念。

<a id="aln-005"></a>
## ALN-005 · RLHF 的 KL 惩罚与 PPO 新旧策略约束有什么区别？

**L2 · 社区题目线索** · 标签：KL / reference model

**30 秒回答**

RLHF 的参考 KL 约束策略偏离参考模型，通常参考 SFT 初始化；PPO 的 clip/KL 则控制当前更新相对采样旧策略的变化。一个约束整体行为漂移，一个稳定局部更新，两个模型的更新节奏也不同。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- RLHF 目标常为 E[r]−βKL(π_θ||π_ref)，参考模型通常冻结。
- PPO old policy 随 rollout 刷新，概率比用于同批样本的策略更新。
- token 上 logπ_θ−logπ_ref 可用于采样估计，单个 token 值不必非负。

### 易错点

- KL 惩罚不能保证事实正确或彻底阻止 reward hacking。

### 面试官可能追问

- 为何增大 β 可能伤害偏好奖励提升？

</details>

**技术依据**

- [APP-S161 · Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/pdf/2307.04964)

**题目出处线索**

- [APP-S007 · 大模型算法岗面试题复盘：RAG、Agent、评测](https://nanhubrain.csdn.net/6a3ce7a8662f9a54cb841a94.html) · `reported_question`：正文明确问 RLHF 为什么需要 KL penalty。

<a id="aln-006"></a>
## ALN-006 · DPO 的损失如何从 KL 正则化 RLHF 目标推出？

**L3 · 社区题目线索** · 标签：DPO / 推导

**30 秒回答**

先解带参考 KL 的奖励最大化，得到最优策略正比于参考策略乘 exp(r/β)；再把奖励改写为策略对数比。代入同一 prompt 的 Bradley–Terry 偏好模型后分区函数抵消，得到直接拟合偏好概率的 DPO 损失。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 需要 β>0、参考分布覆盖候选支持集，且使用相应偏好模型假设。
- r=βlog(π*/π_ref)+βlogZ(x)，比较两个回答时相同 Z(x) 消失。
- 有限数据、受限模型与优化误差意味着实训不能保证恢复理论最优策略。

### 公式

```text
\mathcal L_{\rm DPO}=-\mathbb E\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_{\rm ref}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{\rm ref}(y_l|x)}\right]\right)
```

### 易错点

- 目标推导对应不等于 DPO 和任意 PPO 实训过程完全等价。

### 面试官可能追问

- 当偏好出现循环时，BT 标量奖励假设有什么局限？

</details>

**技术依据**

- [APP-S104 · Direct Preference Optimization](https://arxiv.org/html/2305.18290v3)

**题目出处线索**

- [APP-S001 · 字节多模态大模型面经一面](https://www.nowcoder.com/discuss/932594519835443200) · `search_snippet`：搜索结果明确出现 DPO loss 题。

<a id="aln-007"></a>
## ALN-007 · DPO 的 β 和参考模型如何理解与调参？

**L2 · 编辑补充题** · 标签：DPO / beta

**30 秒回答**

理论 KL 目标中 β 越大，最优策略对奖励的响应越受约束；DPO 损失中 β 同时缩放偏好 logit 与梯度，实训效果还受学习率、数据和训练时长影响。应观察 KL、偏好胜率与通用能力，而不是机械套用单调规律。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 参考模型定义行为锚点与对数概率比，常用对应任务的 SFT 模型。
- 相同数据下用多组 β 做验证，观察 chosen/rejected 隐式奖励间距与 KL。
- 参考模型与数据生成分布不匹配时，先检查格式、长度和支持覆盖。

### 易错点

- 不要把 β 大简单解释成每步梯度必然更小。

### 面试官可能追问

- 同样 β，换参考模型为什么结果会变？

</details>

**技术依据**

- [APP-S159 · DPO Trainer — TRL](https://huggingface.co/docs/trl/dpo_trainer)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="aln-008"></a>
## ALN-008 · PPO 与 DPO 在工程上如何选型？

**L2 · 社区题目线索** · 标签：PPO / DPO / 选型

**30 秒回答**

DPO 适合已有可靠偏好对、希望以较低工程成本建立基线的场景；PPO 可在线采样并优化显式奖励，便于探索但流水线复杂。没有算法普遍胜出的结论，必须在同数据、预算、奖励和评测条件下比较。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 比较 RM 质量、rollout 成本、value/参考模型显存与调参预算。
- DPO 受离线偏好覆盖限制；PPO 也会利用奖励模型漏洞、产生分布偏移。
- 先做 SFT/DPO 基线，再判断在线探索的收益是否足以覆盖复杂度。

### 易错点

- “PPO 一定更强”或“DPO 一定更稳定”都缺少条件。

### 面试官可能追问

- 奖励可执行验证且偏好很少时，你会优先试哪条路线？

</details>

**技术依据**

- [APP-S108 · Is DPO Superior to PPO for LLM Alignment?](https://arxiv.org/abs/2404.10719)

**题目出处线索**

- [APP-S001 · 字节多模态大模型面经一面](https://www.nowcoder.com/discuss/932594519835443200) · `search_snippet`：搜索摘要报告 PPO/DPO trade-off 问题。

<a id="aln-009"></a>
## ALN-009 · 如何把点赞、点踩和日志变成高质量偏好数据？

**L2 · 社区题目线索** · 标签：偏好数据 / 反馈偏差

**30 秒回答**

用户反馈并不自动组成同 prompt 的可比偏好对。应先统一任务、上下文与评分准则，再为同问题生成或匹配候选，排除位置、长度和曝光偏差，抽样复核赢家是否真的更正确，保留不确定与平局记录。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 点踩可能来自答案错误、体验延迟或立场差异，需分类后进入对应训练集。
- 对比候选保持工具证据与用户需求可比，避免把缺证据回答当负例。
- 训练/验证按用户、模板、文档或时间切分，避免近重复泄漏。

### 易错点

- 不能把不同用户不同问题的点赞答案与点踩答案直接配对。

### 面试官可能追问

- AI 评分生成偏好时怎样估计标注噪声？

</details>

**技术依据**

- [APP-S109 · UltraFeedback](https://arxiv.org/html/2310.01377v1)

**题目出处线索**

- [APP-S003 · 面了一轮 Agent 岗，我把问过的问题整理成了文章](https://ac.nowcoder.com/discuss/1680599) · `search_snippet`：搜索可见“线上点赞点踩如何变成 DPO 数据”题。

<a id="aln-010"></a>
## ALN-010 · GRPO 为什么不需要独立价值模型？

**L2 · 社区题目线索** · 标签：GRPO / PPO

**30 秒回答**

原始 GRPO 对同一问题采样一组回答，用组内奖励均值作相对 baseline，再归一化形成优势，代替 PPO 中常见的价值模型估计。它节省 critic 的参数与训练开销，但仍需 rollout、奖励和可能的参考模型。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- outcome supervision 将同一回答的相对终局奖励用于其各 token 优势。
- 策略更新仍使用新旧 token 概率比与 clipped surrogate。
- 显存还受组大小、序列长度、优化器、激活与推理 KV cache 影响。

### 易错点

- 去掉 critic 不等于没有 baseline，也不等于没有奖励模型。

### 面试官可能追问

- 如何分离 rollout 显存和训练显存预算？

</details>

**技术依据**

- [APP-S106 · DeepSeekMath](https://arxiv.org/html/2402.03300v3)

**题目出处线索**

- [APP-S006 · Awesome-LLM-Interview-Questions-and-Answers](https://github.com/DolbyUUU/Awesome-LLM-Interview-Questions-and-Answers) · `reported_topic`：README 列 GRPO 与 RL 显存主题。

<a id="aln-011"></a>
## ALN-011 · GRPO 组内标准差归一化带来哪些问题？

**L3 · 编辑补充题** · 标签：GRPO / 归一化

**30 秒回答**

组内奖励减均值再除标准差能统一数值尺度，但会按每题的奖励方差重加权。二值奖励全对或全错时没有相对区分信号，低非零方差的组也可能放大少数样本；ε 只防数值异常，不能创造学习信号。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 原始 A_i=(r_i−mean(r))/(std(r)+ε)，属于特定 GRPO 定义。
- 组大小改变 baseline/方差估计与出现混合对错样本的概率。
- 可比较去 std、跨 batch 归一化、难度采样等方案，需同步控制总采样预算。

### 易错点

- 不要声称组内标准化天然无偏或对所有任务更优。

### 面试官可能追问

- 奖励为连续多维分数时，量纲怎样影响组优势？

</details>

**技术依据**

- [APP-S162 · GRPO Trainer — TRL](https://huggingface.co/docs/trl/grpo_trainer)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="aln-012"></a>
## ALN-012 · GRPO 的长度偏差与 Dr. GRPO 有什么关系？

**L3 · 编辑补充题** · 标签：GRPO / 长度偏差

**30 秒回答**

按每条响应长度取 token loss 均值会改变不同长度样本的权重：同样正优势下短答的单 token 激励更强，负优势下长错答的单 token 惩罚更弱。Dr. GRPO 移除响应长度和组标准差归一化，以常数缩放恢复其讨论的目标。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 问题在 loss reduction 的分母；response mean、batch token mean 和固定常数并不相同。
- 该分析不能简单归纳为“GRPO 一定偏爱短回答”，错误长答也可能被相对优待。
- 核对实际 mask、EOS、截断规则与采样长度，做长度/正确率联合消融。

### 易错点

- 某项修正有益不等于对所有训练目标无偏；需说明比较的目标函数。

### 面试官可能追问

- 全 batch token mean 会怎样改变样本之间的相对权重？

</details>

**技术依据**

- [APP-S107 · Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/html/2503.20783v2)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="aln-013"></a>
## ALN-013 · RLVR 的可验证奖励如何设计？

**L2 · 编辑补充题** · 标签：RLVR / verifier

**30 秒回答**

RLVR 使用可执行或规则可核对的结果作奖励，例如数学答案检查、代码单测。先定义正确性和环境边界，隔离格式分与内容分，再验证评分器能否识别投机。可验证奖励减少主观标注，却不能保证覆盖所有推理质量。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 数学要处理等价表达和解析失败；代码要用隔离执行、超时与隐藏测试。
- 格式奖励权重过大可能诱导只满足格式；正确结果也未必说明过程可靠。
- 用对抗样本、人工复核和未参与训练的任务监测评分器漏洞。

### 易错点

- 一个正则或少量公开测试不是完备的 correctness oracle。

### 面试官可能追问

- 工具调用成功但业务目标未完成，应怎样给奖励？

</details>

**技术依据**

- [APP-S110 · DeepSeek-R1](https://arxiv.org/html/2501.12948v1)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="aln-014"></a>
## ALN-014 · ORM 与 PRM 的区别和信用分配难点是什么？

**L2 · 社区题目线索** · 标签：ORM / PRM / 信用分配

**30 秒回答**

ORM 针对最终结果评分，PRM 在推理步骤提供反馈，能定位早期错误但需要昂贵且一致的步骤标注。过程分还涉及划分粒度与聚合方式；终局奖励回传给所有 token 并不意味着每个 token 对正确性贡献相同。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 先明确一步的边界与可验证性；自然语言分步可被模型改变粒度投机。
- PRM 能用于训练信号或推理时排序，两者用途与验证方案不同。
- 论文中数学任务收益不应直接外推为所有开放领域任务都优于 ORM。

### 易错点

- 步骤看似合理不代表整个推理或最终结果正确。

### 面试官可能追问

- Agent 的检索、工具、答复步骤如何构造过程评分？

</details>

**技术依据**

- [APP-S111 · Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)

**题目出处线索**

- [APP-S008 · 字节面经：大模型算法岗面经 04](https://www.nowcoder.com/discuss/922308546966847488) · `secondary_report`：公开汇总包含 Agentic RL 的过程评分与 token 分数问题。

<a id="aln-015"></a>
## ALN-015 · 如何识别和缓解 reward hacking？

**L2 · 编辑补充题** · 标签：奖励投机 / Goodhart

**30 秒回答**

奖励是业务目标的代理，优化过强时模型可能利用它的盲点，出现分数上涨而真实质量下降。要分开训练奖励和独立验证指标，追踪长度、套话、异常工具行为，使用人工抽检、对抗样本与更新评分器控制投机。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 典型症状包括重复关键词、讨好裁判、输出空安全答案或绕过弱验证。
- 参考 KL、提前停止、多样化反馈可减缓漂移，但不保证消除漏洞。
- 保存 rollout 样本与奖励分项，检查高分 bad case 是否系统性聚集。

### 易错点

- GRPO 的相对优势或换成 LLM judge 本身不是 reward hacking 的解药。

### 面试官可能追问

- 如果训练奖励上升、独立胜率下降，你先停哪一环？

</details>

**技术依据**

- [APP-S112 · Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="aln-016"></a>
## ALN-016 · 后训练为什么会出现对齐税或遗忘？

**L2 · 社区题目线索** · 标签：alignment tax / 遗忘

**30 秒回答**

后训练把优化重点转向特定任务与偏好，可能让部分原有能力下降，称为对齐税或遗忘现象。要对比基座、SFT、偏好训练各阶段的同条件评测，使用任务数据混合、正则与预训练回放寻找能力和行为控制的平衡。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 退化也可能来自回答格式、拒答或解码差异，先排除评测设置变化。
- InstructGPT 的 PPO-ptx 混入预训练目标缓解部分基准退化，但并未解决全部。
- 按数学、代码、多语、安全分别监测，避免只看偏好总分。

### 易错点

- LoRA 或小学习率可降低变化幅度，但不能保证没有遗忘。

### 面试官可能追问

- 如何区分知识丢失和评测 parser 失配？

</details>

**技术依据**

- [APP-S101 · Training language models to follow instructions with human feedback](https://arxiv.org/html/2203.02155v1)

**题目出处线索**

- [APP-S006 · Awesome-LLM-Interview-Questions-and-Answers](https://github.com/DolbyUUU/Awesome-LLM-Interview-Questions-and-Answers) · `reported_topic`：README 有模型微调通用能力与灾难性遗忘主题。

<a id="aln-017"></a>
## ALN-017 · RLAIF 和 Constitutional AI 如何工作？

**L2 · 编辑补充题** · 标签：RLAIF / Constitutional AI

**30 秒回答**

RLAIF 用模型生成偏好或反馈，降低逐条人工标注成本；Constitutional AI 按一组人类制定原则进行自我批评修订，再使用 AI 偏好训练奖励和策略。监督来源改变并不等于完全无人参与或自动消除偏见。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 原则、提示、评分模型与训练数据共同决定偏好，需明确可审计版本。
- 自我修订样本可用于监督训练，AI 比较标签可用于偏好/RL 阶段。
- 保留专家标注验证集，对少数群体、领域事实和过度拒答独立检查。

### 易错点

- 模型反馈可能传播同类错误，数据量增加不能替代质量核验。

### 面试官可能追问

- 同一个模型生成和打分会产生哪些相关误差？

</details>

**技术依据**

- [APP-S113 · Constitutional AI](https://arxiv.org/abs/2212.08073)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="aln-018"></a>
## ALN-018 · IPO 等 DPO 变种主要试图解决什么问题？

**L3 · 社区题目线索** · 标签：IPO / 偏好建模

**30 秒回答**

偏好学习方法会改变偏好概率映射、正则方式或数据利用方式。IPO 从更一般的成对偏好目标出发，讨论 DPO 对偏好转标量奖励等假设的依赖。面试应先讲要解决的假设与过拟合问题，再比较损失和实证条件。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 人类成对偏好可能不传递，无法总由一个标量奖励精确表示。
- 变种不能只凭名字判断更好，需要看噪声、样本覆盖、参考策略与预算。
- 比较相同验证集的偏好胜率、KL、长度分布和任务正确率。

### 易错点

- 不要把所有 DPO 变种说成只是“加一项 SFT loss”。

### 面试官可能追问

- 若所有标注都绝对偏好同一答案，怎样监测过拟合？

</details>

**技术依据**

- [APP-S114 · A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)

**题目出处线索**

- [APP-S009 · 字节大模型算法实习生：电商业务（已 oc）](https://www.nowcoder.com/discuss/724319940982898688) · `reported_question`：正文明确问 DPO 缺点与相关变种。

<a id="aln-019"></a>
## ALN-019 · 离线偏好优化和在线 RL 的分布差异是什么？

**L2 · 社区题目线索** · 标签：offline / online / distribution shift

**30 秒回答**

离线方法主要在固定候选分布上学习，简单可复现，却可能缺少当前策略的困难负例；在线方法更新采样分布，能探索新回答，也让评分器面对分布外样本。选型需看反馈成本、任务奖励及可承担的 rollout 预算。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 数据生成策略与训练策略差距增大会影响有效覆盖，尤其长回答和罕见任务。
- 可迭代生成偏好数据补覆盖，但要保留独立测试与版本追踪。
- 在线 rollout 与多 epoch 更新间有策略滞后，应监测新旧策略比值。

### 易错点

- 在线 DPO 等变体存在，不能把 DPO 永久等同于固定离线流程。

### 面试官可能追问

- 怎样发现离线数据覆盖不足，而非优化器没收敛？

</details>

**技术依据**

- [APP-S108 · Is DPO Superior to PPO for LLM Alignment?](https://arxiv.org/abs/2404.10719)
- [APP-S164 · Online DPO Trainer — TRL](https://huggingface.co/docs/trl/online_dpo_trainer)

**题目出处线索**

- [APP-S010 · 字节大模型实习算法面经 55min](https://www.nowcoder.com/feed/main/detail/59b472d7ec6645cc95ac0735885234e3) · `search_snippet`：搜索摘要列离线/在线强化学习与项目使用方式。

<a id="aln-020"></a>
## ALN-020 · 多目标奖励发生冲突时如何处理？

**L3 · 社区题目线索** · 标签：多目标 / 安全对齐

**30 秒回答**

正确性、简洁度、有用性与安全并不总能同时提高。应先区分必须满足的约束与可权衡指标，校准各奖励尺度，再用分层约束、加权或约束优化建立可解释权衡，报告各维结果而非一个掩盖冲突的总分。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 权重前先看奖励量纲、方差和样本分布，防止大数值项支配训练。
- 严重违规可定义约束或门禁；一般质量属性可探索 Pareto 前沿。
- 保留各项奖励与失败案例，评估过度拒答及正确但冗长等边界。

### 易错点

- 奖励简单相加不保证满足硬约束，也不保证各项单调改善。

### 面试官可能追问

- 如何确定安全阈值并防止模型输出空答案拿高分？

</details>

**技术依据**

- [APP-S115 · Safe RLHF](https://arxiv.org/abs/2310.12773)

**题目出处线索**

- [APP-S010 · 字节大模型实习算法面经 55min](https://www.nowcoder.com/feed/main/detail/59b472d7ec6645cc95ac0735885234e3) · `search_snippet`：搜索摘要出现多目标奖励冲突题。
