# 强化学习、RLHF 与偏好优化

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 目录

- [强化学习基础](#topic-1)
  - [ALN-021 · 强化学习怎样建模为 MDP？V、Q、Bellman 与 LLM 生成怎样对应？](#aln-021)
  - [ALN-022 · 动态规划、Monte Carlo、TD 与 n-step return 有什么区别？](#aln-022)
  - [ALN-023 · 怎样推导策略梯度与 REINFORCE，baseline 为什么能降方差？](#aln-023)
  - [ALN-024 · Actor-Critic、A2C 与 A3C 怎样工作，Critic 是奖励模型吗？](#aln-024)
  - [ALN-034 · on-policy、off-policy、importance sampling、TRPO 与 PPO 怎样联系？](#aln-034)
  - [ALN-035 · Q-learning 与 SARSA 的更新、探索策略和适用边界有什么区别？](#aln-035)
  - [ALN-036 · DQN 的经验回放和目标网络做什么，Double DQN 与 Dueling DQN 分别改了什么？](#aln-036)
  - [ALN-037 · DDPG、D4PG、TD3 与 SAC 怎样比较，能直接套到 LLM token 生成吗？](#aln-037)
  - [ALN-038 · 稀疏奖励怎样改善，reward shaping、课程学习与 ICM 有什么风险？](#aln-038)
  - [ALN-039 · 行为克隆、DAgger、IRL 与 GAIL 有何区别，怎样对应 Agent 训练？](#aln-039)
- [RLHF 与 PPO](#topic-2)
  - [ALN-001 · SFT、RLHF 与 DPO 分别解决什么问题？](#aln-001)
  - [ALN-003 · PPO 的概率比、clip 和 min 分别起什么作用？](#aln-003)
  - [ALN-004 · GAE 如何计算，λ 与 γ 如何影响优势估计？](#aln-004)
  - [ALN-005 · RLHF 的 KL 惩罚与 PPO 新旧策略约束有什么区别？](#aln-005)
  - [ALN-008 · PPO 与 DPO 在工程上如何选型？](#aln-008)
  - [ALN-025 · RLHF-PPO 的四模型完整流程是什么？Critic 的 V_target 从哪里来？](#aln-025)
  - [ALN-026 · PPO 需要多少张 GPU，怎样估算四模型、rollout 与训练显存？](#aln-026)
- [DPO 与偏好数据](#topic-3)
  - [ALN-006 · DPO 的损失如何从 KL 正则化 RLHF 目标推出？](#aln-006)
  - [ALN-007 · DPO 的 β 和参考模型如何理解与调参？](#aln-007)
  - [ALN-009 · 如何把点赞、点踩和日志变成高质量偏好数据？](#aln-009)
  - [ALN-018 · IPO 等 DPO 变种主要试图解决什么问题？](#aln-018)
  - [ALN-019 · 离线偏好优化和在线 RL 的分布差异是什么？](#aln-019)
- [GRPO 与在线优化](#topic-4)
  - [ALN-010 · GRPO 为什么不需要独立价值模型？](#aln-010)
  - [ALN-011 · GRPO 组内标准差归一化带来哪些问题？](#aln-011)
  - [ALN-012 · GRPO 的长度偏差与 Dr. GRPO 有什么关系？](#aln-012)
  - [ALN-013 · RLVR 的可验证奖励如何设计？](#aln-013)
  - [ALN-014 · ORM 与 PRM 的区别和信用分配难点是什么？](#aln-014)
  - [ALN-027 · GSPO 与 GRPO 的 importance ratio、clip 和梯度单位有什么区别？](#aln-027)
  - [ALN-028 · DAPO 相比原始 GRPO 改了什么，四个核心设计分别解决什么问题？](#aln-028)
  - [ALN-029 · 在 verl 中支持 DAPO，需要改哪些配置或训练模块？](#aln-029)
  - [ALN-030 · GRPO 数据必须标注 Thought 吗，完整训练数据与 rollout 怎样组织？](#aln-030)
  - [ALN-031 · GRPO 不收敛或训练奖励升高但能力退化，怎样排查和调参？](#aln-031)
  - [ALN-032 · 正负样本不对称设计有哪些方式，和 PPO/DAPO 的不对称 clip 有何区别？](#aln-032)
- [奖励与对齐策略](#topic-5)
  - [ALN-002 · 奖励模型如何用成对偏好训练？](#aln-002)
  - [ALN-015 · 如何识别和缓解 reward hacking？](#aln-015)
  - [ALN-016 · 后训练为什么会出现对齐税或遗忘？](#aln-016)
  - [ALN-017 · RLAIF 和 Constitutional AI 如何工作？](#aln-017)
  - [ALN-020 · 多目标奖励发生冲突时如何处理？](#aln-020)
  - [ALN-033 · 提升 RAG 回答质量时怎样选 DPO 或 GRPO，奖励怎样设计？](#aln-033)

<a id="topic-1"></a>
## 强化学习基础

<a id="aln-021"></a>
### ALN-021 · 强化学习怎样建模为 MDP？V、Q、Bellman 与 LLM 生成怎样对应？

**L1**

#### 答案

强化学习优化交互轨迹的期望累计奖励。MDP 用状态、动作、转移、奖励和折扣描述序贯决策；马尔可夫性质要求当前状态包含预测下一步所需的信息。仅有状态转移的是马尔可夫过程，增加奖励得到奖励过程，再增加动作与策略得到决策过程；在 MDP 上固定策略，可诱导一个奖励过程。

LLM 的状态可取 prompt 加已生成前缀，动作是下一个 token，终止由 EOS 或任务规则决定。V 是按策略继续行动的期望回报，Q 是先选某动作再继续的回报；Bellman 方程把当前价值分成即时奖励与后续价值。求最优 Q 后可选择使 Q 最大的动作，但最优策略可能不唯一，不能说 V 与策略存在一一对应。

若观察不完整，使用历史、记忆或 belief state 建模部分可观测性。深度学习是函数表示/优化手段，RL 是学习范式，两者不是对立关系；RL 也可能依赖答案标签、人工反馈或模拟器。

$$
\begin{aligned}G_t&=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}\\ V^\pi(s)&=\mathbb E_{a\sim\pi,\,s^\prime,r}[r+\gamma V^\pi(s^\prime)]\\ Q^\pi(s,a)&=\mathbb E_{s^\prime,r}[r+\gamma\mathbb E_{a^\prime\sim\pi}Q^\pi(s^\prime,a^\prime)]\end{aligned}
$$

#### 易错点

- 不能把“RL不需要标签”“所有状态必须能重复到达”当作定义。

#### 追问

- LLM Agent 的外部网页状态不可见时，怎样构造状态？

<a id="aln-022"></a>
### ALN-022 · 动态规划、Monte Carlo、TD 与 n-step return 有什么区别？

**L2**

#### 答案

动态规划使用已知环境模型计算期望并迭代评价/改进策略；MC 用采样轨迹的实际回报估计价值，通常需要等到终止；TD 用一步奖励加下一状态的估计值更新，可以在线学习但会 bootstrap。Model-based 方法使用已知或学到的转移/奖励模型规划，model-free 方法直接从交互学习价值或策略；前者有模型误差，后者也不天然保证泛化更好。

n-step 把前 n 步实际奖励与末状态价值组合。n 增大通常减少对 bootstrap 的依赖，但累计更多随机奖励可能增加方差；这只是常见权衡，不是期望、偏差与方差都必然单调变化的定理。MC 回报在合适采样条件下可作为 V 的无偏样本，TD 目标则受当前价值误差影响，不能简单说两种最终学习算法“永远无偏”。

真实终止令后续价值为零；时间限制截断未必表示环境终止，需要按任务语义处理 bootstrap。GAE 可看作不同步长优势信息的加权组合。

$$
\begin{aligned}V(s_t)&\leftarrow V(s_t)+\alpha\big[r_{t+1}+\gamma V(s_{t+1})-V(s_t)\big]\\ G_t^{(n)}&=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n V(s_{t+n})\end{aligned}
$$

#### 易错点

- “n越大，价值期望越大”不成立；须区分目标的偏差、估计方差与真实价值。

#### 追问

- 为什么截断成 max_tokens 的回答不应无条件当作真实终止？

<a id="aln-023"></a>
### ALN-023 · 怎样推导策略梯度与 REINFORCE，baseline 为什么能降方差？

**L2**

#### 答案

把轨迹概率写成初始状态、环境转移与各步策略概率的乘积；对期望回报求导，使用 likelihood-ratio 恒等式把梯度转成回报乘轨迹 log-prob 梯度。环境不依赖策略参数时，只有动作概率贡献梯度；再利用因果性，把整条轨迹奖励换成该动作之后的 reward-to-go。

REINFORCE 按当前策略采样轨迹，计算回报，最小化负的 log-prob 与回报乘积。这里回报 G_t 从时刻t开始折扣，若目标是从时刻0起的折扣回报，求和还需乘γ的t次方；LLM有限响应常取γ=1。采样分布每轮变化，代理loss的数值跨轮不能直接作固定监督目标比较，关键是梯度与独立任务奖励。

baseline b(s) 不依赖当前采样动作时，其期望梯度贡献为零，可以降低方差。用 V(s) 得到回报减价值的优势估计。实现中回报/优势应 detach，避免策略梯度通过权重误反传；状态访问权重及折扣须与定义的目标一致。

$$
\begin{aligned}\nabla_\theta J&=\mathbb E_\tau\left[\sum_t \gamma^t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\big(G_t-b(s_t)\big)\right]\\ \mathbb E_{a\sim\pi}[b(s)\nabla_\theta\log\pi(a\mid s)]&=b(s)\nabla_\theta\sum_a\pi(a\mid s)=0\end{aligned}
$$

#### 易错点

- 任意与动作相关的 baseline 都不会产生偏差，这一说法是错的。

#### 追问

- 为什么提高好回答的概率不等于只训练这些回答的 SFT？

<a id="aln-024"></a>
### ALN-024 · Actor-Critic、A2C 与 A3C 怎样工作，Critic 是奖励模型吗？

**L2**

#### 答案

Actor 输出策略，Critic 估计当前策略下的未来回报，帮助构造优势并降低策略梯度方差。Actor 可用负 log-prob 乘优势训练，Critic 回归 n-step 或其他 return target；优势、回归目标通常停止梯度。共享骨干时仍需区分两类损失的作用及相互影响。

A3C 让多个 worker 与独立环境交互，从共享参数同步局部副本，再异步提交更新；多环境有助于分散样本相关性，但异步带来参数滞后。A2C 将多个环境的 rollout 同步聚合后更新，便于批量计算。两者通常归入 on-policy 方法，但 A3C 的异步滞后不能被忽略；是否 off-policy 要看数据与学习目标，不由“多线程”决定。

RLHF 的 RM 评价回答质量，Critic 预测包含 RM/KL 等定义后未来能获得的回报，二者不同。优势 A=Q−V 是期望函数；单步 TD residual 是它的一种估计，不能处处当成完全相等的真值。

$$
\mathcal L_{\mathrm{actor}}=-\mathbb E[\log\pi_\theta(a_t\mid s_t)\operatorname{sg}(\hat A_t)],\qquad \mathcal L_{\mathrm{critic}}=\mathbb E[(V_\phi(s_t)-\operatorname{sg}(\hat G_t))^2]
$$

#### 易错点

- Critic 不是给回答标好坏的判别器，也不一定需要独立估计 Q 网络。

#### 追问

- Actor 与 Critic 共用 Transformer 时，应怎样检查梯度冲突？

<a id="aln-034"></a>
### ALN-034 · on-policy、off-policy、importance sampling、TRPO 与 PPO 怎样联系？

**L2**

#### 答案

On-policy学习当前交互策略，off-policy允许数据由不同的行为策略产生，学习另一个目标策略。是否经验回放、异步或多轮更新不是唯一判据，须明确行为策略与优化目标及所用校正。

重要性采样用目标/行为密度比重加权行为样本；目标有概率的区域必须被行为分布覆盖。轨迹比例是逐步比例的乘积，长轨迹容易造成高方差。截断、自归一化或使用局部surrogate常降低方差，但通常引入偏差或改变目标，不能声称“PPO clip使任意旧样本都完全无偏”。

TRPO用KL约束限制更新，通常涉及二阶近似和线搜索；PPO-clip采用一阶优化的截断代理目标，PPO-penalty则采用KL惩罚。clip并不是严格KL信任域保证，仍应监控KL/采用早停。RLHF参考模型KL又是行为锚点约束，与本轮新旧策略限制分开。

$$
\mathbb E_{z\sim p}[f(z)]=\mathbb E_{z\sim q}\left[\frac{p(z)}{q(z)}f(z)\right],\qquad p(z)>0\Rightarrow q(z)>0
$$

#### 易错点

- PPO通常以on-policy方式收集数据，再有限复用旧策略rollout；不能因存在ratio就称任意off-policy回放算法。

#### 追问

- 为什么重要性比例在长回答上常比短回答更不稳定？

<a id="aln-035"></a>
### ALN-035 · Q-learning 与 SARSA 的更新、探索策略和适用边界有什么区别？

**L2**

#### 答案

两者都是基于采样转移的TD控制方法，区别在下一状态target。Q-learning用下一状态所有动作Q的最大值，学习趋向贪心最优策略；行为仍可用epsilon-greedy探索，因此是off-policy。SARSA使用下一步按当前行为策略实际选择的动作价值，学习包含当前探索行为的策略，因此是on-policy。

Q值是期望回报，不是动作概率；由价值导出的策略也可随机探索。两种方法都使用环境交互数据，不能说Q-learning“不使用行为策略数据”。在悬崖行走等任务，SARSA可能考虑探索风险，而Q-learning的贪心target看起来更激进，但这不表示SARSA始终是更优改进版。

表格型收敛结论需要充分访问、步长条件与有限状态动作等假设，换成神经网络后不能直接套用。连续动作上的max较难，常借助Actor-Critic；LLM词表可作为离散动作，但长程状态、信用分配与计算成本仍令直接Q学习具有挑战。

$$
\begin{aligned}Q_{\mathrm{Q\text{-}learn}}(s_t,a_t)&\leftarrow Q(s_t,a_t)+\alpha[r_{t+1}+\gamma\max_aQ(s_{t+1},a)-Q(s_t,a_t)]\\ Q_{\mathrm{SARSA}}(s_t,a_t)&\leftarrow Q(s_t,a_t)+\alpha[r_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)]\end{aligned}
$$

#### 易错点

- SARSA下一动作要按行为策略选择，但不必实际执行完下一动作后才更新当前transition。

#### 追问

- 若行为策略和target都贪心，两种一步更新还会有什么差别？

<a id="aln-036"></a>
### ALN-036 · DQN 的经验回放和目标网络做什么，Double DQN 与 Dueling DQN 分别改了什么？

**L2**

#### 答案

DQN用神经网络近似动作价值，回归Bellman target。经验回放打散连续样本相关性并复用数据，目标网络暂时固定bootstrap目标以降低追逐同一网络的振荡；二者都不保证训练样本完全独立，也不提供深度函数逼近下的一般收敛保证。

普通DQN用target network同时选最大动作并评价，其max可能放大估计噪声。Double DQN用online network选动作、target network估值，缓解过估计。Dueling将网络分成状态价值V与动作优势A，再聚合成Q；减去动作优势均值等约束处理分解不可辨识，便于在多个动作价值相近时学习状态质量。

Double是target构造的变化，Dueling是网络结构的变化，可以组合。目标网络与actor-critic中的价值模型也不是同一个概念，DQN本身没有独立输出动作的Actor。

$$
\begin{aligned}y_{\mathrm{DDQN}}&=r+\gamma Q_{\mathrm{target}}(s^\prime,\arg\max_aQ_{\mathrm{online}}(s^\prime,a))\\ Q(s,a)&=V(s)+A(s,a)-\frac1{|\mathcal A|}\sum_{a^\prime}A(s,a^\prime)\end{aligned}
$$

#### 易错点

- 不能把Double DQN的两个网络称为两个独立Actor，也不能把Dueling称为对抗训练。

#### 追问

- 目标网络更新太频繁或太慢分别可能产生什么问题？

<a id="aln-037"></a>
### ALN-037 · DDPG、D4PG、TD3 与 SAC 怎样比较，能直接套到 LLM token 生成吗？

**L2**

#### 答案

这些方法主要讨论连续动作控制。DDPG用确定性Actor给出动作，Critic学习Q，通过Q对动作的梯度更新Actor；经验回放、目标网络和行为探索噪声使其属于off-policy Actor-Critic。

D4PG在确定性策略梯度框架加入分布式采样、回报分布表示、n-step与优先回放等设计；其中distributional指回报分布，distributed指多个采样者，二者不能混同。TD3用双Critic取较小target、延迟策略更新和target动作平滑缓解估计误差。SAC学习随机策略，优化回报与策略熵的组合，提高探索和鲁棒性。

LLM token是离散词表动作，普通确定性连续动作梯度不能直接穿过采样token；应使用适配离散动作的目标与实现，而非把DDPG的网络名换成Transformer就当作等价算法。SAC存在离散变体，但需要具体说明动作分布、Q计算成本与采样方式。

$$
J_{\mathrm{SAC}}=\mathbb E\left[\sum_t\gamma^t\big(r_t+\alpha\mathcal H(\pi(\cdot\mid s_t))\big)\right]
$$

#### 易错点

- “策略梯度只适用于连续动作、价值方法只适用于确定策略”都不是正确边界。

#### 追问

- 离散大词表上计算所有动作Q值与连续动作Actor求导有何成本差异？

<a id="aln-038"></a>
### ALN-038 · 稀疏奖励怎样改善，reward shaping、课程学习与 ICM 有什么风险？

**L2**

#### 答案

稀疏奖励只在少数完成事件提供反馈，探索可能长时间收不到有效信号。可以缩短任务、用课程控制难度、加入示范、设计可验证过程奖励或改善探索。密集奖励若鼓励与最终目标无关的代理行为，会造成reward hacking，因此必须单独验证最终成功率。

势函数型shaping用下一状态势能减当前势能的折扣值，在匹配的折扣及终止条件下可以保留最优策略；任意加上“每步多思考一点就给分”没有这样的保证。ICM在特征空间学习逆动力学表征与前向预测，并用预测误差作内在奖励，引导探索难以预测的状态。

预测误差不等于业务价值，随机噪声也可持续提供奖励，产生“noisy TV”问题。对LLM Agent不应直接把新奇token、更多工具调用或更长轨迹当成功；应设内在奖励权重、资源约束与外部目标验收。

$$
\begin{aligned}r_t^\prime&=r_t+\gamma\Phi(s_{t+1})-\Phi(s_t)\\ r_t^{\mathrm{intrinsic}}&\propto\left\|f(\phi(s_t),a_t)-\phi(s_{t+1})\right\|_2^2\end{aligned}
$$

#### 易错点

- 更密集的奖励不保证原任务最优策略不变；好奇心也不保证探索有用信息。

#### 追问

- 如果Agent不断调用返回随机数的工具，怎样发现和修复奖励作弊？

<a id="aln-039"></a>
### ALN-039 · 行为克隆、DAgger、IRL 与 GAIL 有何区别，怎样对应 Agent 训练？

**L2**

#### 答案

行为克隆把专家状态/动作当监督数据，直接学习动作分布；训练只看专家访问的状态，部署时自己的小错误会带到未见状态，造成分布偏移和误差累积。DAgger让当前策略运行，在它实际访问的状态上询问专家动作并聚合数据，代价是需要专家反馈与可控交互环境。

IRL从示范推断能解释专家行为的奖励，再求策略，但奖励通常不唯一，专家也未必最优。GAIL用判别器区分专家与当前策略的状态动作占用分布，策略优化对应代理信号；它是模仿学习目标，不意味着必然恢复唯一真实奖励。判别器与策略的对抗结构类似GAN，但样本来自有环境反馈的轨迹。

Agent SFT相当于示范轨迹上的行为学习；数据应包含工具选择、参数、观察与恢复。修复deployment失误时，可以收集当前Agent错误状态上的示范，而非仅增加更多理想轨迹；危险状态不能为采样而无约束执行。

$$
\mathcal L_{\mathrm{BC}}=-\mathbb E_{(s,a)\sim\mathcal D_E}[\log\pi_\theta(a\mid s)]
$$

#### 易错点

- 模仿学习不能保证把专家全部行为精确复制，也不能据示范唯一识别人类真实意图。

#### 追问

- 让专家只重写Agent最终答案，为什么不等于在错误状态上做DAgger？

<a id="topic-2"></a>
## RLHF 与 PPO

<a id="aln-001"></a>
### ALN-001 · SFT、RLHF 与 DPO 分别解决什么问题？

**L1**

#### 答案

SFT 用优质示范教模型按指令完成任务，通常最小化目标回答的负对数似然；它主要提供行为和任务示范。典型 RLHF 在 SFT 后用偏好数据训练奖励模型，再通过强化学习优化策略，也可以迭代或混合不同阶段。

DPO 直接用同一问题的偏好回答对更新策略，省去独立奖励模型拟合和在线 RL 更新，但仍通过目标函数表达偏好对应的奖励关系。三者都依赖数据质量；选型应结合任务、数据和评测，不能凭单个失败案例判断算法优劣。

经典流程先用示范做SFT，再以同prompt回答的成对偏好训练RM，最后用PPO等采样并优化策略。Actor/Critic训练，Reference/RM在RL阶段通常冻结；新一轮rollout刷新old policy，Reference保持行为锚点。SFT解决示范拟合，偏好优化表达相对质量，两者互补，SFT也不是天然无法学习安全与诚实。

#### 易错点

- 不要说 SFT 只学知识、RL 只学推理；数据与任务同样重要。

#### 追问

- 同一 bad case 何时补 SFT 示范，何时补偏好对？

<a id="aln-003"></a>
### ALN-003 · PPO 的概率比、clip 和 min 分别起什么作用？

**L2**

#### 答案

PPO 使用新旧策略的概率比 $\rho_t=\pi_\theta(a_t\mid s_t)/\pi_{\mathrm{old}}(a_t\mid s_t)$，对本轮旧策略采集的样本进行更新；优势为正时鼓励提高动作概率，为负时鼓励降低概率。

clip 与 min 共同截断有利方向上过度更新带来的目标收益：$A_t>0$ 时限制过度增大概率的收益，$A_t<0$ 时限制过度减小概率的收益。它削弱大幅更新的动机，并非裁剪模型参数，也不保证所有概率比都留在区间内。多轮 minibatch 更新应同时监测 KL、clip fraction、熵和 value loss。

$$
L^{\mathrm{clip}}=\mathbb{E}\left[\min\left(\rho_t A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)\right]
$$

#### 易错点

- clip 不是严格信赖域约束，异常 KL 可触发早停。

#### 追问

- 去掉外面的 min 后，负 advantage 会有什么错误？

<a id="aln-004"></a>
### ALN-004 · GAE 如何计算，λ 与 γ 如何影响优势估计？

**L2**

#### 答案

GAE 把多个 TD 残差按 $(\gamma\lambda)^l$ 衰减求和，可在序列末尾向前递推计算。每步残差为 $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$，其中 $\gamma$ 决定折扣目标，$\lambda$ 决定优势估计对多步信息与 bootstrap 的依赖，两者作用不同。

较小的 $\lambda$ 更依赖局部价值估计，价值模型不准时可能引入偏差；$\lambda$ 接近 1 时更接近回报减基线，通常也有更高方差。终止、超时、截断和 padding 对应的 bootstrap 与 mask 必须和任务定义一致，否则即使递推公式正确，优势也可能算错。

$$
\begin{aligned}\hat A_t^{\mathrm{GAE}}&=\sum_{l\geq 0}(\gamma\lambda)^l\delta_{t+l}\\\delta_t&=r_t+\gamma V(s_{t+1})-V(s_t)\end{aligned}
$$

#### 易错点

- 标准 outcome GRPO 用组内相对奖励，通常没有 GAE/value critic。

#### 追问

- 生成被 `max_tokens` 截断时怎样处理末状态价值？

<a id="aln-005"></a>
### ALN-005 · RLHF 的 KL 惩罚与 PPO 新旧策略约束有什么区别？

**L2**

#### 答案

RLHF 的参考策略 KL 惩罚用于约束模型相对行为锚点的漂移，目标可写为 $\mathbb{E}[r]-\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})$；参考策略通常是冻结的 SFT 模型。PPO 的 clip 或新旧策略 KL 则约束相对本轮采样策略的局部更新，旧策略随 rollout 轮次刷新。

两种约束的参照对象和作用范围不同，不能互相替代。实现中可用采样 token 的 $\log\pi_\theta-\log\pi_{\mathrm{ref}}$ 估计参考 KL，但单个采样项未必非负，不能把逐 token 出现负值直接视为实现错误。

β过大可能让策略难以利用任务奖励，过小可能放大奖励过优化与行为漂移。联合监控参考KL、独立胜率/正确率、熵与输出长度，必要时按目标KL自适应调节；不是只让训练RM分数最高。PPO的参考KL系数、clip范围和DPO β作用路径不同，数值不能直接互相移植。

#### 易错点

- KL 惩罚不能保证事实正确或彻底阻止 reward hacking。

#### 追问

- 为何增大 $\beta$ 可能伤害偏好奖励提升？

<a id="aln-008"></a>
### ALN-008 · PPO 与 DPO 在工程上如何选型？

**L2**

#### 答案

有可靠离线偏好对时，DPO 适合作为成本较低、易复现的基线；需要在线探索并能提供明确奖励时，PPO 更有发挥空间，但引入 rollout、价值模型、参考模型及更多调参成本。没有普遍更优的选择，应在一致的数据、预算、奖励和评测条件下比较。

具体需评估奖励模型质量、rollout 成本以及模型与激活的显存占用。DPO 的瓶颈可能是离线候选覆盖不足，PPO 则可能遭遇奖励投机和分布偏移。通常先建立 SFT/DPO 基线，再验证在线学习带来的收益是否值得增加系统复杂度。

若直接用REINFORCE回报，估计方差可能较大；PPO结合Critic/GAE和有限更新约束提高可控性，但也增加价值模型成本。PPO与DPO应在同一初始模型、数据/奖励质量、推理预算和业务评测上比较，并计入在线采样及打分成本，不能拿一个更强SFT起点的DPO与弱起点PPO证明算法优劣。

#### 易错点

- “PPO 一定更强”或“DPO 一定更稳定”都缺少条件。

#### 追问

- 奖励可执行验证且偏好很少时，你会优先试哪条路线？

<a id="aln-025"></a>
### ALN-025 · RLHF-PPO 的四模型完整流程是什么？Critic 的 V_target 从哪里来？

**L2**

#### 答案

经典 RLHF-PPO 中 Actor 和 Critic 训练，Reference 和 RM 通常冻结。RM 先用偏好数据训练，RL 阶段并非四个模型同时各算一份训练 loss。Actor 从 SFT 初始化，Reference 是固定行为锚点，Critic 预测每个生成前缀的后续回报；四个逻辑角色也不意味着必须部署四套同尺寸独立参数。

先用采样旧策略 rollout，缓存响应 token、mask、旧 log-prob 与旧 value；RM 评价整段回答，按约定加入逐 token 参考 KL 惩罚。根据奖励、旧 value 和终止语义计算 GAE，再形成冻结回归目标 V_target=旧 value+优势。Actor 优化 PPO clipped surrogate，Critic 回归目标，可使用 value clipping；更新若干 mini-batch 后刷新采样策略。

V_target 是当前采样批次的估计回报，不是 RM 分数直接广播，也不是新 Critic 自己给自己作标签。只有回答有效位置参与相应 loss，截断是否 bootstrap、KL 正则是否另作损失、价值归一化须保持一致。

$$
\begin{aligned}\tilde r_t&=r_t^{\mathrm{task}}-\beta(\log\pi_{\mathrm{old}}(y_t\mid s_t)-\log\pi_{\mathrm{ref}}(y_t\mid s_t))\\ \hat A_t&=\delta_t+\gamma\lambda\hat A_{t+1},\quad \delta_t=\tilde r_t+\gamma V_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t)\\ \hat V_t^{\mathrm{target}}&=\operatorname{sg}(V_{\mathrm{old}}(s_t)+\hat A_t)\\ \mathcal L_V&=\tfrac12\operatorname{mean}_{t\in\mathcal V}(V_\phi(s_t)-\hat V_t^{\mathrm{target}})^2\end{aligned}
$$

![RLHF-PPO 的 Actor、Reference、Reward Model 和 Critic](../assets/ppo-roles.svg)

四个角色不要求四份独立物理模型；图示经典 Actor 与 Critic 分开训练的流程。

#### 易错点

- 不要让 GAE/return target 通过计算图回传到旧 value，也不要把 reference 与本轮 old policy 混同。

#### 追问

- PPO actor loss 下降而独立胜率降低，先检查哪些日志？

<a id="aln-026"></a>
### ALN-026 · PPO 需要多少张 GPU，怎样估算四模型、rollout 与训练显存？

**L2**

#### 答案

卡数由参数规模、精度、并行切分、回答长度、rollout 数量与每卡显存共同决定，不能仅凭“PPO四模型”给固定数字。先区分可训练 Actor/Critic 的参数、梯度与优化器状态，冻结 Reference/RM 的参数与前向工作区，以及生成引擎的 KV cache、权重副本和缓存轨迹。

常见 BF16 参数/梯度加 FP32 master weights 与 Adam 两个状态，训练状态粗估约16字节/参数；实现未保存 master、状态压缩或LoRA时需改算。冻结 BF16 权重约2字节/参数。ZeRO/FSDP 分片后减少常驻状态，激活、all-gather 临时峰值和通信buffer仍需另计。生成引擎若与训练引擎保留独立副本，也必须计入。

按 rollout、RM/reference 打分、Actor/Critic 更新分别测峰值；角色共卡/分卡与 offload 改变峰值和吞吐。7B Actor+7B Critic+两个7B冻结模型，在上述未分片假设下仅状态就约252 GB十进制，尚未包含激活/KV，不能据此简单平均后断言4张80GB足够。

$$
M_{\mathrm{peak}}\approx\max_{\mathrm{phase}}\big(M_{\mathrm{train\ states}}+M_{\mathrm{frozen\ weights}}+M_{\mathrm{activations}}+M_{\mathrm{KV}}+M_{\mathrm{buffers}}\big)
$$

#### 易错点

- 共享角色显存和串行切换需要明确调度，不能把所有阶段峰值机械相加或全部理想分片。

#### 追问

- 长响应先在 rollout OOM、短响应在 update OOM，优化策略为何不同？

<a id="topic-3"></a>
## DPO 与偏好数据

<a id="aln-006"></a>
### ALN-006 · DPO 的损失如何从 KL 正则化 RLHF 目标推出？

**L3**

#### 答案

从奖励最大化并惩罚参考策略 KL 的目标出发，最优策略满足 $\pi^*(y\mid x)\propto\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta)$。将奖励改写为 $r(x,y)=\beta\log[\pi^*(y\mid x)/\pi_{\mathrm{ref}}(y\mid x)]+\beta\log Z(x)$，再代入 Bradley–Terry 偏好概率；同一问题的奖励差中，$Z(x)$ 项抵消，得到 DPO 损失。

该推导要求 $\beta>0$，参考策略的支持覆盖候选回答，并采用相应的偏好建模假设。训练时用参数化策略替代最优策略，有限数据、模型容量和优化误差仍然存在，因此推导不能保证实际模型一定达到原目标的最优解。

$$
\mathcal{L}_{\mathrm{DPO}}=-\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right]\right)\right]
$$

#### 易错点

- 目标推导对应不等于 DPO 和任意 PPO 实训过程完全等价。

#### 追问

- 当偏好出现循环时，BT 标量奖励假设有什么局限？

<a id="aln-007"></a>
### ALN-007 · DPO 的 β 和参考模型如何理解与调参？

**L2**

#### 答案

参考模型提供偏好优化的行为锚点，通常选用适合任务的 SFT 模型。理论 KL 正则目标中，较大的 $\beta$ 意味着更强的参考策略约束，最优策略对奖励变化的响应较弱；在实际 DPO 损失中，$\beta$ 同时改变偏好 logit 与梯度尺度，其效果还受学习率、数据和训练时长影响，不能简单断言最终 KL 会单调变化。

调参应比较多组 $\beta$，联合观察 chosen/rejected 的隐式奖励差、KL、胜率和通用能力。如果参考模型与偏好数据的生成模型不一致，先检查模板、长度及候选支持是否匹配，再判断超参数的影响。

#### 易错点

- 不要把 β 大简单解释成每步梯度必然更小。

#### 追问

- 同样 $\beta$，换参考模型为什么结果会变？

<a id="aln-009"></a>
### ALN-009 · 如何把点赞、点踩和日志变成高质量偏好数据？

**L2**

#### 答案

点赞、点踩和交互日志不能自动形成可靠偏好对。应先统一任务上下文与评价规则，再生成或匹配可比较候选；标注中排除展示位置、回答长度和曝光差异的影响，核验优选回答的正确性，并保留不确定、平局或不可比较样本。

点踩可能来自事实错误、延迟或立场差异，这些信号应分开处理。涉及工具、证据或用户需求时，候选必须拥有可比较条件，不能把缺少证据直接当作负例。训练和测试还应按用户、模板、文档或时间划分，并处理近重复，防止日志泄漏抬高结果。

#### 易错点

- 不能把不同用户不同问题的点赞答案与点踩答案直接配对。

#### 追问

- AI 评分生成偏好时怎样估计标注噪声？

<a id="aln-018"></a>
### ALN-018 · IPO 等 DPO 变种主要试图解决什么问题？

**L3**

#### 答案

DPO 变种主要调整偏好概率映射、正则化或数据使用方式。IPO 从更一般的成对偏好目标出发，讨论将偏好映射为标量奖励等假设；人类偏好可能非传递，未必能由单一标量奖励精确表达，因此应先说明建模假设和待解决的过拟合问题，再解释损失。

算法名称不能证明效果更好。比较时应控制偏好噪声、候选覆盖、参考模型和训练预算，在相同评测中联合观察胜率、KL、回答长度与正确率，并核对改动是否适用于当前数据条件。

#### 易错点

- 不要把所有 DPO 变种说成只是“加一项 SFT loss”。

#### 追问

- 若所有标注都绝对偏好同一答案，怎样监测过拟合？

<a id="aln-019"></a>
### ALN-019 · 离线偏好优化和在线 RL 的分布差异是什么？

**L2**

#### 答案

离线偏好优化使用固定候选，便于复现，但数据可能缺少当前策略的困难负例，尤其当生成模型与学习模型差距较大时，长尾问题或罕见回答的覆盖会不足。在线 RL 随策略更新采样，能够探索新行为，也更容易进入奖励模型未见过的分布。

选择时需综合反馈任务、奖励可靠性与 rollout 预算。迭代更新偏好数据应保留独立留出集与数据版本；在线训练还要关注采样策略与多轮更新策略之间的滞后，监测概率比，防止样本过旧影响优化。

#### 易错点

- 在线 DPO 等变体存在，不能把 DPO 永久等同于固定离线流程。

#### 追问

- 怎样发现离线数据覆盖不足，而非优化器没收敛？

<a id="topic-4"></a>
## GRPO 与在线优化

<a id="aln-010"></a>
### ALN-010 · GRPO 为什么不需要独立价值模型？

**L2**

#### 答案

GRPO 对同一问题采样一组回答，以组内平均奖励作为基线并标准化相对优势，从而替代 PPO 中独立价值模型的优势估计，节省 critic 参数和训练成本。它仍需要生成 rollout、计算奖励，具体实现还可能保留参考模型。

采用结果奖励时，一条回答的最终优势通常分配给该回答的各个 token，再用新旧策略的 token 概率比构造裁剪目标。整体显存并非只由 critic 决定，还取决于组大小、回答长度、优化器状态、激活和生成 KV cache。

#### 易错点

- 去掉 critic 不等于没有 baseline，也不等于没有奖励模型。

#### 追问

- 如何分离 rollout 显存和训练显存预算？

<a id="aln-011"></a>
### ALN-011 · GRPO 组内标准差归一化带来哪些问题？

**L3**

#### 答案

原始 GRPO 的组内优势通常写为 $\hat A_i=(r_i-\bar r)/(\operatorname{std}(r)+\epsilon)$；标准差的具体定义需与实现一致。减均值提供相对基线，除以标准差则进一步按每个问题的奖励方差重加权：较小但非零的方差可能放大少数回答的信号。

二元奖励下，全对或全错的组没有相对学习信号；$\epsilon$ 只能改善数值稳定性，不能创造奖励差异。组大小影响方差估计及出现混合结果的概率。比较取消标准差归一化、跨 batch 归一化或难度采样时，应控制总采样量，避免把数据分布变化误认为目标函数改进。

#### 易错点

- 不要声称组内标准化天然无偏或对所有任务更优。

#### 追问

- 奖励为连续多维分数时，量纲怎样影响组优势？

<a id="aln-012"></a>
### ALN-012 · GRPO 的长度偏差与 Dr. GRPO 有什么关系？

**L3**

#### 答案

GRPO 的长度偏差与损失归约方式有关。若每条回答先按自身 token 数取均值，短回答的单 token 权重更大；负优势下，较长错误回答的单 token 惩罚相对更小。因此不能把问题概括为“总是偏爱短回答”，它也可能相对偏好较长的错误回答。

Dr. GRPO 讨论移除回答长度和组内标准差归一化，并采用常数尺度。实现中必须区分回答均值、batch token 均值与常数分母，它们给样本的权重不同。评估还需同时固定 mask、EOS、截断与采样长度设置，联合观察长度和正确率。

#### 易错点

- 某项修正有益不等于对所有训练目标无偏；需说明比较的目标函数。

#### 追问

- 全 batch token mean 会怎样改变样本之间的相对权重？

<a id="aln-013"></a>
### ALN-013 · RLVR 的可验证奖励如何设计？

**L2**

#### 答案

RLVR 用可执行规则提供奖励，例如数学答案校验或代码单元测试，以减少主观偏好标注。奖励设计仍要明确正确性与执行环境的边界，并将格式奖励和内容奖励分开；可验证结果并不等于所有推理过程都可靠。

数学任务需处理等价表达与解析失败，代码任务需隔离执行环境、超时及隐藏测试。格式权重过高可能使模型只优化格式，弱测试器也可能被利用。应通过对抗样本、人工复核和独立测试集检查验证器漏洞，防止把测试器投机当作能力提升。

#### 易错点

- 一个正则或少量公开测试不是完备的 correctness oracle。

#### 追问

- 工具调用成功但业务目标未完成，应怎样给奖励？

<a id="aln-014"></a>
### ALN-014 · ORM 与 PRM 的区别和信用分配难点是什么？

**L2**

#### 答案

ORM 对最终结果评分，PRM 对中间步骤评分，后者有机会更早发现错误，但需要一致的步骤标注与额外成本。信用分配的难点是确定哪些步骤真正贡献了结果；把同一结果奖励广播到所有 token，并不意味着每个 token 的因果贡献相同。

PRM 的步骤边界、可验证性及分数聚合方式会影响结果，模型也可能利用评分粒度。用于训练的过程奖励与用于推理候选排序的过程奖励需要分别验证。数学任务中的实验收益不能直接外推到全部开放领域任务。

结果奖励可以只在序列结束给出，经return/GAE产生逐token优势；这不等于逐token直接获得真实过程奖励。步骤PRM、token-level奖励、sequence-level评分与loss聚合是不同粒度，增加过程监督能提供更细信号，但价值估计与评分器都可能出错，不能声称彻底解决因果信用分配。

#### 易错点

- 步骤看似合理不代表整个推理或最终结果正确。

#### 追问

- Agent 的检索、工具、答复步骤如何构造过程评分？

<a id="aln-027"></a>
### ALN-027 · GSPO 与 GRPO 的 importance ratio、clip 和梯度单位有什么区别？

**L3**

#### 答案

GRPO 使用组内相对奖励作优势，常见实现逐 token 计算新旧策略概率比。GSPO 保留组内优势，但用整段回答似然比的长度归一化几何平均作比例，对整段回答进行 clip，使优化单位与序列奖励对齐。

计算时先在有效回答 token 上求新旧 log-prob 差的均值，再指数化；不能先把序列概率直接相乘，否则长回答容易下溢。这个长度归一化比例也不是未经改变的整条轨迹 importance weight，须按论文目标理解。一个响应的 token 梯度共享序列权重，其 clipping 行为与 token 级 GRPO 不同，因此不能直接照搬同样的 epsilon。

论文在特定 Qwen3/MoE 实验中报告稳定性与效率收益；这不保证所有数据、模型与奖励下都优于 GRPO。比较时应固定 rollout 预算、奖励、响应长度和训练算力，并记录 clip fraction、策略漂移和独立能力。

$$
\begin{aligned}s_i(\theta)&=\exp\left(\frac{1}{|y_i|}\sum_t\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})}\right)\\ J_{\mathrm{GSPO}}&=\mathbb E\left[\frac1G\sum_i\min(s_i\hat A_i,\operatorname{clip}(s_i,1-\epsilon,1+\epsilon)\hat A_i)\right]\end{aligned}
$$

#### 易错点

- “GSPO只是把token loss先平均”不足以描述序列比例和clip条件的变化。

#### 追问

- 两个响应某个 token 的 ratio 极端，但序列均值接近1，GSPO与GRPO会怎样不同？

<a id="aln-028"></a>
### ALN-028 · DAPO 相比原始 GRPO 改了什么，四个核心设计分别解决什么问题？

**L3**

#### 答案

DAPO 是一套大规模推理RL方法与系统方案，不能只理解为换一个优势公式。其主要设计包括上下界分开的 Clip-Higher、动态采样、token级损失聚合，以及超长响应奖励处理。

Clip-Higher 放宽正优势概率增加的上边界，以缓解探索受限，同时保留独立下边界。动态采样剔除同一prompt组内全部相同结果的组，并继续采样补够训练批次，使训练看到有区分度的信号，但也改变了训练问题分布并增加生成成本。token-mean 聚合让所有有效响应 token 共同归一化，与“先每序列平均再对序列平均”权重不同，长短回答的梯度贡献会变化。

超长处理通过长度缓冲区等减轻硬截断对奖励的错误归因；论文recipe去掉显式KL也不能外推为所有任务都应去KL。应同时检查熵、非零优势组比例、长度/截断率与独立正确率；只提高训练奖励不能证明能力改善。

$$
J\propto\frac{\sum_{i,t}m_{i,t}\min(\rho_{i,t}\hat A_i,\operatorname{clip}(\rho_{i,t},1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})\hat A_i)}{\sum_{i,t}m_{i,t}}
$$

#### 易错点

- 不对称clip上下界不等于给正负样本直接乘不同数据权重。

#### 追问

- 过滤全对/全错组会对样本难度和每步采样成本产生什么影响？

<a id="aln-029"></a>
### ALN-029 · 在 verl 中支持 DAPO，需要改哪些配置或训练模块？

**L3**

#### 答案

先固定 verl commit 和 DAPO recipe，配置名随版本变动，不能只在普通 PPO脚本里改一个 estimator 就声称完整复现。官方recipe暴露 actor 的 clip_ratio_low/high、loss_agg_mode，algorithm.filter_groups 的开关/指标/生成批次上限，以及 reward_model.overlong_buffer 的长度和penalty。

例如把损失聚合设为 token-mean；组过滤按 acc、score 等指定指标判断同组是否没有差异，反复生成到训练prompt批次满足要求或达到上限。gen_batch_size控制生成候选数，train_batch_size控制实际训练批次，须区分prompt数与乘 rollout.n 后的轨迹数。长度惩罚应基于有效响应长度，排除padding。

若当前trainer不支持过滤后的补采样，需要改rollout收集/组筛选循环；奖励shaping在奖励路径实现，分离clip和归一化在actor loss实现。用小批次检查mask、组ID、长度边界和优势有效性，再按官方消融逐项开启。这里依据官方文档2025-06-19的recipe接口示例；文档说明它没有实现论文中的额外Overlong Filtering，复现时应锁定commit并交代这一边界，不能当作所有新版本永久不支持。

#### 易错点

- “DAPO=GRPO+调高epsilon”漏掉数据选择、loss归一化和奖励路径。

#### 追问

- 启用filter_groups后一直采不够批次，应检查数据难度、奖励还是采样策略？

<a id="aln-030"></a>
### ALN-030 · GRPO 数据必须标注 Thought 吗，完整训练数据与 rollout 怎样组织？

**L2**

#### 答案

GRPO 的训练基本输入是prompt及奖励计算所需的任务元数据，不要求每条样本都提供人工标注的推理过程。数学任务可保存题目与可核验答案，代码任务可保存测试环境，Agent任务可保存初始状态和成功判定。模型对同一prompt采样G条响应，奖励器评分，再以组内相对结果构造优势并更新策略。

若采用cold-start SFT，需要带高质量目标响应的数据，这是另一个阶段；把SFT的带Thought示例直接当成GRPO必备字段，会混淆监督标签与在线rollout。过程奖励可使用步骤标签或验证器，但不是GRPO定义的强制要求。

实现要保留prompt组ID、有效回答mask、终止/截断状态、采样旧log-prob和奖励；同组不能混入其他prompt。优势与reward通常停止梯度，训练需按具体框架区分组标准差、loss聚合和KL位置。G=1通常无法从组内相对结果提供有效学习信号。

#### 易错点

- “GRPO不需要数据标签”过于绝对：可验证奖励常需要答案/测试/环境信息，只是不必人工Thought。

#### 追问

- 模型全部回答错误时，为什么改成更多人工推理文本未必解决GRPO零优势？

<a id="aln-031"></a>
### ALN-031 · GRPO 不收敛或训练奖励升高但能力退化，怎样排查和调参？

**L3**

#### 答案

先确认奖励可信：离线重放正确、错误、格式异常和截断样本，检查答案解析器、测试超时和格式奖励是否被钻空子。再看每组奖励方差与非零优势比例；全对/全错过多时，可能是题目难度、采样熵或group size不合适，而非单纯学习率过低。

检查组ID、response mask、log-prob对齐、detach与loss归一化；在更新前，新旧策略同权重时ratio应接近1，训练/推理引擎精度和模板不一致会破坏这一条件。监控熵、KL、clip fraction、梯度范数、响应长度、截断率与独立验证正确率。长答上升可能只是奖励偏好冗长，训练平均loss也不应期望像SFT一样单调下降。

用能产生正负结果的少量任务先跑通，再单变量调整学习率、每批更新次数、clip、KL、G、采样温度和回答预算；设稳定SFT基线与冻结评测集。若验证持续退化应回滚checkpoint并定位机制，盲目延长训练通常放大奖励偏差。

#### 易错点

- 策略梯度loss接近0可能是零优势、clip饱和或梯度错误，不等于已经收敛。

#### 追问

- 组内奖励有方差但ratio始终1且权重不变，如何定位未更新问题？

<a id="aln-032"></a>
### ALN-032 · 正负样本不对称设计有哪些方式，和 PPO/DAPO 的不对称 clip 有何区别？

**L3**

#### 答案

先确认“正负”的定义：偏好数据是同prompt的chosen/rejected对，二分类数据是正负标签，策略优化通常按优势正负区分。三者不能使用一个未说明的“正样本权重”概念。

样本加权可改变各偏好对或类别对目标的贡献，用于类别不平衡、错误成本不同或标签可信度不同；DPO本身已在同一logistic loss里提高chosen相对rejected的log-ratio间隔。若分别额外拉高/压低两者绝对概率，便改变了目标，可能带来长度偏差、遗忘或过度拒答，不是标准DPO的等价写法。

PPO/DAPO的clip取决于优势符号：正优势只限制过度增加概率，负优势只限制过度减少概率。不对称上下界改变允许更新区间，而非直接给正负样本乘不同常数。选型应说明业务成本、归一化和最终阈值，并报告正负分层的收益与误伤。

$$
\begin{aligned}\mathcal L_{\mathrm{weighted\ BCE}}&=-\mathbb E[w_+y\log p+w_-(1-y)\log(1-p)]\\ \ell_{\mathrm{PPO}}&=\min(\rho A,\operatorname{clip}(\rho,1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})A)\end{aligned}
$$

#### 易错点

- 不能把类别权重、chosen/rejected概率项、正负优势clip三种机制当作同一算法。

#### 追问

- 高风险拒答任务怎样验证提高负类权重没有导致过度拒答？

<a id="topic-5"></a>
## 奖励与对齐策略

<a id="aln-002"></a>
### ALN-002 · 奖励模型如何用成对偏好训练？

**L2**

#### 答案

奖励模型对完整回答输出一个标量。对同一问题 $x$ 的优选回答 $y_w$ 和非优选回答 $y_l$，用分数差的 sigmoid 拟合偏好概率，再最小化二元负对数似然。这里采用 Bradley–Terry 建模假设，不能把它当作所有真实偏好都必然遵循的规律。

成对比较只能约束相对分数：对同一问题的全部回答加上相同常数，偏好概率不变，因此绝对奖励零点无法由这些比较唯一确定，实际使用还需校准。数据与评测应按领域、回答长度和标注者分层，防止模型主要学到“越长越好”或套话风格。

成对比较常比绝对打分更易校准，但仍有平局、不传递、领域差异与标注噪声；它不直接说明质量差距大小。完整排序并非必须做全部O(K²)比较，可用排序/主动选择减少次数。RM常在语言骨干上加标量head，架构和参数规模不必与Actor完全一致；对奖励排序准确率之外，还应测当前策略生成分布上的泛化。

$$
\mathcal{L}_{\mathrm{RM}}=-\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)\right]
$$

#### 易错点

- RM 高准确率不代表策略优化后分布外仍可靠。

#### 追问

- 如何处理平局与偏好不传递？

<a id="aln-015"></a>
### ALN-015 · 如何识别和缓解 reward hacking？

**L2**

#### 答案

Reward hacking 指模型过度优化奖励代理，却降低了真实任务质量。识别时应联合观察独立指标、回答长度、套话和异常工具行为，并对高分样本做人类或对抗复核；只看奖励上涨无法判断能力是否改善。

常见表现包括重复关键词、迎合裁判、空泛安全回答或绕过弱验证器。参考 KL、提前停止和多样反馈可降低风险，但不能证明问题已经消除。工程上应保留 rollout 与分项奖励，聚类分析高分失败案例，再改进评分器和验证流程。

回答模式化、奉承或内容空洞时，构造事实正确但朴素、华丽却无信息等控制样本，检查RM是否主要奖励风格或长度。比较训练RM与独立人工/judge，排查偏好数据窄、优化过量及KL不足；在新偏好数据中加入反例并保留通用能力回归。模式崩溃、谄媚与对齐税相关但不是同义词。

#### 易错点

- GRPO 的相对优势或换成 LLM judge 本身不是 reward hacking 的解药。

#### 追问

- 如果训练奖励上升、独立胜率下降，你先停哪一环？

<a id="aln-016"></a>
### ALN-016 · 后训练为什么会出现对齐税或遗忘？

**L2**

#### 答案

后训练改变目标与数据分布，可能使部分原有能力下降，表现为对齐税或遗忘。评估应在相同设置下比较基础模型、SFT 和偏好优化模型，并分别报告数学、代码、多语言和安全表现，不能只用一个总分判断。

格式、拒答行为或解码变化也可能造成表面退化，应先排除这些评测因素。混合数据、正则化和预训练数据回放可调节能力保留与新目标之间的权衡；InstructGPT 的 PPO-ptx 展示过混合预训练目标的部分收益，但不保证对所有能力都有效。

Agent场景的对齐税可表现为通用任务成功率下降、过度拒绝合法工具、规划变保守或流程成本增加；应分别评价业务效用与安全约束，保持相同工具权限和预算，分析是否来自能力遗忘、奖励偏差或执行策略。安全验收改善而某类能力下降是具体权衡，不能把所有上线质量下降都统称对齐税。

#### 易错点

- LoRA 或小学习率可降低变化幅度，但不能保证没有遗忘。

#### 追问

- 如何区分知识丢失和评测 parser 失配？

<a id="aln-017"></a>
### ALN-017 · RLAIF 和 Constitutional AI 如何工作？

**L2**

#### 答案

RLAIF 用 AI 提供偏好或反馈，减少人工逐样本标注成本。Constitutional AI 由人定义原则，引导模型自我批评与修订，再利用 AI 偏好构造奖励模型和策略训练；其中自我修订可用于 SFT，AI 成对标签可用于偏好强化学习。

这仍需要人类制定原则和审计结果，并非无需人工，也不自动消除偏差。原则、提示词、裁判与训练数据应可追踪并版本化；用专家和独立留出样本检查少数群体、事实质量及过度拒答，避免反馈模型的偏差被反复放大。

#### 易错点

- 模型反馈可能传播同类错误，数据量增加不能替代质量核验。

#### 追问

- 同一个模型生成和打分会产生哪些相关误差？

<a id="aln-020"></a>
### ALN-020 · 多目标奖励发生冲突时如何处理？

**L3**

#### 答案

正确性、简洁性、有用性和安全性可能冲突，应先区分必须满足的约束与可以交换的质量目标，再采用约束优化、门控或加权奖励，并公开各维度的取舍。一般质量可以分析 Pareto 权衡，严重违规则可作为硬约束处理。

加权前需校准奖励的单位、方差和样本分布，避免数值较大的分项无意占主导。保留分项奖励及失败案例，单独检查过度拒答、正确但冗长等冲突情形；一个综合分数无法完整说明多目标表现。

PPO奖励设计会直接改变策略偏好：准确、证据支持、合规、长度与格式代理应分别验证。调权重前用简单正确/错误/套话/超长样本检查每个奖励分量的方向与尺度，再监控分层实际目标。奖励项更多不必然更可靠，冲突目标可能需约束优化或明确优先级。

#### 易错点

- 奖励简单相加不保证满足硬约束，也不保证各项单调改善。

#### 追问

- 如何确定安全阈值并防止模型输出空答案拿高分？

<a id="aln-033"></a>
### ALN-033 · 提升 RAG 回答质量时怎样选 DPO 或 GRPO，奖励怎样设计？

**L3**

#### 答案

DPO适合已有同一问题、同一证据上下文下的可靠chosen/rejected数据，直接优化回答偏好且不需要在线采样循环。GRPO适合能重复采样并得到相对可信任务奖励的场景，例如基于可核验引用/事实/数据库结果的问答；其优势是可探索当前策略，但奖励设计和rollout成本更高。

构造DPO数据时控制检索证据，避免chosen拿到正确文档而rejected没拿到，使优化主要学习输入差异。在线GRPO可把答案正确、证据支持、引用有效、指令约束等分层验证，防止只奖励引用数量或与参考的词面重合。开放域judge本身有偏差，组内归一化不会将坏奖励变成可靠监督。

若失败源于召回漏证据，单独微调回答模型无法替代改检索。实验先冻结索引、检索器、prompt与预算，比较SFT/DPO/GRPO，并在相同检索条件和端到端系统上分别测效果、拒答、成本与分布外泛化。

#### 易错点

- RAG不是天然必须用DPO；GRPO也不因组内比较就免于奖励作弊。

#### 追问

- 怎样区分策略更会回答与它只是更会迎合引用评分器？

## 参考资料

- [Training language models to follow instructions with human feedback](https://arxiv.org/html/2203.02155v1)
- [Direct Preference Optimization](https://arxiv.org/html/2305.18290v3)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Reward Modeling — TRL](https://huggingface.co/docs/trl/reward_trainer)
- [PPO — Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438)
- [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/pdf/2307.04964)
- [DPO Trainer — TRL](https://huggingface.co/docs/trl/dpo_trainer)
- [Is DPO Superior to PPO for LLM Alignment?](https://arxiv.org/abs/2404.10719)
- [Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [UltraFeedback](https://arxiv.org/html/2310.01377v1)
- [DeepSeekMath](https://arxiv.org/html/2402.03300v3)
- [GRPO Trainer — TRL](https://huggingface.co/docs/trl/grpo_trainer)
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/html/2503.20783v2)
- [DeepSeek-R1](https://arxiv.org/html/2501.12948v1)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)
- [Online DPO Trainer — TRL](https://huggingface.co/docs/trl/online_dpo_trainer)
- [Safe RLHF](https://arxiv.org/abs/2310.12773)
- [Spinning Up: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [Spinning Up: Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [verl Hardware Resource Needed for RL](https://verl.readthedocs.io/en/latest/perf/device_tuning.html)
- [Group Sequence Policy Optimization](https://arxiv.org/html/2507.18071v2)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/html/2503.14476v2)
- [verl DAPO recipe](https://verl.readthedocs.io/en/latest/algo/dapo.html)
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [verl GRPO documentation](https://verl.readthedocs.io/en/latest/algo/grpo.html)
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [Spinning Up: Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/abs/1804.08617)
- [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
- [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)
- [Policy invariance under reward transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686)
- [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)
