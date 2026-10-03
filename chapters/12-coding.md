# 手撕代码与算法

[返回仓库首页](../README.md) · [来源目录](../SOURCES.md) · [阅读全部题目](../exports/questions.jsonl)

L1 基础 · L2 进阶 · L3 深入。难度为编辑判断；社区线索不等于雇主确认真题。所有答案为独立整理。

## 题目导航

- [COD-001 · 手写稳定 Softmax 与交叉熵，为什么要减最大值？](#cod-001)
- [COD-002 · 手撕 Multi-Head Attention：形状、缩放与 mask 怎么写？](#cod-002)
- [COD-003 · KV Cache 增量解码的因果 mask 为什么容易写错？](#cod-003)
- [COD-004 · 手写 RoPE，并证明旋转保持范数与相对位置内积。](#cod-004)
- [COD-005 · 手写 InfoNCE：正样本标签与 in-batch negatives 如何组织？](#cod-005)
- [COD-006 · 实现 top-k / top-p 采样，截断边界怎么处理？](#cod-006)
- [COD-007 · 实现 LoRA Linear 并证明 merge 前后输出一致。](#cod-007)
- [COD-008 · 实现 DPO loss，怎样避免符号和序列概率错误？](#cod-008)
- [COD-009 · 实现 GRPO 组内优势，标准差为零时怎么办？](#cod-009)
- [COD-010 · 数组第 k 大：堆与 Quickselect 怎样取舍？](#cod-010)
- [COD-011 · 岛屿问题：DFS/BFS 的时间、空间与边界。](#cod-011)
- [COD-012 · 手写编辑距离，并压缩到 O(min(m,n)) 空间。](#cod-012)
- [COD-013 · 实现 O(1) 的 LRU Cache，更新已有 key 怎么处理？](#cod-013)
- [COD-014 · 手算并编码 MHA/GQA 的 KV Cache 显存。](#cod-014)
- [COD-015 · 手撕代码时怎样设计能揭露错误的测试？](#cod-015)

<a id="cod-001"></a>
## COD-001 · 手写稳定 Softmax 与交叉熵，为什么要减最大值？

**L1 · 编辑补充题** · 标签：数值稳定 / log-sum-exp

**30 秒回答**

Softmax 对所有 logits 加同一常数不变，因此先减最大值可避免指数溢出。交叉熵最好直接用 log-sum-exp 减目标 logit，避免先求概率再取对数导致极小概率下溢；全被 mask 的行应显式处理。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 证明 exp(z_i-c)/Σexp(z_j-c) 与原概率相同；取 c=max(z)。
- CE=logsumexp(z)-z_target，梯度为 p-one_hot(target)。
- 测试大正负 logits、平移不变性、含 -inf 的部分 mask；全 -inf 没有有效分布。

### 公式

```text
p_i=exp(z_i-m)/Σ_j exp(z_j-m); CE=m+log Σ_j exp(z_j-m)-z_y
```

### 易错点

- 只写 exp(z)/sum(exp(z))；或者用 log(softmax) 在极端值下得到 log(0)。

### 面试官可能追问

- 如何实现 label smoothing 和 ignore_index？

</details>

**技术依据**

- [ENG-P02 · PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="cod-002"></a>
## COD-002 · 手撕 Multi-Head Attention：形状、缩放与 mask 怎么写？

**L1 · 社区题目线索** · 标签：MHA / shape / mask

**30 秒回答**

输入 [B,T,D] 投影 Q/K/V 后拆为 [B,H,T,d]，计算 QKᵀ/√d，加因果或 padding mask，再做 softmax 与 V 相乘，拼接头并投影。讲清每步形状，并用因果不泄漏与输出维度测试检查实现。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 要求 D=H*d；线性权重可合并成一个 D→3D 投影，但逻辑上有三组参数。
- scores 形状 [B,H,Tq,Tk]，softmax 沿 Tk 轴。
- 教学实现可显式生成 T² 分数；生产 SDPA 可能使用融合内核。训练 dropout 与 eval 设置要区分。

### 公式

```text
Attention(Q,K,V)=softmax(QK^T/√d + mask)V
```

### 易错点

- 把布尔 mask 语义弄反；PyTorch 不同 API 的 True 可能表示允许或屏蔽。

### 面试官可能追问

- 加入 GQA 后 Q 头数与 KV 头数如何对应？

</details>

**技术依据**

- [ENG-P01 · PyTorch scaled_dot_product_attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

**题目出处线索**

- [ENG-C01 · 字节大模型算法实习生-电商业务（已oc）](https://www.nowcoder.com/discuss/724319940982898688) · `reported_question`：公开正文列出 multi-head-attention 手撕题；本题为重述与独立答案。

<a id="cod-003"></a>
## COD-003 · KV Cache 增量解码的因果 mask 为什么容易写错？

**L2 · 编辑补充题** · 标签：KV Cache / causal mask / offset

**30 秒回答**

有缓存时新 query 的绝对位置从 past_len 开始，允许访问所有较早 key。若 Tq=1、Tk 很长时直接套左上三角 mask，会只允许第一个 key。应使用绝对位置比较，或核对框架针对非方阵 mask 的定义。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 正确允许关系为 key_pos<=query_pos，query_pos=past_len+i。
- prefill 与逐 token decode 的同一位置输出应在 eval、同精度下近似一致。
- 还需组合 padding mask；缓存的 RoPE 位置与 query 位置必须对齐。

### 公式

```text
allowed[i,j] = (j <= past_len+i)
```

### 易错点

- 认为 is_causal=True 对所有非方阵 attention 都自动等价于缓存解码。

### 面试官可能追问

- 滑动窗口缓存淘汰后位置编号如何维护？

</details>

**技术依据**

- [ENG-P01 · PyTorch scaled_dot_product_attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="cod-004"></a>
## COD-004 · 手写 RoPE，并证明旋转保持范数与相对位置内积。

**L2 · 编辑补充题** · 标签：RoPE / 旋转 / 相对位置

**30 秒回答**

将偶数维向量按二维对旋转，每对使用不同频率。旋转矩阵正交，所以不改变范数；Q、K 在不同位置旋转后，内积只额外依赖位置差。实现时要匹配模型采用的维度配对方式、频率及位置编号。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 每对 (a,b) 变成 (a cosφ-b sinφ, a sinφ+b cosφ)。
- R_mᵀR_n=R_(n-m)，因此 (R_mq)ᵀ(R_nk)=qᵀR_(n-m)k。
- 相邻配对与 split-half 配对可经维度排列对应；不能直接混用已有权重。

### 公式

```text
φ_i=m*base^(-2i/d); ||R_mx||=||x||
```

### 易错点

- 把 RoPE 当作加到 embedding 的位置向量，或对 V 也默认旋转。

### 面试官可能追问

- 频率缩放改变了哪个项？为什么公式可外推不代表效果必然可外推？

</details>

**技术依据**

- [ENG-P04 · RoFormer / RoPE](https://arxiv.org/abs/2104.09864)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="cod-005"></a>
## COD-005 · 手写 InfoNCE：正样本标签与 in-batch negatives 如何组织？

**L2 · 社区题目线索** · 标签：InfoNCE / 对比学习 / false negative

**30 秒回答**

将匹配的 query/key 排在相同 batch 索引，归一化后求两两相似度除以温度，以对角索引做交叉熵标签。负样本来自其他列；若做双向图文训练，可平均两个方向的损失，注意重复样本造成假负例。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- logits 形状 [N,N]，labels=arange(N)；单向损失平均每行 CE。
- 温度必须大于零；更低温度使分布更尖，梯度与难负例影响同时变化。
- 分布式 all-gather 要保证标签 offset 与可回传梯度的实现对应。

### 公式

```text
L=-(1/N)Σ_i log[exp(s(q_i,k_i)/τ)/Σ_j exp(s(q_i,k_j)/τ)]
```

### 易错点

- 把一批同类别的另一个正例也当负例；或打乱 keys 后仍用对角标签。

### 面试官可能追问

- 当 N=1 时还有对比信号吗？

</details>

**技术依据**

- [ENG-P03 · Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- [ENG-P02 · PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)

**题目出处线索**

- [ENG-C01 · 字节大模型算法实习生-电商业务（已oc）](https://www.nowcoder.com/discuss/724319940982898688) · `reported_question`：正文明确要求理解并编码 InfoNCE；答案为独立实现。

<a id="cod-006"></a>
## COD-006 · 实现 top-k / top-p 采样，截断边界怎么处理？

**L2 · 编辑补充题** · 标签：top-p / top-k / sampling

**30 秒回答**

先确定温度和过滤顺序，对概率排序并保留累计质量首次达到 p 的最小前缀，必须包括越过阈值的那个 token，然后重新归一化采样。top-k 是固定候选数；温度零可约定为独立的贪心路径。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 本仓库示例约定 temperature→top-k→归一化→top-p→再归一化，其他实现应说明顺序。
- 例如概率 [0.6,0.3,0.1]、p=0.7，应保留前两项。
- 用固定 RNG 做可复现实验；排序并列值、极小温度和非法参数要有约定。

### 易错点

- 用 cumsum<=p 直接过滤，误删达到阈值的 token，甚至删空。

### 面试官可能追问

- top-k 与 top-p 的组合为什么顺序可能改变候选集？

</details>

**技术依据**

- [ENG-P12 · Transformers GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="cod-007"></a>
## COD-007 · 实现 LoRA Linear 并证明 merge 前后输出一致。

**L2 · 编辑补充题** · 标签：LoRA / 矩阵乘法 / merge

**30 秒回答**

冻结 W，增加 ΔW=(α/r)BA；对行 batch 输入采用 xWᵀ+(α/r)(xAᵀ)Bᵀ。若 dropout 关闭且权重精度一致，合并成 W+ΔW 与未合并前向应一致。实现时要说明 A/B 的维度及初始化。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- W[out,in]、A[r,in]、B[out,r]，新增参数 r(in+out)。
- 常见初始化 A 随机、B 零使初始输出不变；首步 A 梯度为零但 B 可学习。
- 验证冻结参数无梯度，merge 前后误差在精度容限内；量化基座合并需额外处理。

### 易错点

- 把低秩更新写成元素乘法，或 merge 后仍重复加 adapter。

### 面试官可能追问

- B 随机、A 零是否也能让初始更新为零？

</details>

**技术依据**

- [ENG-P05 · LoRA](https://arxiv.org/abs/2106.09685)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="cod-008"></a>
## COD-008 · 实现 DPO loss，怎样避免符号和序列概率错误？

**L2 · 编辑补充题** · 标签：DPO / logprob / loss

**30 秒回答**

对同一 prompt 的 chosen/rejected 计算回答 token 的序列 log probability，减去冻结 reference 对应值，再做 chosen 相对 rejected 的差，乘 β，取负 logsigmoid。用正负 margin 与极端值测试最容易发现符号错。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 标准 DPO 用序列 logprob 求和；长度平均属于改变目标的其他选择。
- prompt 与 padding token 不进入回答 logprob 求和；reference 不反传。
- 用 softplus(-z) 或 -logsigmoid(z) 避免直接 sigmoid 后取 log 的溢出。

### 公式

```text
z=β[(logπθ(y+)-logπref(y+))-(logπθ(y-)-logπref(y-))]; L=softplus(-z)
```

### 易错点

- chosen margin 变好时 loss 却变大；或者漏掉 reference 项。

### 面试官可能追问

- reference 与 policy 初始相同，loss 是多少？

</details>

**技术依据**

- [ENG-P13 · TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="cod-009"></a>
## COD-009 · 实现 GRPO 组内优势，标准差为零时怎么办？

**L2 · 编辑补充题** · 标签：GRPO / advantage / std

**30 秒回答**

对同一 prompt 的一组奖励减组均值，并按明确约定的组内标准差归一化，加 ε 保持数值稳定。若组内奖励全相同，奖励优势为零，该组没有相对奖励梯度；KL 等独立项仍可能产生梯度。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 本示例使用总体标准差 correction=0；框架可能不同，要确认配置。
- G=1 或全部同奖励时都缺乏组内对比信号。
- 记录零方差比例，考虑采样多样性与奖励分辨率；是否去标准差归一化应按具体算法说明。

### 公式

```text
A_i=(r_i-mean(r))/(std(r)+ε)
```

### 易错点

- 说“优势零但 PPO clip 仍必然压缩熵”；clip 不是独立的梯度来源。

### 面试官可能追问

- 奖励缩放和不同难度 prompt 会怎样影响更新？

</details>

**技术依据**

- [ENG-P14 · TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="cod-010"></a>
## COD-010 · 数组第 k 大：堆与 Quickselect 怎样取舍？

**L1 · 社区题目线索** · 标签：TopK / heap / Quickselect

**30 秒回答**

维护大小 k 的小根堆，堆顶就是已读元素的第 k 大，时间 O(n log k)、空间 O(k)，适合流式输入。Quickselect 平均 O(n)、通常原地，但坏 pivot 可退化到 O(n²)，要说明随机化与重复元素处理。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 先说明第 k 大按元素计数，重复值并不去重。
- 堆满后仅当新元素大于堆顶时替换；k 要满足 1<=k<=n。
- Quickselect 可三路 partition 处理大量相等值；不能仅说最坏 O(n)。

### 易错点

- 混淆第 k 大与第 k 个不同的大值。

### 面试官可能追问

- 如果数据不能全部载入内存，如何改？

</details>

**技术依据**

- [ENG-P06 · Python heapq](https://docs.python.org/3/library/heapq.html)
- [ENG-P09 · LeetCode Kth Largest Element](https://leetcode.com/problems/kth-largest-element-in-an-array/)

**题目出处线索**

- [ENG-C01 · 字节大模型算法实习生-电商业务（已oc）](https://www.nowcoder.com/discuss/724319940982898688) · `reported_question`：正文手撕题列出数组第 k 大，并提到堆或快排。

<a id="cod-011"></a>
## COD-011 · 岛屿问题：DFS/BFS 的时间、空间与边界。

**L1 · 社区题目线索** · 标签：图遍历 / DFS / BFS

**30 秒回答**

扫描每个陆地格，对未访问陆地做一次四邻域遍历并累计连通块。每格最多入队或入栈一次，时间 O(RC)；visited 与栈最坏 O(RC)。Python 用迭代栈可避免大岛导致递归深度溢出。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 入栈时标记 visited，可避免重复入栈。
- 先约定输入是整数 0/1 还是字符，是否允许原地修改。
- 测试空矩阵、全海、全陆、对角接触与细长蛇形岛。

### 易错点

- 无意中把对角接触也视为连通，或遗漏边界检查。

### 面试官可能追问

- 若陆地不断增加，如何用并查集增量维护数量？

</details>

**技术依据**

- [ENG-P08 · LeetCode Number of Islands](https://leetcode.com/problems/number-of-islands/)

**题目出处线索**

- [ENG-C01 · 字节大模型算法实习生-电商业务（已oc）](https://www.nowcoder.com/discuss/724319940982898688) · `reported_question`：面经手撕题列出岛屿问题；本题按四连通计数整理。

<a id="cod-012"></a>
## COD-012 · 手写编辑距离，并压缩到 O(min(m,n)) 空间。

**L1 · 社区题目线索** · 标签：DP / 编辑距离

**30 秒回答**

设 dp[i][j] 是两个前缀的编辑距离，由删除、插入、替换三种最后操作转移；字符相同则替换成本零。初始化空串行列为长度，滚动两行可把空间压到较短字符串长度，时间仍是 O(mn)。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+[a≠b])。
- 令第二维是短字符串，滚动 previous/current。
- 压缩后若要恢复具体编辑路径，需保留更多信息或使用分治方法。

### 易错点

- 把最长公共子序列的转移误套到允许替换的编辑距离。

### 面试官可能追问

- 语音识别 WER 里的替换、插入、删除如何关联？

</details>

**技术依据**

- [ENG-P10 · LeetCode Edit Distance](https://leetcode.com/problems/edit-distance/)

**题目出处线索**

- [ENG-C07 · 百度大模型算法岗面经-05（搜索可见）](https://www.nowcoder.com/discuss/927018708290015232) · `search_snippet`：搜索摘录列出编辑距离代码题；未声称读到完整原始面试。

<a id="cod-013"></a>
## COD-013 · 实现 O(1) 的 LRU Cache，更新已有 key 怎么处理？

**L1 · 编辑补充题** · 标签：LRU / 链表 / 哈希

**30 秒回答**

哈希表定位节点，双向链表维护访问顺序；get 命中和 put 更新都移动到最近使用端，超容量淘汰另一端。Python 可用 OrderedDict 教学实现；需要解释它对应的顺序维护语义，不能把普通缓存等同于 LRU。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 典型 get/put 平均 O(1)，hash 冲突与并发是另外的约束。
- 已有 key 更新不增加元素数量，但必须刷新访问位置。
- 测试 capacity=0、重复更新、读取导致淘汰顺序改变。

### 易错点

- 只在插入时调整顺序，导致实现的是 FIFO。

### 面试官可能追问

- 并发服务如何避免 cache stampede 与跨租户键碰撞？

</details>

**技术依据**

- [ENG-P07 · Python collections / OrderedDict](https://docs.python.org/3/library/collections.html)
- [ENG-P11 · LeetCode LRU Cache](https://leetcode.com/problems/lru-cache/)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="cod-014"></a>
## COD-014 · 手算并编码 MHA/GQA 的 KV Cache 显存。

**L2 · 社区题目线索** · 标签：显存 / GQA / 单位

**30 秒回答**

忽略页表、分配碎片和量化 scale，KV 存储字节为 2BLTh_kv d_h s，2 表示 K 与 V，s 是每元素字节。GQA 应代入 KV 头数。结果是缓存本身，不包括模型权重、激活与运行时工作区。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- L=层数、B=batch、T=缓存长度、h_kv=KV heads、d_h=头维。
- B=1,T=4096,L=32,h_kv=8,d_h=128,s=2 得 512 MiB。
- 可变长 batch 更准确的写法用 Σ_b T_b；分页按已分配块估算。

### 公式

```text
bytes=2*B*T*L*h_kv*d_h*s
```

### 易错点

- 用 Q 头数计算 GQA，或把 GB 与 GiB 混用。

### 面试官可能追问

- 同样显存预算下，把上下文翻倍会如何影响并发？

</details>

**技术依据**

- [ENG-P15 · vLLM Metrics](https://docs.vllm.ai/en/latest/design/metrics/)
- [ENG-P29 · vLLM Paged Attention](https://docs.vllm.ai/en/latest/design/paged_attention/)

**题目出处线索**

- [ENG-C02 · 字节跳动大模型算法岗面经-04（公开部分）](https://www.nowcoder.com/discuss/922308546966847488) · `reported_topic`：公开部分有 KV Cache 和显存开销主题；本题补充可运行估算练习。

<a id="cod-015"></a>
## COD-015 · 手撕代码时怎样设计能揭露错误的测试？

**L2 · 编辑补充题** · 标签：property test / 测试 / 边界

**30 秒回答**

先确认输入输出约定，再用边界样例、独立朴素解和数学不变量验证。模型算子重点测 shape、mask、数值稳定与梯度；算法题可在小规模随机输入上对照独立解。测试应检验性质，不能只复刻自己的实现。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- Attention 测未来 token 改变不影响早期输出；cache decode 与 prefill 对照。
- RoPE 测范数与相对内积；LoRA 测 merge 等价；TopK 与排序基线对照。
- 随机种子固定方便重现，但 GPU 完全确定性还依赖设备、内核与版本。

### 易错点

- 只有一个常规样例，或把同一个错误公式写进 expected。

### 面试官可能追问

- 浮点比较为什么应采用相对/绝对容差？

</details>

**技术依据**

- [ENG-P21 · PyTorch Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。
