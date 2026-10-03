# 手撕代码练习

[返回首页](../README.md) · [手撕题与答案](../chapters/14-coding.md)

先口述输入/输出、形状和复杂度，再从空文件完成实现，最后用独立基线和边界验证。

| 文件 | 内容 | 环境 |
|---|---|---|
| [reference.py](reference.py) | 稳定 Softmax/CE、单头 Attention、缓存 mask、RoPE、InfoNCE、采样、LoRA、DPO、GRPO、TopK、岛屿、编辑距离、LRU、KV 估算 | Python 标准库 |
| [torch_primitives.py](torch_primitives.py) | 可训练 MHA、RoPE、LoRA Linear、InfoNCE、回答 token logprob、DPO loss | 可选 PyTorch |
| [two_sum](reference.py#L150) | 两数之和：单遍哈希表、返回不同元素的下标 | Python 标准库 |
| [longest_common_subsequence](reference.py#L161) | 最长公共子序列长度：滚动行动态规划 | Python 标准库 |
| [MLP](torch_primitives.py#L12) / [ImageMLP](torch_primitives.py#L25) | 两层 MLP、逐 token 投影、固定尺寸图像分类 | 可选 PyTorch |
| [BucketBatchSampler](reference.py#L239) | 长度分桶、局部洗牌、epoch 种子与批次索引 | Python 标准库 |
| [vae_reparameterize](torch_primitives.py#L130) / [vae_loss](torch_primitives.py#L145) | 对角高斯重参数化、重建 NLL 与 KL，逐样本求和再 batch 平均 | 可选 PyTorch |
| [integer_sqrt](reference.py#L290) / [longest_palindromic_substring](reference.py#L307) | 整数二分平方根、中心扩展回文子串 | Python 标准库 |
| [unique_permutations](reference.py#L321) | 全排列：回溯恢复现场、重复元素同层去重 | Python 标准库 |
| [ListNode](reference.py#L348) / [reverse_linked_list](reference.py#L353) | 单链表原地反转，修改指针前检查环 | Python 标准库 |
| [max_stock_profit](reference.py#L368) | 至多一次买卖：历史最低价与最佳收益 | Python 标准库 |
| [test_reference.py](test_reference.py) | 26 组边界、不变量和独立算法对照测试 | Python 标准库 |
| [test_torch_primitives.py](test_torch_primitives.py) | 11 组前向、mask、梯度、冻结及数据加载测试 | 无 PyTorch 时明确 skip |

重要约定：

- 布尔 mask 中 `True` 表示允许访问；其他 PyTorch API 可能采用不同约定。
- 标准库 `attention` 是单头数学参考；PyTorch 类实现多头与可学习投影。
- RoPE 使用相邻维配对，不能直接替换某些模型的 split-half 布局。
- GRPO 示例仅计算组内奖励优势，采用总体标准差，不包含 rollout、重要性比率、clip 或 KL 完整训练流程。
- DPO 标量接口接收回答的**序列 log probability**，不是单个 token 概率；标准目标使用求和。
- LRU/采样实现是单进程教学代码；没有线上并发、分布式锁或安全隔离。
- KV 估算不含权重、激活、页表、量化 scale、分配碎片和工作区。
- BucketBatchSampler 是单进程批次采样器，返回索引列表；分布式还需明确 rank 分片和批次数对齐。
- VAE 默认输入 Bernoulli 重建 logits；`unit_gaussian` 分支输入均值、采用固定单位方差。感知损失与对抗训练不属于这个标准 ELBO 示例。

建议练习顺序：Softmax/CE → Attention/mask → RoPE → LoRA/InfoNCE → DPO/GRPO → TopK/岛屿/DP/LRU。
