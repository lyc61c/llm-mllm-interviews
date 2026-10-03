# 手撕代码练习

[返回首页](../README.md) · [手撕题与答案](../chapters/12-coding.md)

先口述输入/输出、形状和复杂度，再从空文件完成实现，最后用独立基线和边界验证。

| 文件 | 内容 | 环境 |
|---|---|---|
| [reference.py](reference.py) | 稳定 Softmax/CE、单头 Attention、缓存 mask、RoPE、InfoNCE、采样、LoRA、DPO、GRPO、TopK、岛屿、编辑距离、LRU、KV 估算 | Python 标准库 |
| [torch_primitives.py](torch_primitives.py) | 可训练 MHA、RoPE、LoRA Linear、InfoNCE、回答 token logprob、DPO loss | 可选 PyTorch |
| [test_reference.py](test_reference.py) | 14 组边界、不变量和独立算法对照测试 | Python 标准库 |
| [test_torch_primitives.py](test_torch_primitives.py) | 6 组前向、mask、梯度和冻结测试 | 无 PyTorch 时明确 skip |

在 `interviews/` 下运行：

```bash
python -m unittest discover -s coding -p "test_*.py" -v
```

本次环境未安装 PyTorch：标准库测试实际执行通过；PyTorch 实现已做语法检查与人工审阅，运行测试未验证，会显示 6 个 skip。安装自己的兼容 PyTorch 环境后可执行同一命令验证，无需 GPU。

重要约定：

- 布尔 mask 中 `True` 表示允许访问；其他 PyTorch API 可能采用不同约定。
- 标准库 `attention` 是单头数学参考；PyTorch 类实现多头与可学习投影。
- RoPE 使用相邻维配对，不能直接替换某些模型的 split-half 布局。
- GRPO 示例仅计算组内奖励优势，采用总体标准差，不包含 rollout、重要性比率、clip 或 KL 完整训练流程。
- DPO 标量接口接收回答的**序列 log probability**，不是单个 token 概率；标准目标使用求和。
- LRU/采样实现是单进程教学代码；没有线上并发、分布式锁或安全隔离。
- KV 估算不含权重、激活、页表、量化 scale、分配碎片和工作区。

建议练习顺序：Softmax/CE → Attention/mask → RoPE → LoRA/InfoNCE → DPO/GRPO → TopK/岛屿/DP/LRU。
