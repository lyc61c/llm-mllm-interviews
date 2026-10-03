# 基础分片的同伴审阅

审阅日期：2026-10-02。审阅对象为 `interviews/data/multimodal.json`、`interviews/coding/reference.py` 与 `interviews/coding/torch_primitives.py`；本轮没有直接改动这些文件。

## multimodal.json

已逐题查看 VLM-001–025、OMM-001–015、DST-001–020 的 quick/detail、公式和来源范围。未发现阻断性的数学或机制错误。以下易错内容的假设已明确：

- CLIP 使用配对图文的双向交叉熵；归一化相似度与 temperature 的作用区分正确。
- tIoU 使用时间区间交集/并集；CTC 先合并连续重复标签、再移除 blank，不能颠倒。
- FP16 参数/梯度加 FP32 主权重与 Adam 状态的 16 bytes/参数估算是特定实现假设；ZeRO 各阶段只分片其管理的状态，公式没有包含激活与通信缓冲。
- 张量并行的列切分/行切分次序与求和正确；GPipe 气泡比例 `(p−1)/(m+p−1)` 有理想化前提；ring all-reduce 的单 rank 发送量 `2(D−1)S/D` 不能当作集群总通信量。
- MFU 的 `6NT` 近似已说明稠密模型和忽略项，不把所有多模态 encoder 计算直接套入该式。

对模型特定架构另读原始资料核验：

- [Qwen2-VL 第 2.1 节](https://arxiv.org/html/2409.12191v2)：2×2 token 合并、M-RoPE 的时间/高/宽分量，以及 3D 卷积的两帧深度，与题库一致。只核验相关章节，不声称通读全文。
- [Qwen2.5-VL 第 2.1 节](https://arxiv.org/html/2502.13923v1)：窗口 attention 配合 4 个全局 attention 层、动态 FPS 与时间位置编码的说明，与题库一致；没有将这些设置写成所有 MLLM 的通用架构。

## coding/reference.py

逐个静态检查标准库参考实现。scaled dot-product 的缩放与 mask、相邻维度 RoPE、LoRA 矩阵形状、DPO 的 log-ratio 差、population-std 版本的组内 advantage、交叉熵的 logsumexp、top-k/top-p 保留越过阈值的 token 与重归一化均一致。边界实现拒绝非法温度、全屏蔽 attention 行及无穷值输入，代码注释明确其教学范围。

实际运行 `python -m unittest discover -s interviews/coding -p 'test_*.py' -v`：20 个测试中，14 个标准库测试通过；6 个可选 PyTorch 测试因当前环境没有安装 PyTorch 而跳过。跳过不能当作 PyTorch 运行验证。

## coding/torch_primitives.py

静态核对 MHA 的 `[B,T,3,H,Dh]` 拆分、causal mask、FP32 score/softmax，LoRA 的冻结底座/零初始化 B/合并权重，配对 InfoNCE，response log-prob 的单次 label shift，以及 DPO reference detach。未发现公式或梯度方向错误。

### C-01 [P2] RoPE 的 positions 长度应拒绝广播

原实现只检查偶数维度和正 base，没有核对注释声明的 `positions [T]`。当 `x` 的序列长度大于 1，而 `positions=torch.tensor([0])` 时，角度可以广播到所有 token，悄悄使全部 token 使用位置 0。建议检查 `positions.ndim == 1` 和 `positions.numel() == x.shape[-2]`，非法输入抛出 ValueError。

本建议已交给总装者处理；再次静态读取时，`rotary_adjacent` 已加入长度检查，并同时补充 x 的 rank 与非空偶数维度检查。未运行 PyTorch，因此这里记录为“已静态确认修复”，不声称动态测试通过。
