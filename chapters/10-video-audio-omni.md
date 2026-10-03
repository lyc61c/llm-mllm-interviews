# 视频、语音与 Omni

[返回仓库首页](../README.md) · [来源目录](../SOURCES.md) · [阅读全部题目](../exports/questions.jsonl)

L1 基础 · L2 进阶 · L3 深入。难度为编辑判断；社区线索不等于雇主确认真题。所有答案为独立整理。

## 题目导航

- [OMM-001 · 视频 VLM 为什么不能简单无限堆叠视频帧？](#omm-001)
- [OMM-002 · 视频时间位置编码为什么要考虑真实时间与 FPS？](#omm-002)
- [OMM-003 · 长视频理解怎样压缩视觉 token，又有什么代价？](#omm-003)
- [OMM-004 · 视频 moment retrieval 与视频问答如何评估？](#omm-004)
- [OMM-005 · Whisper 的输入、架构和多任务接口是什么？](#omm-005)
- [OMM-006 · WER 怎样计算？为什么中文 ASR 常同时报 CER？](#omm-006)
- [OMM-007 · CTC 与自回归语音解码有什么区别？](#omm-007)
- [OMM-008 · 音频连续特征与离散 codec token 分别适合什么？](#omm-008)
- [OMM-009 · VALL-E 式零样本 TTS 为什么能利用短语音提示？](#omm-009)
- [OMM-010 · Qwen2-Audio 这类音频理解模型为什么不等于 ASR？](#omm-010)
- [OMM-011 · Qwen2.5-Omni 的 Thinker-Talker 如何协作？](#omm-011)
- [OMM-012 · ASR→LLM→TTS 流水线与端到端语音对话怎样选？](#omm-012)
- [OMM-013 · 流式语音系统的端到端延迟应怎样拆解？](#omm-013)
- [OMM-014 · 为什么唇读等视觉信息可以帮助噪声下的 ASR？](#omm-014)
- [OMM-015 · Omni 模型怎样评估是否真的融合了声音与视觉？](#omm-015)

<a id="omm-001"></a>
## OMM-001 · 视频 VLM 为什么不能简单无限堆叠视频帧？

**L2 · 编辑补充题** · 标签：视频 / 抽帧

**30 秒回答**

多帧输入增加视觉 token 与语言上下文开销，连续帧又常含大量重复信息。抽帧需要兼顾事件覆盖和细节，应根据任务比较均匀采样、变化采样或分段检索，并保留时间戳；盲目增帧可能既更慢又遗漏短事件。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 按帧数、每帧 token 与文本长度估算整体序列。
- 均匀抽帧覆盖时间范围，关键帧策略可能偏向视觉变化。
- 图像视频联合训练需处理表示和输入协议的一致性。

### 易错点

- 抽帧序号不能代替真实时间，尤其对可变帧率视频。

### 面试官可能追问

- 一分钟短视频和两小时视频如何设不同预算？

</details>

**技术依据**

- [MM-S031 · Video-LLaVA: Learning United Visual Representation by Alignment Before Projection](https://arxiv.org/abs/2311.10122)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="omm-002"></a>
## OMM-002 · 视频时间位置编码为什么要考虑真实时间与 FPS？

**L2 · 编辑补充题** · 标签：视频 / 时间戳

**30 秒回答**

相同帧序号在不同采样率下代表不同真实时刻，只用帧号会混淆速度和事件间隔。Qwen2.5-VL 将时间位置与绝对时间关联并采用动态 FPS 采样，增强时间表达；仍需训练定位任务与一致的时间单位。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 输入要记录原视频时刻，而非只记录取出的第几帧。
- 动态 FPS 可使模型看到不同采样密度的数据。
- 定位评测应检查时间范围、误差与短事件召回。

### 易错点

- 时间编码改进不能保证模型自动精确定位所有事件。

### 面试官可能追问

- 视频被倍速播放时哪些标签需要调整？

</details>

**技术依据**

- [MM-S019 · Qwen2.5-VL Technical Report](https://arxiv.org/html/2502.13923v1)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="omm-003"></a>
## OMM-003 · 长视频理解怎样压缩视觉 token，又有什么代价？

**L3 · 编辑补充题** · 标签：视频压缩 / 时空建模

**30 秒回答**

可用空间池化、时空连接器、分段摘要或检索减少长视频输入。压缩降低成本，却可能抹掉短事件、细节和事件顺序，因此要按任务保留局部证据，并比较压缩前后定位与问答效果，不能只看最终文本是否流畅。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- VideoLLaMA 2 的时空连接器强调空间与时间联合建模。
- 分段处理应携带时间边界，避免摘要覆盖真实证据。
- 细粒度动作题需要比主题总结题更高的保真预算。

### 易错点

- 平均所有帧特征会丢失部分顺序信息。

### 面试官可能追问

- 怎样设计时间顺序交换的消融测试？

</details>

**技术依据**

- [MM-S032 · VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs](https://arxiv.org/abs/2406.07476)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="omm-004"></a>
## OMM-004 · 视频 moment retrieval 与视频问答如何评估？

**L2 · 编辑补充题** · 标签：视频定位 / tIoU

**30 秒回答**

视频问答通常评估答案，而 moment retrieval 还要输出与语言查询对应的起止时刻。可用时间区间交并比和阈值召回评价定位，且与显著性评分分开报告；答案正确并不说明模型找到正确片段。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- Moment-DETR 把片段坐标与显著性作为不同预测目标。
- tIoU 比较预测区间与真实区间的重叠。
- 按短片段、重复动作和多个正确片段分别检查。

### 公式

```text
I=max(0,min(e_p,e_g)−max(s_p,s_g))；tIoU=I/[(e_p−s_p)+(e_g−s_g)−I]，适用于两个非零时长区间。
```

### 易错点

- 不要把时间 IoU 当作逐帧分类准确率。

### 面试官可能追问

- 有多个有效片段时怎样定义召回？

</details>

**技术依据**

- [MM-S033 · QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries](https://arxiv.org/abs/2107.09609)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="omm-005"></a>
## OMM-005 · Whisper 的输入、架构和多任务接口是什么？

**L1 · 社区题目线索** · 标签：Whisper / ASR

**30 秒回答**

Whisper 将音频转成 log-Mel 频谱，经编码器提取表示，再由自回归文本解码器输出识别、翻译或时间戳等 token。原始模型以三十秒音频段训练，长音频需要分窗与时间对齐，不能把它当无延迟流式识别器。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 任务与语言由特殊 token 指定。
- 跨窗口解码应处理重复、遗漏与上下文传播。
- 静音、音乐和噪声样本需独立评估错误转写。

### 易错点

- 转写保留源语言，翻译与转写并非同一任务。

### 面试官可能追问

- 分窗边界落在词中间时怎样处理？

</details>

**技术依据**

- [MM-S034 · Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/html/2212.04356v1)

**题目出处线索**

- [MM-S005 · AI-Engineer-Interview-Questions / Multimodal](https://github.com/ombharatiya/AI-Engineer-Interview-Questions/blob/main/10-multimodal/README.md) · `reported_topic`：社区备考集合明确讨论 Whisper 的 ASR 设计；不是公司面试认证。

<a id="omm-006"></a>
## OMM-006 · WER 怎样计算？为什么中文 ASR 常同时报 CER？

**L1 · 社区题目线索** · 标签：WER / CER / ASR评估

**30 秒回答**

WER 先在词级对齐预测与参考，用替换、删除、插入总数除以参考词数。中文分词规则会影响词级指标，CER 在字符级计算更便于比较；仍须统一标点、数字和大小写规范，并报告业务关键实体的错误率。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 参考词数等于正确、替换与删除的数量之和。
- corpus WER 应先汇总错误和参考词数，再相除。
- 插入可使 WER 超过百分之百，空参考要明确协议。

### 公式

```text
WER = (S + D + I) / N_ref；CER 使用字符级对齐与参考字符数。
```

### 易错点

- 不要无条件平均每句 WER，短句会被过度加权。

### 面试官可能追问

- 数字格式不同但含义相同如何归一化？

</details>

**技术依据**

- [MM-S035 · Hugging Face Evaluate WER 官方实现](https://huggingface.co/spaces/evaluate-metric/wer/blob/main/wer.py)

**题目出处线索**

- [MM-S005 · AI-Engineer-Interview-Questions / Multimodal](https://github.com/ombharatiya/AI-Engineer-Interview-Questions/blob/main/10-multimodal/README.md) · `reported_topic`：社区 ASR 主题列出 WER；此题扩展指标计算和中文协议。

<a id="omm-007"></a>
## OMM-007 · CTC 与自回归语音解码有什么区别？

**L2 · 编辑补充题** · 标签：CTC / ASR

**30 秒回答**

CTC 对所有能折叠为目标文本的单调对齐路径求和，常以 blank 连接音频帧与文本，不显式逐步条件于已生成文本；自回归解码逐 token 依赖历史输出。两者在并行性、语言建模和对齐假设上有取舍。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- CTC 对齐通常要求输入时间步足够覆盖目标及重复符号。
- CTC 可配合外部语言模型解码，不能说完全没有语言信息。
- 自回归模型适合多任务接口，但存在串行解码成本。

### 公式

```text
L_CTC = -log Σ_{π:B(π)=y} Π_t p(π_t|x)，B 合并重复并移除 blank。
```

### 易错点

- 相邻重复标签折叠与 blank 的处理顺序要明确。

### 面试官可能追问

- 为什么重复字符会增加 CTC 的最小输入长度？

</details>

**技术依据**

- [MM-S036 · PyTorch CTCLoss 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CTCLoss.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="omm-008"></a>
## OMM-008 · 音频连续特征与离散 codec token 分别适合什么？

**L2 · 社区题目线索** · 标签：EnCodec / 音频表示

**30 秒回答**

连续音频特征常用于理解与对齐，离散 codec token 则把波形压缩成可生成的符号序列，适合语音语言模型。两者不等价：codec 强调重建质量，理解表征强调语义与任务信息，还要比较帧率、码本数量和时延。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 残差向量量化逐层编码剩余误差，形成多个码本。
- 增加码本可提升重建细节，也增加生成预测负担。
- 评估应同时考虑可懂度、音色、失真与 token 预算。

### 易错点

- 离散音频 token 不是字符 token，也不天然只保留语义。

### 面试官可能追问

- 为什么低比特率 codec 可能损害细节理解？

</details>

**技术依据**

- [MM-S037 · High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)

**题目出处线索**

- [MM-S005 · AI-Engineer-Interview-Questions / Multimodal](https://github.com/ombharatiya/AI-Engineer-Interview-Questions/blob/main/10-multimodal/README.md) · `reported_topic`：社区多模态备考材料讨论神经 codec 表示。

<a id="omm-009"></a>
## OMM-009 · VALL-E 式零样本 TTS 为什么能利用短语音提示？

**L2 · 社区题目线索** · 标签：VALL-E / TTS

**30 秒回答**

VALL-E 把文本到语音建模为声学 codec token 生成，短参考语音提供说话人和声学条件，因此可在不为该说话人单独微调的情况下合成语音。效果仍依赖参考质量、语言覆盖和模型数据分布，需要分别评估内容与音色。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 文本约束要说什么，音频提示提供说话人特征。
- 多个 codec 层需采用适合时延与质量的生成安排。
- 评测关注可懂度、说话人相似度与鲁棒性。

### 易错点

- 零样本不表示训练集中不存在类似声学条件。

### 面试官可能追问

- 嘈杂提示会如何影响合成输出？

</details>

**技术依据**

- [MM-S038 · Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111)

**题目出处线索**

- [MM-S005 · AI-Engineer-Interview-Questions / Multimodal](https://github.com/ombharatiya/AI-Engineer-Interview-Questions/blob/main/10-multimodal/README.md) · `reported_topic`：社区集合提及 VALL-E 风格的 codec 语言模型 TTS。

<a id="omm-010"></a>
## OMM-010 · Qwen2-Audio 这类音频理解模型为什么不等于 ASR？

**L2 · 编辑补充题** · 标签：Qwen2-Audio / 音频理解

**30 秒回答**

ASR 主要输出语音文字，音频理解还涉及声音事件、音乐、说话风格和依赖声学证据的问答。音频编码器接入 LLM 后需要覆盖这些目标的训练与评估；若只把转写文本送入 LLM，部分非语言信息会丢失。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 分别评估语音内容、环境声音和声学属性。
- 文本提示与直接语音指令是不同输入交互形式。
- 用去除语音、替换背景声等对照检查真正的音频依赖。

### 易错点

- 不要把转写准确率当所有音频理解任务的上限。

### 面试官可能追问

- 同一句话不同语气如何构造评测？

</details>

**技术依据**

- [MM-S039 · Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="omm-011"></a>
## OMM-011 · Qwen2.5-Omni 的 Thinker-Talker 如何协作？

**L2 · 编辑补充题** · 标签：Omni / Thinker-Talker

**30 秒回答**

Thinker 处理多模态输入并生成文本与高层表示，Talker 利用这些表示和文本 token 生成流式语音 token。两者支持联合训练，音视频输入通过时间对齐位置表达关联；它不同于先完整生成文本再调用独立 TTS。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- Thinker 使用音频与图像编码器接入文本解码主干。
- Talker 接收连续表示与离散文本以兼顾语义和发音。
- 流式音频解码需要控制感受野、分块和首包延迟。

### 易错点

- 不要把 Omni 模型的全部输入或输出都说成离散 token。

### 面试官可能追问

- TMRoPE 对同步音视频输入有什么帮助？

</details>

**技术依据**

- [MM-S040 · Qwen2.5-Omni Technical Report](https://arxiv.org/html/2503.20215v1)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="omm-012"></a>
## OMM-012 · ASR→LLM→TTS 流水线与端到端语音对话怎样选？

**L3 · 社区题目线索** · 标签：语音对话 / 端到端

**30 秒回答**

流水线容易检查中间文本、接工具和单独替换组件，但转写可能丢失韵律并累积延迟。端到端语音模型可保留更多声学信息和实现多流对话，却需要更复杂的数据、控制与评测，选择应围绕实际交互任务验证。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 比较同一任务的内容准确、打断恢复和声学表达。
- 流水线可并行和分块，延迟不是各阶段整段耗时的简单固定值。
- 多流建模可处理用户与模型同时说话的全双工情境。

### 易错点

- 端到端不等于必须舍弃文字接口或工具调用。

### 面试官可能追问

- 客服对话和情感陪伴的选择标准为何不同？

</details>

**技术依据**

- [MM-S041 · Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/html/2410.00037v2)

**题目出处线索**

- [MM-S005 · AI-Engineer-Interview-Questions / Multimodal](https://github.com/ombharatiya/AI-Engineer-Interview-Questions/blob/main/10-multimodal/README.md) · `reported_topic`：社区集合比较 STT→LLM→TTS 与 native speech-to-speech。

<a id="omm-013"></a>
## OMM-013 · 流式语音系统的端到端延迟应怎样拆解？

**L3 · 编辑补充题** · 标签：流式语音 / 延迟

**30 秒回答**

用户可感知延迟包括采集分块、编码前视、识别或理解、模型生成、音频解码、传输与播放缓冲。理论 codec 帧延迟只占其中一部分；要记录首包、持续实时率和打断反应，再通过流式重叠优化关键路径。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 区分算法前视、推理耗时和网络/播放器缓冲。
- 低帧率减少 token 负担，却可能增加最小输出块时长。
- 保持长对话实时率，同时测试尾延迟和设备负载。

### 易错点

- 不能用论文理论延迟直接承诺用户端端到端延迟。

### 面试官可能追问

- 降低音频块大小为什么可能降低质量和吞吐？

</details>

**技术依据**

- [MM-S041 · Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/html/2410.00037v2)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="omm-014"></a>
## OMM-014 · 为什么唇读等视觉信息可以帮助噪声下的 ASR？

**L2 · 编辑补充题** · 标签：AV-HuBERT / 音视频

**30 秒回答**

嘴部运动提供与语音相关的可见线索，在音频噪声或缺失时可补充声学证据。音视频表征必须同步，还要处理遮挡、侧脸和多个说话人；可视信息无法区分所有语音，因此不能认为视觉支路能替代可靠音频。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- AV-HuBERT 通过多模态掩码与聚类目标学习联合表示。
- 预处理检查嘴部裁剪和音视频时钟一致性。
- 按噪声、遮挡、视角和不同说话人分组评估。

### 易错点

- 相似口型可对应不同声音，唇读本身存在歧义。

### 面试官可能追问

- 音视频错位多少时应触发降级？

</details>

**技术依据**

- [MM-S042 · Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction](https://arxiv.org/abs/2201.02184)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="omm-015"></a>
## OMM-015 · Omni 模型怎样评估是否真的融合了声音与视觉？

**L3 · 编辑补充题** · 标签：OmniBench / 联合证据

**30 秒回答**

要设计必须联合两种模态才可解答的样本，再分别移除或替换音频、视觉做对照，比较完整输入和单模态基线。单一模态能猜中的题不能证明融合；同步、矛盾线索和模态缺失也应独立评测并保留证据标注。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- OmniBench 强调视觉、声音、文字共同识别和推理。
- 标明答案依赖的时间片段与模态来源。
- 衡量正确答案之外的证据一致性和不确定性处理。

### 易错点

- 多模态输入可被某个强模态捷径主导。

### 面试官可能追问

- 声音与画面相矛盾时应输出什么？

</details>

**技术依据**

- [MM-S043 · OmniBench: Towards The Future of Universal Omni-Language Models](https://arxiv.org/abs/2409.15272)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。
