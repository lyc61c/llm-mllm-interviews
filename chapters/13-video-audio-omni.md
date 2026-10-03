# 视频、语音与 Omni

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 目录

- [视频采样与时序](#topic-1)
  - [OMM-001 · 视频 VLM 怎样构建与训练，帧数和 token 预算如何控制？](#omm-001)
  - [OMM-002 · 视频时间位置编码为什么要考虑真实时间与 FPS？](#omm-002)
  - [OMM-003 · 长视频理解怎样压缩视觉 token，又有什么代价？](#omm-003)
  - [OMM-004 · 视频定位如何输出准确时间点，怎样评估？](#omm-004)
  - [OMM-017 · 开放场景下如何跟踪多个视频实例，并处理遮挡和镜头切换？](#omm-017)
  - [OMM-024 · 1–10分钟第一人称操作视频只有粗 caption，如何切子动作并用于时序训练？](#omm-024)
- [视频生成与控制](#topic-2)
  - [OMM-019 · Wan2.1 有哪些版本，视频生成架构是什么，如何选择并验证原生微调基线？](#omm-019)
  - [OMM-020 · Wan2.2 的高噪与低噪专家怎样切换，SNR、时间门限和 LLM MoE 有什么区别？](#omm-020)
  - [OMM-021 · 视频生成怎样评测时序和动作、设计多维奖励，并校准 Qwen-VL 打分？](#omm-021)
  - [OMM-022 · 视频生成为何在 SFT 后做 GRPO，batch size=8 时怎样采样、打分和更新？](#omm-022)
  - [OMM-023 · 视频生成 RL 的 CLIP/Qwen-VL 评分如何加速，多模型显存和训练耗时怎样估算？](#omm-023)
  - [OMM-025 · 第一人称视频的相机抖动如何建模，camera pose 条件怎样注入生成网络？](#omm-025)
  - [OMM-026 · 手部小目标扭曲模糊如何处理，逐帧 YOLO 手框怎样作为视频生成条件？](#omm-026)
- [语音识别与编码](#topic-3)
  - [OMM-005 · Whisper 的输入、架构和多任务接口是什么？](#omm-005)
  - [OMM-006 · WER 怎样计算？为什么中文 ASR 常同时报 CER？](#omm-006)
  - [OMM-007 · CTC 与自回归语音解码有什么区别？](#omm-007)
  - [OMM-008 · 音频连续特征与离散 codec token 分别适合什么？](#omm-008)
  - [OMM-010 · Qwen2-Audio 这类音频理解模型为什么不等于 ASR？](#omm-010)
  - [OMM-014 · 为什么唇读等视觉信息可以帮助噪声下的 ASR？](#omm-014)
- [语音生成与流式](#topic-4)
  - [OMM-009 · VALL-E 式零样本 TTS 为什么能利用短语音提示？](#omm-009)
  - [OMM-012 · ASR→LLM→TTS 流水线与端到端语音对话怎样选？](#omm-012)
  - [OMM-013 · 流式语音系统的端到端延迟应怎样拆解？](#omm-013)
  - [OMM-018 · 实时视频通话 AI 如何主动响应、判断轮次并处理用户打断？](#omm-018)
- [Omni 融合与评测](#topic-5)
  - [OMM-011 · Qwen2.5-Omni 的 Thinker-Talker 如何协作？](#omm-011)
  - [OMM-015 · Omni 模型怎样评估是否真的融合了声音与视觉？](#omm-015)
  - [OMM-016 · ImageBind 如何用图像桥接多个模态，音频又如何编码？](#omm-016)

<a id="topic-1"></a>
## 视频采样与时序

<a id="omm-001"></a>
### OMM-001 · 视频 VLM 怎样构建与训练，帧数和 token 预算如何控制？

**L2** · 腾讯

#### 答案

增加视频帧会增加视觉 token 和语言上下文开销，而相邻帧常含大量重复信息。抽帧要兼顾事件覆盖与细节，可按任务比较均匀采样、变化采样或分段检索，并保留时间戳；盲目增帧可能更慢，也仍会遗漏短事件。

按帧数、每帧 token 和文本长度估计总序列。均匀抽帧覆盖时间范围，关键帧策略可能偏向视觉变化；图像视频联合训练还需统一表示和输入协议。

视频预训练网络可以从逐帧或时空视觉编码器提取特征，再通过时序聚合和连接器接入语言模型；帧顺序与真实时间必须保留。任务决定监督：视频文本对齐可用对比目标，视频描述/问答可用文本生成目标，动作识别可用分类目标，不存在所有视频预训练都只做一种loss的统一答案。采样和token压缩改变可见信息，要用短事件、动作顺序和时间定位任务验证。

#### 易错点

- 抽帧序号不能代替真实时间，尤其对可变帧率视频。

#### 追问

- 一分钟短视频和两小时视频如何设不同预算？

<a id="omm-002"></a>
### OMM-002 · 视频时间位置编码为什么要考虑真实时间与 FPS？

**L2**

#### 答案

同一帧序号在不同采样率下对应不同真实时刻，仅用帧号会混淆速度和事件间隔。Qwen2.5-VL 将时间位置关联到绝对时间，并用动态 FPS 采样呈现不同采样密度的数据。

输入需记录该帧在原视频中的时刻，统一时间单位，而非只记取出的第几帧。时间表达仍需定位任务训练，评测要检查预测范围、误差和短事件召回。

#### 易错点

- 时间编码改进不能保证模型自动精确定位所有事件。

#### 追问

- 视频被倍速播放时哪些标签需要调整？

<a id="omm-003"></a>
### OMM-003 · 长视频理解怎样压缩视觉 token，又有什么代价？

**L3** · 美团 / 快手

#### 答案

长视频可通过空间池化、时空连接器、分段摘要或检索压缩视觉 token。VideoLLaMA 2 的时空连接器强调联合建模空间与时间；分段处理应携带时间边界，避免用摘要覆盖原始证据。

压缩降低成本，也可能抹掉短事件、局部细节和事件顺序。要按任务保留证据，比较压缩前后的定位与问答效果；细粒度动作题通常需要比主题总结更高的保真预算，不能只看最终文本是否流畅。

动态剪枝要区分编码器/连接器与LLM内部：前者省更多后续计算，后者已支付前端开销。视觉冗余法按空间/跨帧相似性压缩；query/task-aware法按文本Q与视觉K关联保留任务证据，仍须注意分数归约和位置偏差。V2Drop则按同一视觉token在相邻LLM层的表征变化渐进筛选，不是跨帧L2，也不显式索取attention矩阵；变化小只是一种启发式，并非无损证明。任务评分、冗余、手部/交互对象ROI先验可组合，但要消融各自贡献和漏检回退。

LLaVA迁移Qwen-VL必须重建真实视觉span、动态grid/merger对应、帧与空间position IDs及MRoPE，不能复用固定576 token或文本偏移。FlashAttention通常不返回完整attention，强制取权重可能回退并增加显存；若只mask不缩短矩阵，理论剪枝未必加速。公平比较固定模型、输入帧/分辨率、生成长度、硬件/精度与内核，以各层平均保留预算和剪枝层位对齐，同时报selector/ROI开销、端到端延迟、显存及手部/短事件质量；复现还需对齐数据、prompt、预处理和代码版本。生成输出latent网格能否恢复须另验证。

#### 易错点

- 平均所有帧特征会丢失部分顺序信息。

#### 追问

- 怎样设计时间顺序交换的消融测试？
- 手部框漏检时怎样设置最低预算和回退？
- 视觉token只在LLM深层删掉，哪些前端计算已经花掉？
- 跨层L2变化和跨帧L2冗余有什么区别？
- 把LLaVA固定视觉token偏移直接移植Qwen-VL会出什么错？
- 相同最终保留token数为何仍可能计算预算不同？

<a id="omm-004"></a>
### OMM-004 · 视频定位如何输出准确时间点，怎样评估？

**L2** · 腾讯

#### 答案

视频问答通常评估答案，moment retrieval 还需输出与语言查询对应的起止时刻。Moment-DETR 将片段坐标与显著性作为不同目标，评测可用时间区间交并比 tIoU 和阈值召回衡量定位，并与显著性评分分开报告。

设预测和真实区间分别为 $`[s_p,e_p]`$ 与 $`[s_g,e_g]`$，交集长度为 $`I`$，tIoU 是交集长度除以并集长度，适用于两个非零时长区间。按短片段、重复动作和多个正确片段分别检查；答案正确不代表定位正确。

要输出可信时间点，输入侧应保留采样帧的真实PTS或秒级时间戳，训练数据给出事件区间或时间标签，模型学习时间坐标/时间token与视觉事件的对应。稀疏采样只给出粗候选时，可在候选区间二次密集采样细化；逐帧索引不能直接当秒数，可变帧率更不能用固定FPS强行换算。校验起止顺序、视频时长边界和采样对齐，并以定位误差/tIoU测精度；更细数字表示本身不会增加未被采样到的视觉证据。

```math
\begin{aligned} I&=\max\left(0,\min(e_p,e_g)-\max(s_p,s_g)\right)\\ \mathrm{tIoU}&=\frac{I}{(e_p-s_p)+(e_g-s_g)-I} \end{aligned}
```

#### 易错点

- 不要把时间 IoU 当作逐帧分类准确率。

#### 追问

- 有多个有效片段时怎样定义召回？

<a id="omm-017"></a>
### OMM-017 · 开放场景下如何跟踪多个视频实例，并处理遮挡和镜头切换？

**L3** · 腾讯

#### 答案

先明确要跟踪的是同类全部实例还是指定对象，以及身份只需在单镜头内一致，还是要跨镜头关联。以 SAM3 为例，检测器按概念发现当前帧实例，记忆跟踪器传播已有掩码轨迹，再按重叠关系匹配更新；未匹配检测可创建新轨迹，利用时间确认、重复轨迹抑制和高置信检测重新提示，缓解误检、遮挡和漂移。可靠更新记忆很重要，错误掩码反复写入会让漂移自我强化。

镜头切换应作为额外系统设计处理：用帧内容变化或自适应阈值检测切点，并区分硬切、渐变与快速运动。确定切换后，不继续把上个镜头的空间运动和掩码记忆直接传播到新画面，而应重新检测并建立镜头内轨迹；切点误报则会破坏连续身份，因此要单独评测切点精度和召回。

需要跨镜头身份时，另建外观特征、轨迹摘要与全局关联层，根据外观、时间和场景约束匹配，对证据不足的对象保留新 ID 或待确认状态。相同概念不是相同实体，SAM3 的掩码传播不保证任意跨镜头重识别。评估应同时看掩码质量、漏检误检、身份切换、关联指标和延迟；HOTA 可辅助拆解检测与关联能力，不能只报单帧 IoU。

#### 易错点

- 重检测后为相同类别分配旧 ID，会把不同实体错误合并。
- 清理跨镜头空间记忆是工程策略，不能冒称 SAM3 论文内置的自动切镜头机制。
- 遮挡、目标离场和场景切换需要不同恢复条件，不能统一当作删除目标。

#### 追问

- 快速摇镜与硬切如何区分，切点误报会怎样影响跟踪？
- 新增实例确认延迟与实时响应怎样取舍？

<a id="omm-024"></a>
### OMM-024 · 1–10分钟第一人称操作视频只有粗 caption，如何切子动作并用于时序训练？

**L3** · 美团

#### 答案

粗caption可能只说“做早餐”，不能唯一给出“拿杯子、倒水、搅拌”的顺序和边界，更不能把整段描述直接贴到每个短clip。第一人称还存在镜头/手同时运动、手物遮挡、重复操作、目标出画与细小状态变化，画面变化或切镜头不等于子动作边界。

先保留真实时间戳和视频级目标，用多尺度滑窗找到候选，再结合手物接触、物体状态变化、运动特征和语义定位细化边界。可让视频VLM在候选片段上产生“起止时间、动词、对象、前后状态、证据帧、置信度”的伪标签，再合并重复、检查顺序/覆盖和人工抽审；不能只让LLM根据常识写一套必然发生的步骤。Ego4D明确区分整段summary与带时间窗口的动作narration；只有粗summary时应承认弱监督，少量人工动作时间点或边界能提供额外锚点。

离线伪标注便于复核、缓存和重复训练；在线视频步骤拆解可按问题动态细采样，但增加推理成本，生成解释并不能证明边界正确。两者可组合为“离线候选库，在线检索和细化”，按训练视频分组隔离评测，避免相邻clip跨集合泄漏。

训练理解模型可用可信clip-描述对齐、动作识别或带时间坐标的定位目标；生成模型则用短clip及其局部动作条件继续去噪/速度回归，保留前段上下文学习动作衔接。低置信伪标签降权或保留为弱约束，不冒充逐帧真值。评测以人工时间边界、重复动作和未见操作检查定位/分割，再看生成动作完成及跨片段连续性，不能只比较caption流畅度。

#### 易错点

- CliMer等粗时间戳方法仍使用narration和粗时间定位，不能外推成仅一个整段caption就能恢复精确动作边界。
- 手物接触可能不可见，低置信度必须保留，不应让模型补成确定事实。
- 未来视频生成若依赖前段上下文，训练不得把后段真实画面泄露给可部署条件。

#### 追问

- 同一视频中两次倒水如何避免伪标注合并为一个长动作？
- 只新增少量人工监督时，标动作时间点、完整边界或长caption怎样选？

<a id="topic-2"></a>
## 视频生成与控制

<a id="omm-019"></a>
### OMM-019 · Wan2.1 有哪些版本，视频生成架构是什么，如何选择并验证原生微调基线？

**L2** · 快手 / 百度

#### 答案

先明确项目实际用的是 Wan2.1 的哪个 checkpoint，而不是只说“用了万相”。核心 T2V 版本为 1.3B 和 14B：1.3B 官方推荐 480P，14B 支持 480P/720P；I2V 为 14B，分别提供 480P 和 720P 权重。1.3B 生成 720P 并非结构上不可能，而是该分辨率训练有限、官方推荐更稳定的 480P。参数量标签主要描述生成主干，不能直接代表包括文本编码器、VAE、激活和优化器在内的总显存。

架构由 Wan-VAE、umT5 文本编码器和视频 DiT 组成。3D 因果 VAE 按时间×高度×宽度约 4×8×8 压缩为16通道 latent，首帧只做空间压缩；DiT 对 latent 进行时空 patch 化，用时空 RoPE 与自注意力建模视频，并通过 cross-attention 注入文本条件。Wan2.1-I2V 在输入通道拼接待去噪 latent、参考首帧的 VAE latent 与重排后的 mask；另将投影后的 CLIP 图像特征作为独立图像 K/V，与文本分支共用视觉 Q 做分离的 cross-attention，合并两路结果。文本经 umT5 编码后提供文本 K/V，不能把这两种图像条件都描述成普通文本 token。因果 VAE 不意味着 DiT 按帧自回归生成。

基础生成训练采用 Flow Matching：下式按论文约定令 x0 为噪声、x1 为真实视频 latent，学习线性插值轨迹的速度，采样从0走向1。论文先做低分辨率图像预训练，再逐步联合训练更高分辨率的图像和视频；生成模型训练冻结 VAE 和文本编码器。实现若使用“0数据、1噪声”的 scheduler，需同步转换速度符号与积分方向。

选型要说明任务条件和资源：纯文本生成从 T2V 出发，参考首帧驱动优先评估对应分辨率的 I2V；1.3B 可作较低成本消融，14B 的收益须实际评测。对姿态、轨迹等严格控制，原生模型未必具备所需条件接口，可增加控制分支，但仍应先评估冻结原模型、LoRA/原生 DiT 微调等基线，给出选择新结构的依据。固定数据划分、条件、分辨率、帧数与采样设置，在相同或明确报告的算力预算下比较控制遵循、画质、时序一致性、泛化、耗时与峰值显存；项目未做的实验应明确说明，不能将官方模型数字冒充实测结果。

```math
\begin{aligned}x_t&=(1-t)x_0+t x_1,\qquad x_0\sim\mathcal N(0,I),\ t\in[0,1]\\v^*&=x_1-x_0,\qquad \mathcal L_{\mathrm{FM}}=\mathbb E\left[\|u_\theta(x_t,c,t)-v^*\|_2^2\right]\\(F,H,W)=(4k+1,H,W)&\longrightarrow(16,k+1,H/8,W/8)\end{aligned}
```

#### 易错点

- 将Wan2.2的MoE架构或其他版本的能力混入Wan2.1；只写14B而不明确T2V/I2V、分辨率及checkpoint。
- 把4×8×8压缩简单写成所有帧数F/4：首帧单独处理，给定公式限定F=4k+1且空间尺寸可整除。
- 混用论文和实现的时间方向；论文下式0为噪声、1为数据，反方向scheduler需同步改变速度符号。
- 将新增控制分支必然优于原生微调当作结论，或将官方推理显存/速度宣传数字当作个人训练实测。

#### 追问

- 如果原生I2V不能稳定跟随人体姿态，怎样设计受控消融证明控制分支的必要性？
- 冻结VAE和文本编码器后，什么时候可以缓存latent/embedding，什么时候数据增强或prompt变化会使缓存失效？
- 1.3B与14B比较怎样区分参数规模、训练预算和采样设置带来的影响？

<a id="omm-020"></a>
### OMM-020 · Wan2.2 的高噪与低噪专家怎样切换，SNR、时间门限和 LLM MoE 有什么区别？

**L3** · 百度

#### 答案

Wan2.2 中应具体指 T2V-A14B 或 I2V-A14B：它们按去噪阶段使用两个完整生成专家，高噪专家主要建整体布局，低噪专家进一步细化；官方标注总参数约27B、每步激活约14B。TI2V-5B 则是 dense 模型，不能把所有“2.2”都称作相同 MoE。A14B 仍使用 Wan2.1-VAE，TI2V-5B 才采用更高空间压缩的 Wan2.2-VAE。

这里的选择由噪声时间决定，同一步使用一个专家处理整段 latent，不是 LLM 中每个 token 经可学习 router 选择不同 FFN，也不是按视频帧序号轮换。官方推理先算 boundary=b×N，再比较当前 scheduler timestep：t≥boundary 用高噪专家，t<boundary 用低噪专家。所核验默认 N=1000，T2V-A14B 的 b=0.875，I2V-A14B 为0.900；对应875/900是时间门限，不是激活参数比例或固定采样步数。

按实现的噪声坐标，接近1是噪声、接近0是数据，生成由高噪走向低噪。对单位方差、独立的数据/噪声线性插值，下式 SNR=(1-s)²/s²，随噪声量s增加而下降，因此高噪早期SNR低、细化后期SNR高。实际代码不在每一步估算图像内容的SNR再做学习式路由，而是使用与噪声阶段对应的预设t门限；时间shift会改变哪些离散采样步落在门限两边，不能用“40步按比例切成35+5”代替真实scheduler。

两专家增加总容量而控制每步主干计算，仍需保存两套权重；全部驻留显存更大，CPU offload可以减少驻留但增加搬运，实际延迟须测量。微调时明确更新哪一专家、采样哪些噪声区间，并检查切换附近的轨迹和画质；更改solver、shift、门限或CFG后，重新验证两专家覆盖，不能把默认阈值称作跨任务最优值。

```math
\begin{aligned}E(t)&=\begin{cases}E_{\mathrm{high}},&t\ge bN\\E_{\mathrm{low}},&t\lt bN\end{cases}\\x_s&=(1-s)x_{\mathrm{data}}+s\epsilon,\qquad 0\lt s\lt 1\\\mathrm{SNR}(s)&=\frac{(1-s)^2}{s^2},\qquad\frac{d\,\mathrm{SNR}}{ds}=-\frac{2(1-s)}{s^3}\lt 0\end{aligned}
```

#### 易错点

- 把Wan2.2所有型号都当作双专家；将dense TI2V-5B与A14B的参数、VAE或路由机制混写。
- 大t表示高噪低SNR，而不是高SNR；视频帧轴、去噪时间轴与采样循环序号不能混用。
- SNR公式限定独立且单位方差的线性插值；若改用其他路径、缩放或时间约定，应重新计算信号/噪声系数。
- 每步激活14B不等于整个推理系统只存14B权重；offload与并行方式影响显存和时延。

#### 追问

- 调大sample_shift而采样步数不变，会怎样改变高噪专家实际使用的步数？
- 为什么T2V与I2V默认门限不同，如何做门限扫描而不把训练集当评测？
- 只训练低噪专家时，能否保证主体布局或复杂运动能力随之改善？

<a id="omm-021"></a>
### OMM-021 · 视频生成怎样评测时序和动作、设计多维奖励，并校准 Qwen-VL 打分？

**L3** · 美团 / 快手

#### 答案

相邻帧CLIP相似度只是外观一致性代理：静止、重复帧也能很高，真实抓取或镜头运动反而可能降低；不能据此证明动作完成。VBench把主体/背景一致性、闪烁、运动平滑和动态程度分开，闪烁评测还需筛除静态投机。动作应定义为可观察的对象、接触、位移和前后状态，例如“抓住杯子并移到桌面”，再分别测语义遵循、动作完成、手部质量和时序稳定；运动应符合任务，不是越大越好。

FID用图像特征比较集合分布；把视频帧汇成无序集合计算的逐帧FID，打乱帧序也可能不变，不能评时序。FVD使用视频特征比较生成与参考集合分布，可包含时序信息，但不是单视频动作正确率；需固定特征网络、抽帧/FPS、时长、分辨率、预处理和样本量，报告不确定性。LPIPS是图像对感知距离，做重建评测须有正确对齐的参考；开放文本生成不能随意配一段真值，邻帧距离小也可能奖励静止。三者均不能替代文本遵循、动作完成和多样性评测。

奖励组合前先把分项统一成越大越好，距离/误差项须取负或作单调下降映射；再用固定留出样本校准尺度，按独立人工偏好选权重并保留分项结果，避免画质抵消未完成动作，此统计与GRPO组内归一化不同。CLIP偏语义/外观，Qwen-VL也不应独占时序裁判：结合时间定位、状态/接触或姿态检查与经人评验证的专用评分器，承认不可见和稀疏抽帧的盲区。

Qwen-VL评分须固定模型、抽帧时间戳、rubric和schema，为0、5、10提供人工锚点；仅输出0–10不产生可靠量尺。检查重测稳定性、人工排序相关性及对静止、倒序、畸形手的辨别，用留出人评校准，解析失败标无效。VideoScore是用人工多维反馈训练的评分模型，与通用Qwen的prompt评分不同，量表须核对版本。SFT/RL对照固定条件、帧数与采样预算，展示整段和手部局部，多种子盲评动作、质量、多样性；训练奖励升而独立质量降应排查reward hacking。

```math
\begin{aligned}r_{\mathrm{sim}}&=\frac{1}{F-1}\sum_{j=1}^{F-1}\frac{f_j^\top f_{j+1}}{\lVert f_j\rVert_2\lVert f_{j+1}\rVert_2}\\ R(v,c)&=\sum_{k=1}^{K}w_k\,\frac{r_k(v,c)-\mu_k}{\sigma_k+\epsilon},\qquad w_k\ge0,\quad\sum_k w_k=1\end{aligned}
```

#### 易错点

- 相似度奖励要求至少两帧、非零特征范数；静止视频可达高分，但不表示目标动作正确。
- 评分锚点和校准需要独立人工样本；同一裁判既训练又验收会隐藏偏差。
- 稀疏抽帧看不到的手部接触或畸变不能靠裁判提示词恢复。
- FID/FVD是集合分布指标，不能把一条视频的分数当标准实现结果；采样协议和集合规模须一致。
- LPIPS低表示感知上近，不等于动作正确；对合法开放生成用任意真值配对可能误罚多样性。

#### 追问

- 同一组视频都得到9分，怎样判断评分饱和还是动作确实相同？
- “打开抽屉”怎样分别验证动作、终态和手部几何？
- 打乱视频帧顺序后逐帧FID不变，说明什么评测盲区？
- 论文FVD复现不一致，如何逐项对齐抽帧、样本量和特征实现？

<a id="omm-022"></a>
### OMM-022 · 视频生成为何在 SFT 后做 GRPO，batch size=8 时怎样采样、打分和更新？

**L3** · 美团 / 快手

#### 答案

视频SFT通常优化扩散去噪或Flow Matching目标，学习真实场景、运动与手物交互分布；不是对像素做文本回答交叉熵。RL用结果偏好补充动作完成、语义或画质，但依赖可信奖励和采样多样性，不能替代缺失的运动先验。

先说明batch=8指每卡视频数、prompt数、组大小G还是训练microbatch。P个条件各生成G条轨迹，本轮共P×G；完整同条件组评分并跨卡汇总后算相对优势，再按microbatch和梯度累积更新。可以逐采样批异步评分，收齐本轮buffer后更新；不意味着生成完全部6000条。这里G至少为2，示式std采用总体标准差，其他框架须核对配置。G越大相对比较更丰富但成本增长，应看有效奖励方差和独立质量。帧数/时长依动作与模型限制，去噪步数依质量成本曲线；patch size通常由检查点架构固定，改变可能涉及投影权重形状、位置编码和重新适配，不能当普通batch超参随改。

连续latent策略须定义合法随机转移密度，保存状态、下一状态、时间、旧log-prob与采样配置。这里约定0为数据、1为噪声，生成由t走向t-h，h>0；纯ODE条件转移是确定性的，不能照搬LLM token概率比。Flow-GRPO原实验主要是图像，迁移视频须核对scheduler、噪声、CFG及高维log-prob归约。

最终视频只给一次奖励，不等于只更新最后一步。基础score-function梯度把结果奖励乘以各个有效随机转移的log-prob梯度；不需穿过黑盒裁判或整条采样链BPTT。下式末行是未裁剪期望奖励梯度，GRPO另用组内优势、旧策略ratio、clip/KL构造surrogate，并非该梯度的完整无偏实现；统计分配也不能直接证明某步造成何种缺陷。可微奖励沿全链反传是另一类pathwise方案。

集成先核验同策略同轨迹ratio≈1、随机转移方差非零、时间/预测参数化/归约一致、轨迹与group ID不混乱，再查LoRA梯度、数值范围、零方差组和评分有效性。监测KL、clip比例、滞后与独立画质，限制旧样本复用。多少步提升须据固定验证集学习曲线回答，不能报通用数字。

```math
\begin{aligned}N_{\mathrm{rollout}}&=PG,\qquad G\ge2\\ \bar R_p&=\frac1G\sum_{i=1}^G R_{p,i},\quad s_p=\sqrt{\frac1G\sum_{i=1}^G(R_{p,i}-\bar R_p)^2},\quad\hat A_{p,i}=\frac{R_{p,i}-\bar R_p}{s_p+\epsilon}\\ \rho_{p,i,t}&=\exp\!\left[\log\pi_\theta(x_{t-h}\mid x_t,t,c_p)-\log\pi_{\mathrm{old}}(x_{t-h}\mid x_t,t,c_p)\right]\\ \nabla_\theta J&=\mathbb E_{\tau\sim\pi_\theta}\left[R(\tau)\sum_t\nabla_\theta\log\pi_\theta(x_{t-h}\mid x_t,t,c)\right]\end{aligned}
```

#### 易错点

- 视频帧序号与去噪时间步是两条轴，不能把每帧当一次去噪动作。
- 奖励只需黑盒评分时无需穿过Qwen-VL反传；可微奖励梯度优化是另一类方法。
- 异步过旧样本会偏离当前采样策略，须限制滞后并保留正确旧概率；clip不保证任意滞后都安全。
- 核对实际reshape与梯度累积：访问时官方SD3脚本实际重分批未直接采用train.batch_size，不能只改配置字段就声称microbatch已变。
- 完整联合latent密度与按维度平均的log-prob归约尺度不同；采样、训练、clip和KL须保持一致。
- 公式末行是基础未裁剪策略梯度示意，不是GRPO完整带clip/KL目标；只对合法随机转移谈普通连续密度。
- 组内含自身奖励的均值与标准差归一化是GRPO surrogate的一部分，不能冒称未改变原目标的无偏基线。
- latent patch尺寸、帧数和扩散步骤不是同一类参数，修改须符合具体检查点架构。

#### 追问

- 8张卡各有同一prompt的部分样本，如何算组内均值和标准差？
- 确定性去噪步骤是否能参与普通高斯log-prob的ratio更新？
- 最终视频才有奖励，为什么前面的去噪步也可以得到参数梯度？
- score-function与对可微奖励全链反传的显存需求有何区别？
- 组大小变大但奖励全部相同，是否仍有相对学习信号？
- 如何用ratio≈1与单步梯度检查定位扩散GRPO实现bug？

<a id="omm-023"></a>
### OMM-023 · 视频生成 RL 的 CLIP/Qwen-VL 评分如何加速，多模型显存和训练耗时怎样估算？

**L3** · 美团 / 快手

#### 答案

先区分“在线”是当前策略生成的新视频要重新评分，还是评分器本身也训练。RL常使用冻结的CLIP或Qwen-VL作为在线奖励器；不能预先给未来尚未生成的视频打好分。离线可缓存固定数据的伪标注、条件文本embedding及同一视频的可复用特征，必须连同模型、prompt、抽帧和预处理版本标记。

拆解生成去噪、VAE解码、视频取帧、评分排队、模型前向和更新耗时。可按分辨率/帧数拼评分批，复用解码帧与CLIP特征，冻结评分器并关闭反向图，减少不必要输出，或用独立GPU/服务与生成流水重叠。昂贵裁判可蒸馏为小评分器，但要用留出人评和难例确认排序没有改变。降低抽帧率/分辨率会遗漏短动作和手指细节，不能只报吞吐收益。Flow-GRPO官方图像循环会异步提交奖励、收齐buffer后再更新，这是一种实现示例，不代表所有视频系统的唯一调度。

显存包括生成权重与优化器、训练激活、rollout latent轨迹、VAE、多个评分器权重及其输入/激活峰值。同卡驻留简单但会竞争，分卡增加传输与排队；CPU offload或分阶段换入能降驻留显存，也增加搬运成本。小batch降低峰值未必加快；评分器量化后须复核分数与排序，不能假定奖励不变。

6000条条件并不足以报训练小时数：还需组大小、分辨率、帧数、去噪步数、轮数、更新复用、卡型与并行方案。先用代表性小批实测每轮分项时间和峰值显存，再按真实处理轮次估算总时长；下式仅是串行调度的估算，流水重叠应按关键路径重算，并计入验证、检查点与失败重试。

按转移密度做RL时，不需要一直保存整条去噪链的反向激活：可先无梯度采样，保留方法要求的状态/下一状态及旧概率，再逐步或按子批重算有梯度的前向，配合checkpointing。latent轨迹缓存仍占显存，不能把“无需全链反传”说成“无需存轨迹”。具体步骤采样与归约须符合所选算法目标。

```math
T_{\mathrm{total}}\approx U\big(T_{\mathrm{generate}}+T_{\mathrm{decode/score}}+T_{\mathrm{update}}\big)+T_{\mathrm{eval}}+T_{\mathrm{save/retry}}
```

#### 易错点

- 缓存键必须覆盖视频内容和评分协议；策略更新后新视频不能复用旧视频的奖励。
- 流水重叠只有资源与依赖允许时才产生收益，线程异步提交不保证同卡GPU计算真正并行。
- 梯度累积降低更新microbatch峰值，不会自动消除rollout缓存或评分器权重占用。

#### 追问

- 评分GPU成为瓶颈，怎样权衡抽帧、增评分卡和小模型蒸馏？
- 一轮rollout有大量latent，哪些必须保存以重算同一动作的log-prob？

<a id="omm-025"></a>
### OMM-025 · 第一人称视频的相机抖动如何建模，camera pose 条件怎样注入生成网络？

**L3** · 美团

#### 答案

先区分希望复现的相机运动、要抑制的高频抖动与独立物体运动。画面光流混合三者，仅用光流不能直接得到可信camera pose；可用带内参的视频和姿态估计获得轨迹，但遮挡、快速运动、滚动快门与单目尺度不确定都需检查。统一相机到世界/世界到相机约定，并用首帧作相对坐标锚点；抽帧、裁剪和resize后，时间与内参也要同步变换。

条件可把每帧内外参编码为时序token，再以交叉注意力/调制注入；更显式的做法是逐像素射线。约定c2w为旋转R和平移o，下式用K反投影得到相机射线，旋转到世界后归一化，再以Plücker的方向与力矩组成6通道。相机平移是射线原点，不应直接加进方向向量。CameraCtrl将该序列送入camera encoder，形成多尺度特征注入时序attention；迁移到DiT可设计条件token或adapter，但须训练和消融，不能把U-Net实现直接当DiT现成接口。

注入pose使模型有机会把相机变化与主体运动分开，并不自动消除模糊或恢复遮挡几何。想生成稳定视频，需要在推理指定可信平滑轨迹并验证可控性；滤平原轨迹也可能去掉任务必需的转头，不能将任意平滑都当正确。比较无条件、原轨迹、平滑/扰动轨迹，在同场景检查位姿响应、手物交互和时序质量；实际项目还需防止pose encoder利用训练外观捷径。

```math
\begin{aligned}d_{t,u,v}&=\frac{R_tK_t^{-1}[u,v,1]^\top}{\lVert R_tK_t^{-1}[u,v,1]^\top\rVert_2}\\ p_{t,u,v}&=[o_t\times d_{t,u,v};\ d_{t,u,v}]\in\mathbb R^6\end{aligned}
```

#### 易错点

- 公式明确R为camera-to-world旋转，o为相机中心；若用world-to-camera外参需先转换。
- pose估计误差与条件控制误差要分开评测，不能把估计轨迹当绝对真值。
- 相同pose的条件应跟实际帧对应，乱序、不同FPS或裁剪后的旧内参会导致几何错位。

#### 追问

- 视频水平翻转后，camera ray map和图像内参怎样同步处理？
- 平滑轨迹使手物操作看起来不对，怎样区分相机条件冲突和生成先验不足？

<a id="omm-026"></a>
### OMM-026 · 手部小目标扭曲模糊如何处理，逐帧 YOLO 手框怎样作为视频生成条件？

**L3** · 美团

#### 答案

先判断手部损坏来自输入采样/裁剪、VAE压缩、训练数据或生成模型：整帧看似清楚但手只有几个latent格时，RL奖励很难补回已经丢失的细节。SFT可用清晰、覆盖不同姿态和接触状态的真实手物clip建立先验，保留全局上下文并增加手部高分辨率或局部监督；不能简单删掉快速运动中的真实模糊而改变任务分布。RL在模型能产生合理候选后，用可靠局部几何、接触与时序奖励调偏好，两阶段应固定预算比较。

YOLO的每帧四元组先统一成裁剪/resize后坐标，归一化并加时间、左右手或track ID、置信度与缺失mask，处理多手、框抖动和遮挡。可借鉴GLIGEN，把框的Fourier位置特征与手/对象语义特征经MLP形成条件token，供视频时空层交叉注意力读取；也可栅格化成时空mask送adapter，或用ROI分支采样局部特征并带回原坐标。需训练融合，直接把四个数字拼到hidden上不保证模型懂位置。

框只给粗位置与尺度，无法约束指头数、关节、掌面朝向或真实接触；这些目标需更丰富的关键点/姿态或手物关系监督。训练时从目标视频检测的未来手框，推理时也必须由用户轨迹、规划器或预测器提供；否则是不可用的未来条件，不能靠读取尚未生成的真值帧解决。

评估同时看局部细节、完整操作和时间一致性，避免只强化裁判喜欢的静态大手。若用手部先验保护视觉token，保留手及交互对象附近细节，同时给背景、动作前后帧与运动区域最低预算；手框漏检时要回退。注意力显著性小不代表手部无用，具体剪枝代价应结合实际延迟和任务错误验证。

```math
\begin{aligned}\tilde b_{t,k}&=(x_{\min}/W,\ y_{\min}/H,\ x_{\max}/W,\ y_{\max}/H)\\ h_{t,k}&=\mathrm{MLP}\big([e_{\mathrm{hand},k};\ \phi(\tilde b_{t,k});\ e_t;\ q_{t,k}]\big)\end{aligned}
```

#### 易错点

- 归一化只适用于相同变换后的图像坐标；YOLO输出xyxy或xywh需先明确转换。
- GLIGEN原文是图像grounding，时序ID、视频融合与手部监督是需验证的工程扩展。
- 缺失框不能当作手不存在；框条件不能替代手部几何与动作监督。

#### 追问

- 某帧左右手框交叉，如何保持条件身份而不把两只手交换？
- 手框定位正确但六根手指，下一步改条件、数据还是奖励？

<a id="topic-3"></a>
## 语音识别与编码

<a id="omm-005"></a>
### OMM-005 · Whisper 的输入、架构和多任务接口是什么？

**L1**

#### 答案

Whisper 先将音频转换为 log-Mel 频谱，由编码器提取表示，再通过自回归文本解码器生成识别、翻译或时间戳等 token，任务和语言由特殊 token 指定。

原始模型以 30 秒音频段训练，长音频需分窗和时间对齐，并处理跨窗口的重复、遗漏与上下文传播。它不能直接视为无延迟流式识别器，静音、音乐和噪声样本还需单独评估错误转写。

#### 易错点

- 转写保留源语言，翻译与转写并非同一任务。

#### 追问

- 分窗边界落在词中间时怎样处理？

<a id="omm-006"></a>
### OMM-006 · WER 怎样计算？为什么中文 ASR 常同时报 CER？

**L1**

#### 答案

WER 在词级对齐预测与参考，将替换 $`S`$、删除 $`D`$、插入 $`I`$ 的总数除以参考词数 $`N_{\mathrm{ref}}`$；参考词数等于正确、替换和删除的数量之和。CER 则在字符级对齐并使用参考字符数。

中文分词规则影响 WER，CER 更便于比较，但仍须统一标点、数字和大小写规范。corpus WER 应先汇总错误数和参考词数再相除；插入可使 WER 超过 100%，空参考需明确协议。业务上还应报告关键实体错误率。

```math
\mathrm{WER}=\frac{S+D+I}{N_{\mathrm{ref}}}
```

#### 易错点

- 不要无条件平均每句 WER，短句会被过度加权。

#### 追问

- 数字格式不同但含义相同如何归一化？

<a id="omm-007"></a>
### OMM-007 · CTC 与自回归语音解码有什么区别？

**L2**

#### 答案

CTC 对所有能折叠成目标文本的单调对齐路径求和，用 blank 连接音频帧与文本。折叠映射 $`B`$ 合并重复并移除 blank；输入时间步通常要足够覆盖目标和重复符号。它不显式逐步条件于已生成文本，但可配合外部语言模型解码，不能说完全没有语言信息。

自回归解码逐 token 依赖历史输出，适合多任务接口，却有串行解码成本。两者在并行性、语言建模和对齐假设上各有取舍。

```math
\mathcal L_{\mathrm{CTC}}=-\log\sum_{\pi:\,B(\pi)=y}\prod_t p(\pi_t\mid x)
```

#### 易错点

- 相邻重复标签折叠与 blank 的处理顺序要明确。

#### 追问

- 为什么重复字符会增加 CTC 的最小输入长度？

<a id="omm-008"></a>
### OMM-008 · 音频连续特征与离散 codec token 分别适合什么？

**L2** · 腾讯

#### 答案

连续音频特征常用于理解与跨模态对齐，强调语义和任务信息；离散 codec token 将波形压缩成可生成的符号序列，强调重建质量，适合语音语言模型。

残差向量量化逐层编码剩余误差，形成多个码本。增加码本可改善重建细节，也增加生成预测负担；比较表示时需同时看帧率、码本数量、时延、可懂度、音色、失真与 token 预算。

理解型音频分支可用波形前端或log-Mel频谱编码，再按时序特征接入下游任务；ASR常以文本生成/CTC为监督，音频语义对齐可用配对对比损失，codec则围绕波形重建与码率设计。跨模态对齐还需明确配对数据、投影与共享表示；编码形式、对齐目标和语音生成目标应分开回答。

若做音频与文本融合，要先定义任务与坐标：音频可保留帧级节奏/声学特征，文本可提供语义或动作约束。经投影后可用交叉注意力、带模态标记的拼接或条件调制接入生成网络，不能仅因hidden维度相同就直接把时序当成已对齐。音频驱动动作时需把音频帧率、动作帧率和时间窗口对齐；文本全局条件与帧级音频条件的作用不同。以去掉某模态、打乱时间和错配文本的对照检查模型是否真正利用条件，具体编码器和融合层以实际项目实现为准。

#### 易错点

- 离散音频 token 不是字符 token，也不天然只保留语义。

#### 追问

- 为什么低比特率 codec 可能损害细节理解？

<a id="omm-010"></a>
### OMM-010 · Qwen2-Audio 这类音频理解模型为什么不等于 ASR？

**L2**

#### 答案

ASR 主要将语音转为文字，音频理解还包括声音事件、音乐、说话风格及依赖声学证据的问答。音频编码器接入 LLM 后，需有覆盖这些目标的训练和评估；只把转写文本交给 LLM 会丢失部分非语言信息。

分别评估语音内容、环境声音和声学属性，区分文本提示与直接语音指令的交互方式。通过去除语音、替换背景声等对照，检查答案是否真正依赖音频。

#### 易错点

- 不要把转写准确率当所有音频理解任务的上限。

#### 追问

- 同一句话不同语气如何构造评测？

<a id="omm-014"></a>
### OMM-014 · 为什么唇读等视觉信息可以帮助噪声下的 ASR？

**L2**

#### 答案

嘴部运动提供与语音相关的视觉线索，在音频噪声或缺失时补充声学证据。AV-HuBERT 通过多模态掩码与聚类目标学习联合表示，但相似口型可对应不同声音，视觉不能区分所有语音，也不能完全替代可靠音频。

预处理需检查嘴部裁剪和音视频时钟同步。按噪声、遮挡、视角和说话人分组评估，并考虑侧脸、多说话人及音视频错位时的降级。

#### 易错点

- 相似口型可对应不同声音，唇读本身存在歧义。

#### 追问

- 音视频错位多少时应触发降级？

<a id="topic-4"></a>
## 语音生成与流式

<a id="omm-009"></a>
### OMM-009 · VALL-E 式零样本 TTS 为什么能利用短语音提示？

**L2**

#### 答案

VALL-E 将文本到语音建模为声学 codec token 生成。文本约束内容，短参考语音提供说话人特征和声学条件，因此可不为该说话人单独微调就合成语音。

多个 codec 层的生成安排要权衡质量与时延。效果仍受参考质量、语言覆盖和模型数据分布影响，评测需分别检查可懂度、说话人相似度及鲁棒性。

#### 易错点

- 零样本不表示训练集中不存在类似声学条件。

#### 追问

- 嘈杂提示会如何影响合成输出？

<a id="omm-012"></a>
### OMM-012 · ASR→LLM→TTS 流水线与端到端语音对话怎样选？

**L3**

#### 答案

ASR→LLM→TTS 流水线便于检查中间文本、接工具和替换组件，但转写可能丢失韵律，并累积延迟。端到端语音模型可保留更多声学信息、实现多流对话，但需要更复杂的数据、控制和评测。

选择时在同一任务上比较内容准确率、打断恢复和声学表达。流水线也可分块与并行，其延迟并非各阶段整段耗时的固定相加；多流建模可处理用户和模型同时说话的全双工情境。

#### 易错点

- 端到端不等于必须舍弃文字接口或工具调用。

#### 追问

- 客服对话和情感陪伴的选择标准为何不同？

<a id="omm-013"></a>
### OMM-013 · 流式语音系统的端到端延迟应怎样拆解？

**L3**

#### 答案

流式语音的可感知延迟包括采集分块、编码前视、识别或理解、模型生成、音频解码、传输及播放缓冲。需区分算法前视、实际推理耗时和网络/播放器缓冲，理论 codec 帧延迟只是其中一部分。

记录首包延迟、持续实时率和打断反应，再以流式重叠优化关键路径。低帧率减少 token 负担，却可能增大最小输出块时长；长对话还应检查尾延迟与设备负载，不能直接用论文理论延迟承诺用户端性能。

#### 易错点

- 不能用论文理论延迟直接承诺用户端端到端延迟。

#### 追问

- 降低音频块大小为什么可能降低质量和吞吐？

<a id="omm-018"></a>
### OMM-018 · 实时视频通话 AI 如何主动响应、判断轮次并处理用户打断？

**L3** · 字节跳动

#### 答案

把持续感知、是否开口和生成内容分开设计。音视频按时间戳同步，维护用户问题、当前事件、任务状态和已经说过的内容；输入必须只使用当前及过去信息。视觉事件出现不等于立即讲解，需综合用户是否持有话轮、语义是否完成、任务相关性、置信度和重复程度决定回答、等待或保持沉默，并用冷却和去重避免连续评论。VideoLLM-online 的 Streaming EOS 说明响应时机可以训练，但它不直接解决所有音频轮次问题。

用户停顿不能只靠固定静音时长判断：结合 VAD、语义与韵律判断是在思考、附和还是交出话轮；看图或注视线索可作辅助，不能当确定命令。主动提示应围绕当前任务和用户偏好，普通事件等待合适空隙，明确的高优先提醒按预先约定策略处理。全双工模型如 Moshi 可以同时处理输入输出音频，仍需评测误打断与漏响应。

播放期间继续监听，先抑制自身扬声器回声，识别真正的 barge-in；用户要求停止或换问题时，停止播放、清空待播队列并取消过期生成，避免模型停了而音频还继续。历史只提交实际播出的内容及打断位置，以新话语重规划；短暂附和不应一律取消回答。训练数据需含沉默、重叠、停顿、主动触发和打断后恢复，评测分开看触发精确率/召回、过早过晚响应、误打断、打断至静音延迟和恢复正确率，不能只测答案质量或首包速度。

#### 易错点

- 全双工不意味着用户说话时必须始终输出语音，生成沉默也是有效行为。
- 取消推理请求、停止服务端TTS和停止客户端已缓存播放不是同一个操作。
- 流式评测不能提前读取完整视频或未来转写决定何时响应。

#### 追问

- 用户说“嗯”与“停一下”时，应怎样区分附和和打断？
- 网络抖动导致待播队列很长，怎样保证barge-in及时停止实际声音？

<a id="topic-5"></a>
## Omni 融合与评测

<a id="omm-011"></a>
### OMM-011 · Qwen2.5-Omni 的 Thinker-Talker 如何协作？

**L2**

#### 答案

Qwen2.5-Omni 的 Thinker 用音频和图像编码器接入文本解码主干，处理多模态输入并生成文本及高层表示；Talker 接收连续表示与离散文本 token，兼顾语义和发音，生成流式语音 token。

两者支持联合训练，音视频输入通过时间对齐的位置表达关联。该方式不同于先生成完整文本，再调用独立 TTS；流式音频解码还需控制感受野、分块和首包延迟。

#### 易错点

- 不要把 Omni 模型的全部输入或输出都说成离散 token。

#### 追问

- TMRoPE 对同步音视频输入有什么帮助？

<a id="omm-015"></a>
### OMM-015 · Omni 模型怎样评估是否真的融合了声音与视觉？

**L3**

#### 答案

设计必须联合声音和视觉才可解答的样本，再移除或替换音频、视觉，比较完整输入与单模态基线。单一模态就能猜中的题不能证明融合；OmniBench 强调视觉、声音、文字的联合识别和推理。

标注答案所依赖的时间片段与模态，分别评测同步、矛盾线索和模态缺失。除了答案正确率，还需衡量证据一致性和不确定性处理，防止强模态捷径主导结果。

#### 易错点

- 多模态输入可被某个强模态捷径主导。

#### 追问

- 声音与画面相矛盾时应输出什么？

<a id="omm-016"></a>
### OMM-016 · ImageBind 如何用图像桥接多个模态，音频又如何编码？

**L2** · 腾讯

#### 答案

ImageBind 为图像/视频、文本、音频、深度、热成像和 IMU 分别设置编码器与投影头，将输出映射到同维度的归一化表示。训练不要求每条样本同时含六个模态，也不需要覆盖所有模态配对：以图像为共同桥梁，使用图文、视频音频、图像深度等自然配对，分别做双向 InfoNCE，拉近同源配对并区分 batch 中其他样本。原始主要设置冻结已有 OpenCLIP 的图像和文本编码器，训练音频、深度、热成像与 IMU 分支。

音频先由波形转成 Mel 频谱，再切时频 patch 送入 Transformer，最后池化、投影和归一化。原始方案将 16 kHz 的两秒音频转为 128 个 Mel 频带，音频 ViT 使用 patch16、stride10；时间与频率位置不能混成文本词的位置。

当音频与文本各自对齐图像空间时，可出现未经直接音频文本配对训练的跨模态检索和零样本分类。这个涌现是实验观察，不能视为任意模态间精确对齐的保证；声画错配、视觉不可见的声音和假负例都可能损害结果。ImageBind 本体提供表征，生成文本或音频还需接相应解码器或 LLM。

```math
\begin{aligned} \ell_{I\to M}&=-\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(z_i^{I\top}z_i^M/\tau_M)}{\sum_{j=1}^{B}\exp(z_i^{I\top}z_j^M/\tau_M)}\\ \mathcal L&=\sum_M\left(\ell_{I\to M}+\ell_{M\to I}\right),\qquad \|z_i^M\|_2=1 \end{aligned}
```

#### 易错点

- 不需要六模态共现，仍需要真实配对监督；不能说完全没有监督信号。
- 对齐到共同空间不代表共享一个编码器，也不直接提供自回归生成能力。

#### 追问

- 同一视频中的声音来自画外时，怎样过滤配对噪声？
- 如何设计去除图像桥梁或加入直接音频文本配对的消融？

## 参考资料

- [Video-LLaVA: Learning United Visual Representation by Alignment Before Projection](https://arxiv.org/abs/2311.10122)
- [Qwen2.5-VL Technical Report](https://arxiv.org/html/2502.13923v1)
- [VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs](https://arxiv.org/abs/2406.07476)
- [FastV 官方实现与延迟验证说明](https://github.com/pkunlp-icler/FastV)
- [Variation-aware Vision Token Dropping for Faster Large Vision-Language Models（v2）](https://arxiv.org/html/2509.01552v2)
- [V2Drop 官方 LLaVA 实现](https://github.com/xuyang-liu16/V2Drop/blob/main/llava/model/language_model/V2Drop.py)
- [QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries](https://arxiv.org/abs/2107.09609)
- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/html/2212.04356v1)
- [Hugging Face Evaluate WER 官方实现](https://huggingface.co/spaces/evaluate-metric/wer/blob/main/wer.py)
- [PyTorch CTCLoss 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CTCLoss.html)
- [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)
- [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/html/2305.05665)
- [ImageBind 官方模型实现](https://github.com/facebookresearch/ImageBind/blob/main/imagebind/models/imagebind_model.py)
- [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111)
- [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759)
- [Qwen2.5-Omni Technical Report](https://arxiv.org/html/2503.20215v1)
- [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/html/2410.00037v2)
- [Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction](https://arxiv.org/abs/2201.02184)
- [OmniBench: Towards The Future of Universal Omni-Language Models](https://arxiv.org/abs/2409.15272)
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/html/2408.00714v2)
- [SAM 3: Segment Anything with Concepts（v2）](https://arxiv.org/html/2511.16719v2)
- [PySceneDetect Detection Algorithms 官方文档](https://www.scenedetect.com/docs/api/detectors.html)
- [HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking](https://arxiv.org/abs/2009.07736)
- [VideoLLM-online: Online Video Large Language Model for Streaming Video](https://arxiv.org/html/2406.11816)
- [Wan2.1 官方仓库：模型下载与版本配置](https://github.com/Wan-Video/Wan2.1)
- [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/html/2503.20314)
- [Wan2.1 官方代码：WanModel 与时空 RoPE](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py)
- [Wan-AI/Wan2.1-I2V-14B-480P 官方模型卡](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)
- [Wan2.2 官方仓库：A14B MoE 与 TI2V-5B 架构](https://github.com/Wan-Video/Wan2.2)
- [Wan2.2 T2V-A14B 官方配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_t2v_A14B.py)
- [Wan2.2 I2V-A14B 官方配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_i2v_A14B.py)
- [Wan2.2 T2V 官方 timestep 专家选择](https://github.com/Wan-Video/Wan2.2/blob/main/wan/text2video.py)
- [Wan2.2 I2V 官方 timestep 专家选择](https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/wan/image2video.py)
- [Wan2.2 官方 FlowUniPC scheduler](https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/wan/utils/fm_solvers_unipc.py)
- [Wan2.2 官方共享配置](https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/wan/configs/shared_config.py)
- [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/abs/2311.17982)
- [VBench 官方评测仓库](https://github.com/Vchitect/VBench)
- [VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation](https://arxiv.org/html/2406.15252)
- [VideoScore-Qwen2-VL 官方模型卡](https://huggingface.co/TIGER-Lab/VideoScore-Qwen2-VL)
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
- [Towards Accurate Generative Models of Video: A New Metric and Challenges](https://arxiv.org/abs/1812.01717)
- [Google Research 官方 Fréchet Video Distance 实现](https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py)
- [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924)
- [LPIPS 官方 PerceptualSimilarity 仓库](https://github.com/richzhang/PerceptualSimilarity)
- [Flow-GRPO: Training Flow Matching Models via Online RL（v4）](https://arxiv.org/html/2505.05470v4)
- [Flow-GRPO official SD3 training loop](https://github.com/yifan123/flow_grpo/blob/main/scripts/train_sd3.py)
- [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/html/2305.13301)
- [Ego4D 官方 Annotation Guidelines](https://ego4d-data.org/docs/data/annotation-guidelines/)
- [Learning Temporal Sentence Grounding From Narrated EgoVideos](https://proceedings.bmvc2023.org/332/)
- [Temporal Action Segmentation from Timestamp Supervision](https://arxiv.org/abs/2103.06669)
- [CameraCtrl: Enabling Camera Control for Video Diffusion Models](https://arxiv.org/html/2404.02101v2)
- [CameraCtrl 官方相机射线与 Plücker 编码](https://github.com/hehao13/CameraCtrl/blob/main/cameractrl/data/dataset.py)
- [GLIGEN: Open-Set Grounded Text-to-Image Generation](https://arxiv.org/html/2301.07093)
- [HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction](https://hoi4d.github.io/)
