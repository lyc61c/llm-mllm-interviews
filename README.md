# LLM / MLLM 大厂面试题与答案

中文面试备考仓库，按知识体系整理公开面经里的题目线索，并补齐必要的原理、工程和项目追问。采集日期为 **2026-10-02，Asia/Shanghai**。

当前版本共 **229 题、14 类**：106 题有社区题目/主题/摘录/二手报道线索，123 题为编辑补充。每题提供 **30 秒短答 → 展开要点 → 公式（适用时）→ 易错点 → 追问 → 技术依据 → 题目来源线索**。答案独立编写，社区答案不作为技术事实的最终依据。

**开始阅读：[离线搜索与自测](index.html) · [全部题目索引](QUESTION-INDEX.md) · [公司线索索引](COMPANY-INDEX.md) · [来源目录](SOURCES.md) · [学习路线](guides/study-plan.md)。**

## 类别

| 编号 | 类别 | 适合准备的方向 |
|---|---|---|
| TFM | [Transformer 与数学基础](chapters/01-transformer.md) | QKV、位置编码、归一化、优化基础 |
| PRE | [预训练、数据与优化](chapters/02-pretraining.md) | 数据清洗、Scaling、优化器、MoE |
| FT | [SFT、LoRA 与参数高效微调](chapters/03-finetuning.md) | 指令数据、训练目标、低秩适配 |
| INF | [推理、KV Cache 与量化](chapters/04-inference.md) | Attention、调度、量化、长上下文 |
| ALN | [RLHF、DPO、PPO 与 GRPO](chapters/05-alignment.md) | 奖励、优势、偏好、RLVR |
| RAG | [RAG、检索与重排](chapters/06-rag.md) | 切片、召回、重排、归因、多跳 |
| AGT | [Agent、工具与上下文工程](chapters/07-agents.md) | 执行模式、状态、工具、协议 |
| EVA | [评测、幻觉与安全](chapters/08-evaluation.md) | 基准、人评、裁判、污染与攻击 |
| VLM | [视觉语言与图文多模态](chapters/09-vision-language.md) | CLIP、BLIP、LLaVA、Qwen-VL、视觉 token |
| OMM | [视频、语音与 Omni](chapters/10-video-audio-omni.md) | 时间建模、ASR、Codec、流式交互 |
| DST | [分布式训练与显存工程](chapters/11-distributed.md) | DDP、ZeRO、FSDP、TP/PP/SP/CP |
| COD | [手撕代码与算法](chapters/12-coding.md) | Attention、RoPE、损失、采样、TopK、DP |
| SYS | [系统设计与线上故障](chapters/13-system-design.md) | SLO、延迟、灰度、隔离、重试与回滚 |
| PRJ | [项目、论文与行为追问](chapters/14-project.md) | 个人贡献、消融、debug、复现与沟通 |

**L1** 基础题；**L2** 深入推导与工程取舍；**L3** 复杂情境与系统判断。难度是编辑判断，没有通过非代表性网络样本推断真实提问频率。

## 来源怎么读

- **社区题目线索**：题目或主题在可读社区资料中出现。证据分为明确问题、主题、搜索摘录和二手报道，不能统一称为“已认证大厂真题”。
- **编辑补充题**：为补全体系、设计代码练习或项目追问而编写；没有挂靠公司。
- **技术依据**：论文、项目官方仓库与官方文档。只读到论文摘要时会注明，公式还需要独立推导、代码或全文证据。
- **小红书边界**：找到带原帖链接的二手调研报告，但原帖返回登录/无效链接/访问错误；本版少量小红书线索明确标为 `secondary_report`，未声称直接抓取登录后笔记或评论。
- **知乎边界**：部分正文可读，部分只有公开搜索摘录；对应记录保留 `full` 或 `snippet`。

每个来源记录都包含 URL、平台、访问范围、日期和备注。公司字段仅表示来源作者/转载者的说法。[采集与核验方法](guides/methodology.md)解释全部标注。

## 本地使用

打开 `index.html` 即可搜索、筛选、收藏和隐藏答案自测。网页完全离线，公式采用文本形式，无 CDN 或账号依赖。命令行工具仅需 Python 3.10+ 标准库。

在本目录下运行：

```bash
python -X utf8 tools/search.py LoRA --answers
python -X utf8 tools/search.py --category VLM --level L2
python -X utf8 tools/search.py --platform 知乎 --answers
python -X utf8 tools/search.py --company 字节跳动
python -X utf8 tools/search.py --category ALN --mock 8 --seed 42
python -X utf8 tools/search.py --id COD-002 --answers
python -X utf8 tools/search.py --evidence editorial --json
```

多个关键词按 AND 匹配题目、标签与答案。命令行抽题默认不展开答案，`--answers` 可展开；`--json` 输出完整结构，便于自己做 Anki/脚本，不是 Anki 专用格式。

## 维护与验证

```bash
python -X utf8 tools/build.py
python -X utf8 tools/validate.py
python -X utf8 -m unittest discover -s coding -p "test_*.py" -v
```

优先编辑 `data/*.json` 中的规范分片；`build.py` 重新生成章节、索引、来源目录、导出与离线页面，避免手工改生成文件后被覆盖。结构约定见 [数据规范](guides/schema.md)，贡献方法见 [CONTRIBUTING.md](CONTRIBUTING.md)。[统计](exports/stats.json)保留题量、分类、证据与来源记录数，来源记录数和唯一 URL 数分别计算。

标准库代码有真实执行的性质与对照测试。PyTorch 是可选练习代码，本次机器未安装，相关 6 组运行测试标记为 skip；详见 [代码说明](coding/README.md)与 [审阅记录](review/README.md)。

已安装 Node.js 时，可运行 `node tools/test_explorer.cjs` 检查页面交互，或用 `python -X utf8 tools/run_checks.py` 重跑全部本地检查并保存报告；日常离线阅读不需要 Node.js。`python -X utf8 tools/package.py` 生成独立压缩包和文件校验清单。

## 目录

```text
data/          规范题库与来源分片
chapters/      按类别生成的中文问答
research/      检索与访问日志
guides/        学习路线、方法、数据约定
coding/        标准库 / PyTorch 手撕参考与测试
tools/         构建、检索、校验与打包工具
exports/       JSONL、CSV、来源 JSON、统计
review/        交叉审阅与实际执行检查
index.html     离线题库
```

本目录可独立作为仓库使用，不依赖父目录原有教程。许可与外部内容边界见 [LICENSE.md](LICENSE.md)。公开链接可能失效；目前这是有访问日期的静态版本，后续新增来源需重新核验。
