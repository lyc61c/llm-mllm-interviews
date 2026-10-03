# 工程 / 小红书渠道采集记录

日期：2026-10-02（Asia/Shanghai）。本记录是检索日志摘要，不保存账号、Cookie 或安全 token。

## 实际查询

- `site.xiaohongshu.com/explore 大模型 面试`、`site.xiaohongshu.com 多模态 面经`、`LLM面试 小红书`。
- `site.zhihu.com 大模型 面试 手撕 算法 系统设计`。
- `site.nowcoder.com/discuss 大模型 面经 手撕 字节 阿里`。
- `site.nowcoder.com/discuss 腾讯 混元 多模态 大模型 面经 2025 2026`。
- `site.nowcoder.com/discuss 阿里 通义 多模态 大模型 面经 手撕`。
- `site.nowcoder.com/discuss 百度 大模型 算法 面经 2025`。

## 小红书直接访问

搜索未获得可直接读正文的 LLM/MLLM 原帖，发现 holynova 的“小红书 AI Agent 开发岗位面试调研报告”，其中有原帖链接。

已实际点击三个原帖线索：

- “字节AI Agent全栈开发一面（贼难）”，ID `6a4f1bb60000000006035103`：带参数链接跳转登录错误，提示 URL invalid；`/explore/` 规范链接也不可读。
- “淘天ai应用开发一面面经”，ID `69e4fffd0000000021004d25`：跳转 website-login/error。
- “字节跳动Agent开发岗二面（贼难）”，ID `6a1ab93a0000000035029e6f`：抓取超时/失败。

Tabbit 技能已读取。本机浏览器启动器在默认沙箱拒绝执行，提权检查时命令返回 exit 1 且无诊断正文，未建立可读取会话；没有读取登录后帖子，也没有抓评论。公开报告只作为二手**题目**线索。报告多个问题套用相同答案，本版不采用其答案或热度统计。

`ENG-C03` 是可读二手报告，SYS-004/SYS-014/PRJ-002 标 `secondary_report`。`ENG-C04` 是待核原帖，`blocked`，没有用它直接支持任何题。

搜索结果中的“小红书大模型算法面经”常指**小红书公司**的面试，发布平台可能是牛客/第三方站。本仓库不会据标题把它计作小红书平台原帖。

## 其他社区线索

- `ENG-C01`：牛客字节电商实习正文可读，COD-002/005/010/011 分别来自 MHA、InfoNCE、TopK、岛屿主题。
- `ENG-C02`：牛客字节大模型面经合集仅读公开前段，COD-014/PRJ-001/004 使用显存、项目价值与消融主题，未读付费尾部。
- `ENG-C05`：知乎机构文章搜索摘录可读、open 失败；SYS-012 只取微调上线衰减提问，不采用招聘数字。
- `ENG-C06`：牛客腾讯/百度转载总结的搜索摘录；SYS-005/008 只取重试安全与知识库不停服更新主题。
- `ENG-C07`：百度面经合集搜索摘录；COD-012/SYS-009 使用编辑距离和 RAG 归因主题。

## 答案核验与运行范围

已读取 PyTorch SDPA/CE、Python heapq/collections、Transformers Generation、TRL DPO/GRPO、vLLM metrics、OpenTelemetry、Prometheus、Kubernetes、Google SRE 与 Rules of ML 等相关公开章节。CPC/RoFormer/LoRA/HELM 本组只读摘要，来源明确 `snippet`；代码中的旋转、低秩合并与损失数值公式由编辑独立推导并测试，不声称已读这些论文全文。

标准库 14 组测试包含未来 token 不泄漏、缓存 decode/prefill 对照、旋转不变量、LoRA 合并等价、DPO margin 符号、零方差 GRPO、TopK 排序对照、岛屿并查集对照和 DP 小规模穷举对照。PyTorch 环境未安装，6 组测试 skip，源码可解析，运行结果未验证。
