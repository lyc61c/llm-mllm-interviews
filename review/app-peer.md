# 应用与对齐分片的工程同伴审阅

日期：2026-10-02。审阅范围：`interviews/data/engineering.json` 的 COD-001—015、SYS-001—015、PRJ-001—010；静态读 `interviews/coding/reference.py` 及现有标准库测试；实际执行一次针对极小温度的复现。本轮不改其他分片，不重复根任务已通过的常规测试。

## A-01 [P2] COD-006 的合法小正温度会在缩放前溢出

位置：`interviews/coding/reference.py` 的 `sample_logits`，当前约第 99 行。

复现：`sample_logits([1.0, 2.0], temperature=1e-320)` 报 `ValueError: nonempty logits with no NaN or +inf required`。输入 logits 有限，温度为合法正值，期望近似确定地选第二项；错误来自先计算 `logits/temperature` 产生 +inf，之后再在 softmax 内减最大值已来不及。

修法：候选过滤后先求候选最大值 m，以 `(logits[i]-m)/temperature` 送入 softmax。最大项保持 0，其余可安全成为 -inf。维持温度零的独立贪心路径与当前 top-k/top-p 顺序。可补 `[1,2]` 和 `[-2,-1]` 在极小正温度下选最大项的边界测试，不改变普通采样分布。

## A-02 [P2] SYS-004 的 Agent 压缩建议需要直接出处

答案对目标、未完成状态、外置引用和摘要幻觉的边界判断合理；现有 ENG-P25 SLO 与 ENG-P26 SRE 故障定位支持质量/回放原则，但未直接说明 Agent compaction 与持久化笔记机制。

修法：追加已核验的 [Anthropic context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)（本分片 APP-S138），或建立相同 URL 的工程来源。正文的 Compaction / Structured note-taking 段支持保留决策、未解决问题、外部笔记与压缩丢细节的风险。notes 仍应说明权限、版本与评测指标属于编辑综合。

## 已核对通过的关键项

- COD-008 / `dpo_loss`：标准 sigmoid DPO 的 chosen/rejected/reference 顺序、softplus(-z) 符号与稳定计算正确。使用回答序列 logprob 求和而非长度平均，reference 冻结，prompt/padding 排除，初始同策略 loss=log(2) 均正确。
- COD-009 / `grpo_advantages`：总体标准差 correction=0 明示，不冒称完全复现所有框架；全同奖励与 G=1 的相对奖励优势为零，ε 不制造信号，KL 等独立项仍可能更新。clip 不是额外熵梯度来源的陷阱说明正确。
- COD-003 的 KV offset 因果 mask、COD-004 的 RoPE 相对旋转符号、COD-007 的 LoRA 形状与初始梯度、COD-014 的 512 MiB 均正确。
- SYS-003 不平均实例 P95、SYS-005 未知结果超时/幂等、SYS-007 检索候选权限边界、SYS-008 索引版本与删除传播、SYS-009 oracle 归因的条件均合理。
- 现有标准库测试覆盖有意义的性质与独立算法对照；没有把静态检查当作 PyTorch 运行验证。

## 来源与小红书 provenance

ENG-C03 明确是可读二手报告，ENG-C04 是无法读取的原帖；SYS-004/SYS-014/PRJ-002 使用 `secondary_report`，没有把 blocked 原帖作为直接题目证据。这一划分诚实，工程研究记录也明确浏览器访问失败、未读登录会话或评论。报告答案模板重复及热度统计未采用，避免把二手错误搬入答案。

`claimed_company` 应持续解释为页面自称或二手对应条目，不代表公司官方确认；合集若覆盖多个公司，要靠逐题 note 或拆来源保留实际条目归属，不能把整个报告所有题都当同一家公司真题。

多模态同伴指出的 COD-014 / SYS-011 / SYS-013 直接引用不足，本轮已见根任务追加直接资料的修订；不将同一问题重复列为待修数学错误。工程的摘要来源 notes 与 snippet 口径明确，未冒称读取论文全文。

