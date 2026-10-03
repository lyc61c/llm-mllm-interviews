# 最终修订记录

日期：2026-10-02。[返回审阅说明](README.md)

| 发现 | 修订 | 验证范围 |
|---|---|---|
| 三题把搜索摘录标成正文题目 | TFM-001/013、FT-001 改为 search_snippet | 数据证据校验通过；未虚增可读范围 |
| 在线 softmax/重计算说明只有摘要依据 | 实际补读 FlashAttention 第 3.1 节及 Algorithm 1，更新 CORE-S057 范围 | 内容同伴核查，未复现 GPU 内核 |
| Orca 来源只读摘要却标 full | CORE-S059 改 snippet 并保留读取说明 | 来源记录核对 |
| Prompt-Tuning 和采样缺直接官方依据 | 增补 CORE-S068/069 与对应题引用 | 官方文档实际读取 |
| KV 显存、prefix cache、多模态成本引用过于通用 | 增补 ENG-P29/30/31/32，覆盖缓存 shape/hash、分辨率；公式为编辑推导 | 网页读取、字节数测试 |
| 上下文压缩只有通用系统文档 | SYS-004 增加 APP-S138 的直接 context engineering 来源 | 官方正文已读 |
| ALN-001 只引用 InstructGPT，未支持 DPO | 补 APP-S104 | 引用边校验 |
| 安全评测过度拒答缺直接来源 | EVA-008 加实际已读 XSTest 摘要 APP-S165 | 摘要级核验；未声称阅读全文 |
| pass@k 未说明独立固定分布采样条件 | EVA-014 补前提，排除自适应搜索/去重候选机械套无偏公式 | 数学同伴审阅 |
| TFM-013 的 detail 文本被符号拆开 | 合并正确表达、使用有语义的完整要点 | 生成正文核查 |
| 极小正温度先除法会产生 +inf | 采样先减候选最大 logit 再除温度，增加 1e-320 回归样例 | 标准库测试实际执行通过 |
| RoPE positions=[1] 可广播到多个位置 | PyTorch rotary_adjacent 显式检查位置长度与维数；补边界检查 | 语法/静态审阅通过，PyTorch 运行因未安装而 skip |

交叉审阅的覆盖范围见 [core-peer.md](core-peer.md)、[app-peer.md](app-peer.md)、[mm-peer.md](mm-peer.md)。早期报告中的“待修”保留作为审阅历史，最终处理状态以本记录和 [validation.json](validation.json)为准。

没有发现需要停止交付的未解决内容错误。仍然保留的限制：社区面经真实性未由雇主确认；小红书原帖未读；PyTorch 未运行；网页无真实浏览器截图验收；外部链接可随时间失效。

## 用户补题后的修订

用户本条消息包含14项编号题和Q-Former总问题，共15项需求，映射到18道相关题。新增TFM-016/017/018/019及VLM-026；深化12道既有题，另将FT-005的原理题关联到LoRA总问题，不复制重复题目。

| 发现 | 修订 | 最终状态 |
|---|---|---|
| 补充映射错误类型会抛异常，blocked文章可通过未知status伪装完成 | 校验list[str]、状态枚举和blocked禁提取，缺用户条目阻止构建 | 同伴内存边界测试通过 |
| ITG只称因果掩码会歧义 | VLM-007写明query互看但不看文本、文本看全部query及历史文本 | 与BLIP-2正文核对 |
| CE概率梯度与logits梯度混淆 | TFM-013区分∂CE/∂q_i=−p_i/q_i与∂CE/∂z_i=q_i−p_i | fixed-p、归一化与softmax条件明确 |
| 秩/特征值题缺具体计算方法 | TFM-018增加特征多项式、2×2手算与Hessenberg/Schur概述，增加SUP-P001 | 手算及官方算法文档核对 |
| 可对角化被误写成数秩的必要条件，零奇异值被统称非零谱 | 改为充分条件，区分非零谱与矩形补零 | 数学同伴意见已采纳 |
| 无位置编码的置换性质可能被套到固定因果mask | TFM-019短答明确全可见self-attention前提 | 与正文条件一致 |

当时指定知乎文章689965833仍为blocked，核心30问目录未取得（0/30）；没有将用户独立问题或通用题单冒充该文原题。用户后续提供文本后的处理见下一节，题目统一位于[分类索引](../QUESTION-INDEX.md)。

## 用户随后补贴核心30问

上一节的0/30是收到用户文本前的历史状态。用户随后提供30题及答案，本轮30/30个Z编号已完成映射，合并到23道相关规范题；题库新增TFM-020/021/022/023，从225题增至229题。已有公式与答案独立核验后补充，原网页状态仍为blocked。

| 核验点 | 当前处理 |
|---|---|
| 多处公式粘贴为空，6组以上问题重复 | 恢复attention、MHA、PE、FFN、LN、残差、LM head、global norm clip、Noam及AR概率公式；保留全部30个原编号对应关系 |
| 将标准层/头描述为默认共享参数 | TFM-014区分结构相同、同层跨位置、embedding/head tying与ALBERT跨层共享 |
| 将小权重称为标准Negative Attention | TFM-023区分非负softmax权重、负logit、负V/输出与其他特定命名方法 |
| 将动态attention矩阵视为独立参数或充分解释 | TFM-021给输入与WQ/WK计算路径、梯度公式及解释性研究范围 |
| LN轴和BN适用性、mask施加位置不严谨 | TFM-005明确逐token隐藏轴和epsilon/仿射；TFM-003明确softmax前score掩码及变长padding处理 |
| 将warmup、累积或词面指标泛化为统一答案 | PRE-014限定原始Noam与loss scaling；DST-003区分内存瓶颈；EVA-002组合任务指标、人评与自由生成核验 |
| 将用户文本合并伪装为已读取blocked网页 | 新状态merged_from_user_text保留user_message provenance与原页未核验；请求URL/access必须与来源目录一致 |
| 原题编号被替换或请求字段为空时遗漏/崩溃 | 校验Z01–Z30完整键集、结构与映射；8组覆盖及来源守卫回归测试实际执行通过 |

内容和数学检查见[core30-content-peer.md](core30-content-peer.md)与[core30-merge-peer.md](core30-merge-peer.md)，当前验证结果见[validation.json](validation.json)。
