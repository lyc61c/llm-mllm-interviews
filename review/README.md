# 审阅与验证

[返回首页](../README.md)

内容分工完成后进行同伴交叉审阅，审阅者优先核查公式、假设、模型版本、常见误解以及社区线索的实际读取范围。审阅是有限范围的质量检查，不能称为全部答案的形式化证明或第三方认证。

具体审阅意见与修复保留在本目录。构建后的精确统计位于 [stats.json](../exports/stats.json)。最终验证运行记录见 [validation.json](validation.json)。

所有题目统一由[分类题目索引](../QUESTION-INDEX.md)进入；需求映射保留在内部采集记录中，用于检查是否漏题。先前15项检查见 [supplement-peer.md](supplement-peer.md)和 [supplement-core-peer.md](supplement-core-peer.md)；这些历史报告中的30问待补状态发生在用户提供完整文本之前，当前覆盖以最新验证报告为准。原知乎网页仍未直接读取。

用户补贴30问后的最新内容检查见 [core30-content-peer.md](core30-content-peer.md)，新增四题和文本来源守卫检查见 [core30-merge-peer.md](core30-merge-peer.md)。当前30/30个原编号均有规范答案；保留原题重复关系与纠错说明。

运行检查包括规范字段、唯一 ID、来源边、题目出处与编辑标注一致性、本地 Markdown 链接、生成页面数据、代码性质测试和导出。外部 URL 的全量在线可用性没有自动重新探测。

Python 标准库代码实际执行。PyTorch 未安装，相关运行测试明确跳过；源码语法检查和人工审阅不替代运行验证。浏览器本机 Tabbit 路由失败，页面交互由 DOM 测试检查；没有声称完成真实浏览器截图验收。
