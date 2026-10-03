# 规范数据结构

[返回首页](../README.md)

`data/*.json` 是规范数据。顶层为 `{"sources": [...], "questions": [...]}`。生成文件位于 `chapters/`、`exports/` 和索引页面。新分片不会自动免除审阅，必须运行构建与校验。

## Question

| 字段 | 类型与约束 |
|---|---|
| `id` | `分类-三位数字`；全仓库唯一 |
| `category` | TFM/PRE/FT/INF/ALN/RAG/AGT/EVA/VLM/OMM/DST/COD/SYS/PRJ |
| `title` | 中文题目，不冒称未经核实的公司真题 |
| `level` | L1/L2/L3，编辑判断 |
| `tags` | 检索标签列表 |
| `answer.quick` | 可口述短答 |
| `answer.detail` | 至少 3 条展开要点 |
| `answer.pitfalls` | 至少 1 条易错点 |
| `answer.followups` | 至少 1 条进一步问题 |
| `answer.formula` | 可选纯文本公式，明确变量/假设 |
| `reference_ids` | 至少一个技术来源 ID，source.type=primary |
| `community_evidence` | 题目线索列表，每项为 source_id/support/note |
| `editorial` | 没有社区线索时必须为 true，否则 false |
| `user_supplement_keys` | 可选；用户指定问题的映射键，如 U00/U01。映射到 research/supplement-request-2026-10-02.json，不改变社区证据或公司声明 |
| `article_supplement_keys` | 可选；后补核心30问的映射键，如 Z01/Z30。原题文本由用户粘贴，映射到同一请求文件的 article.items，不提升原网页可读范围 |

`support` 只能为 `reported_question`、`reported_topic`、`search_snippet`、`secondary_report`。只可读搜索摘录的社区源不得标为正文明确原题；blocked 不得作为提问证据。可以有“主题来源 + 编辑改写”的题，须在 note 中说明。

## Source

| 字段 | 约束 |
|---|---|
| `id` | 全仓库唯一，如 ENG-P01、MM-S001 |
| `title` / `url` | 真实页面标题/主题与支持它的 HTTP(S) URL |
| `platform` | 知乎、牛客、GitHub、arXiv、官方文档等 |
| `type` | community / primary |
| `access` | full / partial / snippet / blocked |
| `published_date` | 能确认时 YYYY-MM-DD，否则 null；不能把搜索抓取时间当发布日期 |
| `accessed_date` | YYYY-MM-DD，本次 2026-10-02 |
| `notes` | 访问范围、真实性边界、版本和不采用的内容 |
| `topics` | 从页面实际读取的主题 |
| `claimed_company` | 可选；来源声称的面试公司，非雇主确认 |
| `original_url` | 可选；二手报道指向的原帖，需要在 notes 说明原帖访问情况 |

同一 URL 在不同分片可有独立记录，以保留各次读取范围；统计区分 source_records 与 unique_source_urls。可读范围的冲突不自动合并，应根据实际访问记录解释。

## 导出

- `exports/questions.jsonl`：每行完整问题，包括全部答案与证据。
- `exports/questions.csv`：扁平短答版本，便于查看和导入；完整答案仍在 JSONL。
- `exports/sources.json`：全部来源。
- `exports/stats.json`：构建时生成的精确计数。

`validate.py` 检查结构、ID、引用边和本地链接，不验证每个网页当前仍可访问，也不替代数学与事实审阅。
