"""Build Markdown chapters, machine-readable exports and an offline explorer."""
import csv
import html
import json
from collections import Counter
from common import ROOT, CATEGORIES, load_bank, validate
from supplement import validate_supplement


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def bullet(items):
    return "\n".join("- " + item for item in items)


def main():
    sources, questions = load_bank()
    errors = validate(sources, questions)
    errors.extend(validate_supplement(questions, sources))
    if errors:
        raise SystemExit("\n".join(errors))
    smap = {s["id"]: s for s in sources}
    counts = Counter(q["category"] for q in questions)
    for cat, (slug, title) in CATEGORIES.items():
        rows = [q for q in questions if q["category"] == cat]
        lines = [f"# {title}", "", "[返回仓库首页](../README.md) · [来源目录](../SOURCES.md) · [阅读全部题目](../exports/questions.jsonl)", "",
                 "L1 基础 · L2 进阶 · L3 深入。难度为编辑判断；社区线索不等于雇主确认真题。所有答案为独立整理。", "", "## 题目导航", ""]
        lines.extend(f"- [{q['id']} · {q['title']}](#{q['id'].lower()})" for q in rows)
        for q in rows:
            a = q["answer"]
            evidence = "编辑补充题" if q["editorial"] else "社区题目线索"
            lines.extend(["", f"<a id=\"{q['id'].lower()}\"></a>", f"## {q['id']} · {q['title']}", "",
                          f"**{q['level']} · {evidence}** · 标签：" + " / ".join(q["tags"]), "", "**30 秒回答**", "", a["quick"], "",
                          "<details>", "<summary>展开答案、易错点与追问</summary>", "", "### 展开要点", "", bullet(a["detail"])])
            if a.get("formula"):
                lines.extend(["", "### 公式", "", "```text", a["formula"], "```"])
            lines.extend(["", "### 易错点", "", bullet(a["pitfalls"]), "", "### 面试官可能追问", "", bullet(a["followups"]), "", "</details>", "", "**技术依据**", ""])
            lines.extend(f"- [{sid} · {smap[sid]['title']}]({smap[sid]['url']})" for sid in q["reference_ids"])
            if q["community_evidence"]:
                lines.extend(["", "**题目出处线索**", ""])
                for e in q["community_evidence"]:
                    s = smap[e["source_id"]]
                    lines.append(f"- [{s['id']} · {s['title']}]({s['url']}) · `{e['support']}`：{e['note']}")
            else:
                lines.extend(["", "**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。"])
        write(ROOT / "chapters" / f"{slug}.md", "\n".join(lines) + "\n")
    index = ["# 题目索引", "", "[返回首页](README.md)", "", "支持按题号定位；全文搜索与筛选请打开 [离线题库](index.html)。", "", "| 题号 | 类别 | 难度 | 题目 | 题目线索 |", "|---|---|---|---|---|"]
    for q in questions:
        slug, title = CATEGORIES[q["category"]]
        label = "编辑补充" if q["editorial"] else "社区线索"
        qtitle = q["title"].replace("|", " / ")
        index.append(f"| {q['id']} | {title} | {q['level']} | [{qtitle}](chapters/{slug}.md#{q['id'].lower()}) | {label} |")
    write(ROOT / "QUESTION-INDEX.md", "\n".join(index) + "\n")
    catalog = ["# 来源目录与访问边界", "", "[返回首页](README.md) · [采集方法](guides/methodology.md) · [公司线索索引](COMPANY-INDEX.md)", "",
               "访问日期：2026-10-02（Asia/Shanghai）。`full`=正文可读；`partial`=公开部分可读；`snippet`=摘要/搜索摘录；`blocked`=正文未读。", "",
               "同一 URL 可能在不同分片中保留多个记录，以保留各自的读取情况；唯一 URL 数量另见统计。论文摘要不代表全文公式已核验。", ""]
    for s in sources:
        catalog.extend([f"<a id=\"{s['id'].lower()}\"></a>", f"## {s['id']} · {s['title']}", "",
                        f"- 链接：[{s['title']}]({s['url']})", f"- 平台：{s['platform']}；类型：{s['type']}；可读范围：{s['access']}",
                        f"- 发布日期：{s.get('published_date') or '未确认'}；访问日期：{s['accessed_date']}",
                        "- 主题：" + " / ".join(s["topics"]), "- 备注：" + s["notes"], ""])
        if s.get("claimed_company"):
            catalog.extend(["- 作者/转载者声称的面试公司：" + s["claimed_company"] + "（未经雇主确认）", ""])
        if s.get("original_url"):
            catalog.extend([f"- 报道所指原帖：[{s.get('original_title', '原帖')}]({s['original_url']})；该链接的访问状态见备注。", ""])
    write(ROOT / "SOURCES.md", "\n".join(catalog))
    companies = ["# 公司线索索引", "", "[返回首页](README.md)", "", "只收录来源中明确提到的公司；这里呈现作者或转载者的说法，不证明雇主曾问过这些题，也不推断公司的固定偏好。", ""]
    company_names = sorted({s.get("claimed_company") for s in sources if s.get("claimed_company")})
    for company in company_names:
        sids = {s["id"] for s in sources if s.get("claimed_company") == company}
        rows = [q for q in questions if any(e["source_id"] in sids for e in q["community_evidence"])]
        companies.extend([f"## {company}", ""])
        for q in rows:
            slug = CATEGORIES[q["category"]][0]
            companies.append(f"- [{q['id']} · {q['title']}](chapters/{slug}.md#{q['id'].lower()})")
        if not rows:
            companies.append("- 保留来源线索，本版未据此抽取题目。")
        companies.append("")
    write(ROOT / "COMPANY-INDEX.md", "\n".join(companies))
    exports = ROOT / "exports"
    exports.mkdir(exist_ok=True)
    write(exports / "questions.jsonl", "\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n")
    write(exports / "sources.json", json.dumps(sources, ensure_ascii=False, indent=2) + "\n")
    with (exports / "questions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "level", "title", "quick_answer", "tags", "editorial", "reference_ids"])
        for q in questions:
            w.writerow([q["id"], q["category"], q["level"], q["title"], q["answer"]["quick"], ";".join(q["tags"]), q["editorial"], ";".join(q["reference_ids"])])
    stats = {"date": "2026-10-02", "questions": len(questions), "categories": dict(counts),
             "levels": dict(Counter(q["level"] for q in questions)), "community_questions": sum(not q["editorial"] for q in questions),
             "editorial_questions": sum(q["editorial"] for q in questions), "source_records": len(sources),
             "unique_source_urls": len({s["url"] for s in sources}), "source_types": dict(Counter(s["type"] for s in sources)),
             "source_access": dict(Counter(s["access"] for s in sources))}
    write(exports / "stats.json", json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    payload = json.dumps({"questions": questions, "sources": sources, "categories": CATEGORIES}, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = (ROOT / "tools" / "explorer-template.html").read_text(encoding="utf-8")
    write(ROOT / "index.html", template.replace("__BANK_JSON__", payload))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
