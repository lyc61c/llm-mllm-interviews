"""Search questions or sample a mock interview. Run with --help."""
import argparse
import json
import random
from common import CATEGORIES, load_bank


def main():
    p = argparse.ArgumentParser(description="LLM / MLLM 中文面试题库；多个关键词按 AND 匹配")
    p.add_argument("keywords", nargs="*")
    p.add_argument("--category", choices=list(CATEGORIES))
    p.add_argument("--level", choices=["L1", "L2", "L3"])
    p.add_argument("--id", help="精确题号，例如 VLM-001")
    p.add_argument("--evidence", choices=["community", "editorial"])
    p.add_argument("--platform", help="社区证据平台子串，例如 知乎、小红书、牛客")
    p.add_argument("--company", help="作者声称的面试公司；仅匹配 source.claimed_company")
    p.add_argument("--mock", type=int, default=0, metavar="N", help="随机抽 N 题；只输出题目")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--answers", action="store_true", help="展开答案")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    if args.mock < 0:
        p.error("--mock 必须大于或等于 0")
    sources, questions = load_bank()
    smap = {s["id"]: s for s in sources}
    def matches(q):
        hay = json.dumps(q, ensure_ascii=False).casefold()
        evidence_sources = [smap[e["source_id"]] for e in q["community_evidence"]]
        return (all(k.casefold() in hay for k in args.keywords)
                and (not args.id or q["id"] == args.id)
                and (not args.category or q["category"] == args.category)
                and (not args.level or q["level"] == args.level)
                and (not args.evidence or (bool(q["community_evidence"]) == (args.evidence == "community")))
                and (not args.platform or any(args.platform.casefold() in s["platform"].casefold() for s in evidence_sources))
                and (not args.company or any(args.company.casefold() in s.get("claimed_company", "").casefold() for s in evidence_sources)))
    selected = [q for q in questions if matches(q)]
    if args.mock:
        selected = random.Random(args.seed).sample(selected, min(args.mock, len(selected)))
    if args.json:
        print(json.dumps(selected, ensure_ascii=False, indent=2))
        return
    print(f"共 {len(selected)} 题")
    for q in selected:
        provenance = "社区题目线索" if q["community_evidence"] else "编辑补充"
        print(f"\n[{q['id']}] {q['level']} · {provenance} · {q['title']}")
        if args.answers:
            a = q["answer"]
            print(a["quick"])
            for key, label in (("detail", "展开"), ("pitfalls", "易错"), ("followups", "追问")):
                print(label + "：")
                for item in a[key]:
                    print("  - " + item)
            if a.get("formula"):
                print("公式：" + a["formula"])
            for sid in q["reference_ids"]:
                print(f"依据：{smap[sid]['title']} — {smap[sid]['url']}")
            for e in q["community_evidence"]:
                print(f"题目线索（{e['support']}）：{smap[e['source_id']]['url']}；{e['note']}")


if __name__ == "__main__":
    main()
