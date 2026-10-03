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
    p.add_argument("--topic", help="主题子串，例如 MoE、位置编码")
    p.add_argument("--id", help="精确题号，例如 VLM-001")
    p.add_argument("--mock", type=int, default=0, metavar="N", help="随机抽 N 题；只输出题目")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--answers", action="store_true", help="展开答案")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    if args.mock < 0:
        p.error("--mock 必须大于或等于 0")
    _, questions = load_bank()
    def matches(q):
        hay = json.dumps({k: v for k, v in q.items() if k not in ('reference_ids', 'community_evidence', 'editorial')}, ensure_ascii=False).casefold()
        return (all(k.casefold() in hay for k in args.keywords)
                and (not args.id or q["id"] == args.id)
                and (not args.category or q["category"] == args.category)
                and (not args.level or q["level"] == args.level)
                and (not args.topic or args.topic.casefold() in q.get('topic','').casefold()))
    selected = [q for q in questions if matches(q)]
    if args.mock:
        selected = random.Random(args.seed).sample(selected, min(args.mock, len(selected)))
    if args.json:
        print(json.dumps([{k: q[k] for k in ('id','category','topic','level','title','tags','answer')} for q in selected], ensure_ascii=False, indent=2))
        return
    print(f"共 {len(selected)} 题")
    for q in selected:
        print(f"\n[{q['id']}] {q['level']} · {q['title']}")
        if args.answers:
            a = q["answer"]
            print(a["body"])
            if a.get("formula"):
                print("\n$$\n" + a["formula"] + "\n$$")
            for key, label in (("pitfalls", "易错"), ("followups", "追问")):
                print(label + "：")
                for item in a[key]:
                    print("  - " + item)


if __name__ == "__main__":
    main()
