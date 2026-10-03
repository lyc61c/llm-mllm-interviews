"""Shared, dependency-free loading and validation for the interview bank."""
from pathlib import Path
import json
import re
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "TFM": ("01-transformer", "Transformer 与数学基础"),
    "PRE": ("02-pretraining", "预训练、数据与优化"),
    "FT": ("03-finetuning", "SFT、LoRA 与参数高效微调"),
    "INF": ("04-inference", "推理、KV Cache 与量化"),
    "ALN": ("05-alignment", "RLHF、DPO、PPO 与 GRPO"),
    "RAG": ("06-rag", "RAG、检索与重排"),
    "AGT": ("07-agents", "Agent、工具与上下文工程"),
    "EVA": ("08-evaluation", "评测、幻觉与安全"),
    "VLM": ("09-vision-language", "视觉语言与图文多模态"),
    "OMM": ("10-video-audio-omni", "视频、语音与 Omni"),
    "DST": ("11-distributed", "分布式训练与显存工程"),
    "COD": ("12-coding", "手撕代码与算法"),
    "SYS": ("13-system-design", "系统设计与线上故障"),
    "PRJ": ("14-project", "项目、论文与行为追问"),
}
SUPPORT = {"reported_question", "reported_topic", "search_snippet", "secondary_report"}


def load_bank():
    sources, questions = [], []
    for path in sorted((ROOT / "data").glob("*.json")):
        # One canonical shard per owner; generated exports live in exports/.
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        sources.extend(data["sources"])
        questions.extend(data["questions"])
    order = {cat: i for i, cat in enumerate(CATEGORIES)}
    questions.sort(key=lambda q: (order.get(q["category"], 999), q["id"]))
    return sources, questions


def validate(sources, questions, require_complete=True):
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    for label, rows in (("source", sources), ("question", questions)):
        duplicates = [key for key, n in Counter(x.get("id") for x in rows).items() if n > 1]
        check(not duplicates, f"duplicate {label} IDs: {duplicates}")
    smap = {s["id"]: s for s in sources}
    for s in sources:
        sid = s["id"]
        for key in ("title", "url", "platform", "type", "access", "accessed_date", "notes", "topics"):
            check(key in s, f"{sid}: missing {key}")
        check(s.get("type") in {"primary", "community"}, f"{sid}: invalid source type")
        check(s.get("access") in {"full", "snippet", "blocked", "partial"}, f"{sid}: invalid access")
        check(s.get("url", "").startswith(("https://", "http://")), f"{sid}: invalid URL")
    for q in questions:
        qid = q["id"]
        check(q.get("category") in CATEGORIES, f"{qid}: invalid category")
        check(re.fullmatch(re.escape(q.get("category", "")) + r"-\d{3}", qid), f"{qid}: invalid ID")
        check(q.get("level") in {"L1", "L2", "L3"}, f"{qid}: invalid level")
        check(bool(q.get("title", "").strip()), f"{qid}: empty title")
        a = q.get("answer", {})
        check(len(a.get("quick", "")) >= 25, f"{qid}: short or missing quick answer")
        for key in ("detail", "pitfalls", "followups"):
            check(isinstance(a.get(key), list) and bool(a[key]), f"{qid}: missing {key}")
        check(len(a.get("detail", [])) >= 3, f"{qid}: fewer than 3 detail points")
        check(bool(q.get("reference_ids")), f"{qid}: no technical reference")
        for sid in q.get("reference_ids", []):
            check(sid in smap, f"{qid}: dangling reference {sid}")
            if sid in smap:
                check(smap[sid]["type"] == "primary", f"{qid}: reference {sid} is not primary")
                check(smap[sid]["access"] != "blocked", f"{qid}: blocked technical reference {sid}")
        ev = q.get("community_evidence", [])
        check(q.get("editorial") == (not bool(ev)), f"{qid}: editorial/evidence mismatch")
        for e in ev:
            sid = e.get("source_id")
            check(sid in smap, f"{qid}: dangling evidence {sid}")
            check(e.get("support") in SUPPORT, f"{qid}: invalid evidence support")
            if sid in smap:
                check(smap[sid]["type"] == "community", f"{qid}: community evidence uses primary source")
                check(smap[sid]["access"] != "blocked", f"{qid}: cannot infer question from blocked source")
                if smap[sid]["access"] == "snippet":
                    check(e.get("support") == "search_snippet", f"{qid}: snippet misrepresented as full evidence")
    if require_complete:
        check(len(questions) >= 200, f"expected >=200 questions, got {len(questions)}")
        for cat in CATEGORIES:
            check(any(q["category"] == cat for q in questions), f"empty category {cat}")
    return errors
