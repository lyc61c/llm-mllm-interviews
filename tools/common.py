"""Shared, dependency-free loading and validation for the interview bank."""
from pathlib import Path
import json
import re
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "BAS": ("01-fundamentals", "NLP、数学与深度学习基础"),
    "TFM": ("02-transformer", "Transformer、Attention 与位置编码"),
    "ARC": ("03-model-architectures", "模型架构、MoE 与模型家族"),
    "PRE": ("04-pretraining", "预训练、语料与优化"),
    "FT": ("05-finetuning", "SFT、PEFT、蒸馏与模型编辑"),
    "ALN": ("06-alignment", "强化学习、RLHF 与偏好优化"),
    "INF": ("07-inference", "推理、解码、量化与服务引擎"),
    "DST": ("08-distributed", "分布式训练、并行与显存"),
    "RAG": ("09-rag", "RAG、检索、重排与图检索"),
    "AGT": ("10-agents", "Agent、规划、工具与多智能体"),
    "EVA": ("11-evaluation", "评测、幻觉、安全与鲁棒性"),
    "VLM": ("12-vision-language", "视觉语言模型与图文多模态"),
    "OMM": ("13-video-audio-omni", "视频、语音与 Omni"),
    "COD": ("14-coding", "手撕代码与算法"),
    "SYS": ("15-system-design", "系统设计、性能与可靠性"),
    "PRJ": ("16-project", "项目、论文与工程实践"),
}
TOPICS = {
    'BAS': ('分词与文本表示', '序列任务与经典 NLP', '数学、概率与统计', '深度学习与优化基础'),
    'TFM': ('注意力机制与掩码', '位置编码与长上下文', '归一化、FFN 与残差', '复杂度与实现机制'),
    'ARC': ('架构范式与参数', '预训练模型与家族对比', 'MoE 路由、训练与压缩', '推理模型与 MLA'),
    'PRE': ('语言建模与规模规律', '语料清洗与数据配比', '训练精度与优化', '继续预训练与域适配'),
    'FT': ('指令数据与训练目标', 'LoRA 与其他 PEFT', '训练技巧与排错', '知识蒸馏与模型编辑'),
    'ALN': ('强化学习基础', 'RLHF 与 PPO', 'DPO 与偏好数据', 'GRPO 与在线优化', '奖励与对齐策略'),
    'INF': ('KV Cache 与服务调度', '解码与采样策略', '注意力与算子加速', '量化与稀疏推理'),
    'DST': ('训练显存与状态分片', '并行策略与通信', '精度与激活优化', '训练故障与可靠性'),
    'RAG': ('文档切分与索引', '召回、融合与重排', '查询优化与图检索', '评测、权限与多模态 RAG'),
    'AGT': ('提示、推理与规划', '工具调用与协议', '记忆与上下文', '多 Agent 协同与训练', '可靠性与评测'),
    'EVA': ('评测协议与基准', '指标与统计检验', '幻觉与可信度', '安全与鲁棒性'),
    'VLM': ('图文表示与预训练', '视觉连接器与训练', '动态分辨率与位置编码', '模型结构与版本对比', '视觉生成与扩散模型', '感知、文档与视觉评测'),
    'OMM': ('视频采样与时序', '语音识别与编码', '语音生成与流式', 'Omni 融合与评测'),
    'COD': ('模型算子与数值实现', '损失函数与训练代码', '采样与缓存实现', '通用算法与数据结构'),
    'SYS': ('系统设计与资源预算', '性能、缓存与并发', '质量诊断与版本治理', '权限、容错与可观测性'),
    'PRJ': ('项目贡献与实验设计', '调试、数据与复现', '论文、视野与工程习惯'),
}
SUPPORT = {"reported_question", "reported_topic", "search_snippet", "secondary_report"}


def load_bank():
    sources, questions = [], []
    for path in sorted((ROOT / "data").glob("*.json")):
        # Canonical shards hold original answers and internal reference data.
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
        check(q.get('topic') in TOPICS.get(q.get('category'), ()), f'{qid}: invalid topic')
        check(re.fullmatch(r'(?:'+'|'.join(CATEGORIES)+r')-\d{3}', qid), f"{qid}: invalid ID")
        check(q.get("level") in {"L1", "L2", "L3"}, f"{qid}: invalid level")
        check(bool(q.get("title", "").strip()), f"{qid}: empty title")
        companies = q.get("company_tags", [])
        valid_companies = isinstance(companies, list) and all(
            isinstance(company, str) and 0 < len(company) <= 40
            and company == company.strip()
            and not re.search(r'[\x00-\x1f<>|$`\[\]*]', company)
            for company in companies
        )
        check(valid_companies, f"{qid}: company_tags must be a list of plain company names")
        if valid_companies:
            check(len(companies) == len(set(companies)), f"{qid}: duplicate company tag")
        a = q.get("answer", {})
        check(isinstance(a.get("body"), str) and len(a.get("body", "")) >= 80, f"{qid}: short or missing answer")
        check("quick" not in a and "detail" not in a, f"{qid}: answer must be a single body")
        check('$$' not in a.get('body', ''), f'{qid}: put block math in formula, not body')
        for key in ("pitfalls", "followups"):
            check(isinstance(a.get(key), list) and bool(a[key]), f"{qid}: missing {key}")
        check(isinstance(a.get("formula", ""), str), f"{qid}: formula must be LaTeX text")
        for link in q.get('code_links', []):
            path = link.get('path', '')
            line = link.get('line', 0)
            function = link.get('function', '')
            valid = bool(re.fullmatch(r'coding/[a-z0-9_]+\.py', path)) and isinstance(line, int) and line > 0 and bool(re.fullmatch(r'[A-Za-z_]\w*', function))
            check(valid, f'{qid}: invalid code link')
            if valid:
                file = ROOT / path
                contents = file.read_text(encoding='utf-8').splitlines() if file.is_file() else []
                check(line <= len(contents) and bool(re.match(r'(?:def|class)\s+' + re.escape(function) + r'\b', contents[line-1])) if line <= len(contents) else False, f'{qid}: code link does not point to its function or class')
        check('$' not in a.get('formula', ''), f'{qid}: formula must not contain math delimiters')
        math_expressions = [a.get('formula', '')]
        for value in (a.get('body', ''), *a.get('pitfalls', []), *a.get('followups', [])):
            math_expressions.extend(re.findall(r'\$([^$\n]+)\$', value))
        check(not any(re.search(r'\\operatorname\b', expression) for expression in math_expressions), f'{qid}: use portable math names such as \\mathrm instead of operatorname')
        for value in (a.get('body', ''), a.get('formula', ''), *a.get('pitfalls', []), *a.get('followups', [])):
            check(not re.search(r'[\x00-\x09\x0b-\x1f]', value), f'{qid}: unexpected control character in answer')
        for figure in q.get('figures', []):
            figure_path = figure.get('path', '')
            check(bool(re.fullmatch(r'assets/[a-z0-9-]+\.svg', figure_path)), f'{qid}: invalid figure path')
            check(bool(figure.get('alt', '').strip()), f'{qid}: missing figure description')
            resolved = (ROOT / figure_path).resolve()
            check(resolved.is_relative_to((ROOT / 'assets').resolve()) and resolved.is_file(), f'{qid}: missing or external figure')
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
