"""Build concise chapters, the question index and the offline explorer."""
import json
from collections import Counter
from common import ROOT, CATEGORIES, TOPICS, load_bank, validate
from supplement import validate_supplement
from presentation import answer_html


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')


def bullet(items):
    return '\n'.join('- ' + item for item in items)


def main():
    sources, questions = load_bank()
    errors = validate(sources, questions) + validate_supplement(questions, sources)
    if errors:
        raise SystemExit('\n'.join(errors))
    smap = {s['id']: s for s in sources}
    counts = Counter(q['category'] for q in questions)
    home = ['# LLM / MLLM 面试题库', '', f'**{len(questions)} 道题 · {len(CATEGORIES)} 个分类**。每题包含完整答案、必要公式、易错点和追问。', '', '[全部题目](QUESTION-INDEX.md) · [搜索与自测](index.html) · [手撕代码](coding/README.md) · [复习路线](guides/study-plan.md)', '', '| 分类 | 题数 |', '|---|---:|']
    home.extend(f'| [{title}](chapters/{slug}.md) | {counts[cat]} |' for cat, (slug, title) in CATEGORIES.items())
    home.extend(['', 'L1 基础 · L2 进阶 · L3 深入', '', '[使用许可](LICENSE.md)', ''])
    write(ROOT / 'README.md', '\n'.join(home))
    for cat, (slug, title) in CATEGORIES.items():
        rows = [q for q in questions if q['category'] == cat]
        introductions = ('TFM-004', 'TFM-014', 'TFM-015')
        rows.sort(key=lambda q: (0 if q['id'] in introductions else 1, q['id']))
        groups = [(topic, [q for q in rows if q['topic'] == topic]) for topic in TOPICS[cat]]
        groups = [(topic, group) for topic, group in groups if group]
        lines = [f'# {title}', '', '[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)', '', '## 目录', '']
        for n, (topic, group) in enumerate(groups, 1):
            lines.append(f'- [{topic}](#topic-{n})')
            lines.extend(f"  - [{q['id']} · {q['title']}](#{q['id'].lower()})" for q in group)
        for n, (topic, group) in enumerate(groups, 1):
            lines.extend(['', f'<a id="topic-{n}"></a>', f'## {topic}'])
            for q in group:
                a = q['answer']
                lines.extend(['', f'<a id="{q["id"].lower()}"></a>', f"### {q['id']} · {q['title']}", '', f"**{q['level']}**", '', '#### 答案', '', a['body']])
                if a.get('formula'):
                    lines.extend(['', '$$', a['formula'], '$$'])
import json
from collections import Counter
from common import ROOT, CATEGORIES, TOPICS, load_bank, validate
from supplement import validate_supplement
from presentation import answer_html


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')


def bullet(items):
    return '\n'.join('- ' + item for item in items)


def main():
    sources, questions = load_bank()
    errors = validate(sources, questions) + validate_supplement(questions, sources)
    if errors:
        raise SystemExit('\n'.join(errors))
    smap = {s['id']: s for s in sources}
    counts = Counter(q['category'] for q in questions)
    home = ['# LLM / MLLM 面试题库', '', f'**{len(questions)} 道题 · {len(CATEGORIES)} 个分类**。每题包含完整答案、必要公式、易错点和追问。', '', '[全部题目](QUESTION-INDEX.md) · [搜索与自测](index.html) · [手撕代码](coding/README.md) · [复习路线](guides/study-plan.md)', '', '| 分类 | 题数 |', '|---|---:|']
    home.extend(f'| [{title}](chapters/{slug}.md) | {counts[cat]} |' for cat, (slug, title) in CATEGORIES.items())
    home.extend(['', 'L1 基础 · L2 进阶 · L3 深入', '', '[使用许可](LICENSE.md)', ''])
    write(ROOT / 'README.md', '\n'.join(home))
    for cat, (slug, title) in CATEGORIES.items():
        rows = [q for q in questions if q['category'] == cat]
        lines = [f'# {title}', '', '[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)', '', '## 目录', '']
        for n, topic in enumerate(TOPICS[cat], 1):
            members = [q for q in rows if q['topic'] == topic]
            if members:
                lines.append(f'- [{topic}](#topic-{n})')
                lines.extend(f"  - [{q['id']} · {q['title']}](#{q['id'].lower()})" for q in members)
        for n, topic in enumerate(TOPICS[cat], 1):
            members = [q for q in rows if q['topic'] == topic]
            if not members:
                continue
            lines.extend(['', f'<a id="topic-{n}"></a>', f'## {topic}'])
            for q in members:
                a = q['answer']
                lines.extend(['', f'<a id="{q["id"].lower()}"></a>', f"### {q['id']} · {q['title']}", '', f"**{q['level']}**", '', '#### 答案', '', a['body']])
                if a.get('formula'):
                    lines.extend(['', '$$', a['formula'], '$$'])
                for figure in q.get('figures', []):
                    lines.extend(['', f"![{figure['alt']}](../{figure['path']})"])
                    if figure.get('caption'):
                        lines.extend(['', figure['caption']])
                lines.extend(['', '#### 易错点', '', bullet(a['pitfalls']), '', '#### 追问', '', bullet(a['followups'])])
        references = {}
        for q in rows:
            for sid in q['reference_ids']:
                s = smap[sid]
                references.setdefault(s['url'], s['title'])
        lines.extend(['', '## 参考资料', ''])
        lines.extend(f'- [{title}]({url})' for url, title in references.items())
        write(ROOT / 'chapters' / f'{slug}.md', '\n'.join(lines) + '\n')
    index = ['# 题目索引', '', '[首页](README.md) · [搜索与自测](index.html)', '', '| 题号 | 类别 | 主题 | 难度 | 题目 |', '|---|---|---|---|---|']
    for q in questions:
        slug, title = CATEGORIES[q['category']]
        qtitle = q['title'].replace('|', ' / ')
        index.append(f"| {q['id']} | {title} | {q['topic']} | {q['level']} | [{qtitle}](chapters/{slug}.md#{q['id'].lower()}) |")
    write(ROOT / 'QUESTION-INDEX.md', '\n'.join(index) + '\n')
    public_questions = [{key: q[key] for key in ('id', 'category', 'topic', 'level', 'title', 'tags')} | {'answer_html': answer_html(q['answer'], q.get('figures', [])), 'search_text': '\n'.join([q['answer']['body'], *q['answer']['pitfalls'], *q['answer']['followups']])} for q in questions]
    payload = json.dumps({'questions': public_questions, 'categories': CATEGORIES, 'topics': TOPICS}, ensure_ascii=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    template = (ROOT / 'tools' / 'explorer-template.html').read_text(encoding='utf-8')
    write(ROOT / 'index.html', template.replace('__BANK_JSON__', payload))
    print(json.dumps({'questions': len(questions), 'categories': dict(counts)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
