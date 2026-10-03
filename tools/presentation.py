"""Render the same answer as GitHub Markdown and offline HTML with MathML."""
import html
import base64
import re
import sys
from xml.etree import ElementTree
from common import ROOT

local_packages = ROOT / '.cache' / 'math-python'
if local_packages.exists():
    sys.path.insert(0, str(local_packages))
from latex2mathml.converter import convert


def math_html(expression, display=False):
    # latex2mathml supports split, which has the same two-column alignment
    # used by our aligned expressions. Keep the canonical LaTeX unchanged.
    expression = expression.replace(r'\begin{aligned}', r'\begin{split}').replace(r'\end{aligned}', r'\end{split}')
    # Use an equivalent named operator for commands unsupported by latex2mathml.
    expression = expression.replace(r'\arg\max', r'\operatorname{argmax}').replace(r'\arg\min', r'\operatorname{argmin}')
    expression = re.sub(r'\\arg\s*\\?(max|min)\b', lambda match: r'\operatorname{arg' + match[1] + '}', expression)
    rendered = convert(expression, display='block' if display else 'inline')
    ElementTree.fromstring(rendered)
    return rendered


def inline_html(text):
    parts = re.split(r'(`[^`\n]+`|\$[^$\n]+\$)', text)
    result = []
    for part in parts:
        if part.startswith('`') and part.endswith('`'):
            result.append('<code>' + html.escape(part[1:-1]) + '</code>')
        elif part.startswith('$') and part.endswith('$'):
            result.append(math_html(part[1:-1]))
        else:
            result.append(html.escape(part))
    return ''.join(result)


def prose_html(text):
    result = []
    for paragraph in text.split('\n\n'):
        lines = paragraph.splitlines()
        if len(lines) >= 2 and all(line.startswith('|') and line.endswith('|') for line in lines) and re.fullmatch(r'\|(?:\s*:?-+:?\s*\|)+', lines[1]):
            cells = lambda line: [cell.strip() for cell in line.strip('|').split('|')]
            header = '<tr>' + ''.join('<th>' + inline_html(cell) + '</th>' for cell in cells(lines[0])) + '</tr>'
            rows = ''.join('<tr>' + ''.join('<td>' + inline_html(cell) + '</td>' for cell in cells(line)) + '</tr>' for line in lines[2:])
            result.append('<div class="table-scroll"><table><thead>' + header + '</thead><tbody>' + rows + '</tbody></table></div>')
        elif lines and all(line.startswith('- ') for line in lines):
            result.append('<ul>' + ''.join('<li>' + inline_html(line[2:]) + '</li>' for line in lines) + '</ul>')
        else:
            result.append('<p>' + inline_html(' '.join(lines)) + '</p>')
    return ''.join(result)


def answer_html(answer, figures=()):
    result = ['<div class="answer-body">', prose_html(answer['body'])]
    if answer.get('formula'):
        result.append('<div class="formula">' + math_html(answer['formula'], True) + '</div>')
    for figure in figures:
        image_data = base64.b64encode((ROOT / figure['path']).read_bytes()).decode('ascii')
        result.append('<figure><img src="data:image/svg+xml;base64,' + image_data + '" alt="' + html.escape(figure['alt'], quote=True) + '">')
        if figure.get('caption'):
            result.append('<figcaption>' + inline_html(figure['caption']) + '</figcaption>')
        result.append('</figure>')
    for key, title in (('pitfalls', '易错点'), ('followups', '追问')):
        result.append('<h3>' + title + '</h3><ul>' + ''.join('<li>' + inline_html(item) + '</li>' for item in answer[key]) + '</ul>')
    result.append('</div>')
    return ''.join(result)
