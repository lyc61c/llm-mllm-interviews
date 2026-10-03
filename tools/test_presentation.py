"""Regression checks for formulas, tables and local diagrams in offline answers."""
import re
import json
import copy
import unittest
from xml.etree import ElementTree
from common import ROOT, load_bank, validate
from presentation import answer_html, math_html, markdown_prose, markdown_formula


class PresentationTests(unittest.TestCase):
    def test_inline_math_is_protected_without_changing_code(self):
        text = r'$Q=XW_Q$、$K=XW_K$，$D_{\mathrm{KL}}(p\|q)$，`$literal$`'
        self.assertEqual(markdown_prose(text), r'$`Q=XW_Q`$、$`K=XW_K`$，$`D_{\mathrm{KL}}(p\|q)`$，`$literal$`')

    def test_block_math_avoids_html_tags_and_extra_line_breaks(self):
        expression = r'\begin{aligned}p(y_t\mid y_{<t})&>0\\' + '\n' + r'q&=1\end{aligned}'
        protected = markdown_formula(expression)
        self.assertEqual(protected, r'\begin{aligned}p(y_t\mid y_{\lt t})&\gt 0\\ q&=1\end{aligned}')
        self.assertEqual(math_html(expression, True), math_html(protected, True))

    def test_chapter_math_keeps_canonical_tex(self):
        from collections import Counter
        from common import CATEGORIES
        _, questions = load_bank()
        for category, (slug, _) in CATEGORIES.items():
            chapter = (ROOT / 'chapters' / (slug+'.md')).read_text(encoding='utf-8')
            blocks = re.findall(r'```math\n([\s\S]*?)\n```', chapter)
            inline = re.findall(r'\$`([^`\n]+)`\$', chapter)
            expected_blocks, expected_inline = [], []
            for question in questions:
                if question['category'] != category:
                    continue
                answer = question['answer']
                if answer.get('formula'):
                    expected_blocks.append(answer['formula'].replace('\n', ' ').replace('<', r'\lt ').replace('>', r'\gt '))
                for value in [answer['body'], *answer['pitfalls'], *answer['followups']]:
                    expected_inline.extend(re.findall(r'\$([^$\n]+)\$', value))
            with self.subTest(category=category):
                self.assertEqual(Counter(blocks), Counter(expected_blocks))
                self.assertEqual(Counter(inline), Counter(expected_inline))

    def test_blocked_macro_is_rejected_before_publication(self):
        sources, questions = load_bank()
        question = copy.deepcopy(next(q for q in questions if q['id'] == 'BAS-001'))
        question['answer']['formula'] = r'D(i)=\operatorname{DAG}(i)'
        self.assertTrue(any('operatorname' in error for error in validate(sources, [question], require_complete=False)))

    def test_formulas_avoid_blocked_markdown_macro(self):
        _, questions = load_bank()
        for question in questions:
            with self.subTest(question=question['id']):
                answer = question['answer']
                values = [answer['body'], answer.get('formula', ''), *answer['pitfalls'], *answer['followups']]
                self.assertFalse(any(re.search(r'\\operatorname\b', value) for value in values))
        rendered = math_html(r'D(i)=\max_{j\in\mathrm{DAG}(i)}\{\log P(x_{i:j})+D(j+1)\}')
        self.assertIn('DAG', ''.join(ElementTree.fromstring(rendered).itertext()))
        for expression in (r'\arg\max_a Q(a)', r'\arg\min_x f(x)'):
            self.assertNotIn('\\', ''.join(ElementTree.fromstring(math_html(expression)).itertext()))

    def test_generated_answers_match_canonical_data(self):
        _, questions = load_bank()
        html = (ROOT / 'index.html').read_text(encoding='utf-8')
        payload = json.loads(re.search(r'<script id="bank" type="application/json">([\s\S]*?)</script>', html)[1])
        published = {q['id']:q for q in payload['questions']}
        self.assertEqual(set(published), {q['id'] for q in questions})
        for question in questions:
            with self.subTest(question=question['id']):
                self.assertEqual(published[question['id']]['answer_html'], answer_html(question['answer'], question.get('figures', []), question.get('code_links', [])))
                self.assertEqual(published[question['id']]['topic'], question['topic'])

    def test_all_math_renders_without_literal_commands(self):
        _, questions = load_bank()
        for question in questions:
            with self.subTest(question=question['id']):
                rendered = answer_html(question['answer'], question.get('figures', []))
                for fragment in re.findall(r'<math\b[\s\S]*?</math>', rendered):
                    root = ElementTree.fromstring(fragment)
                    self.assertNotIn('\\', ''.join(root.itertext()))

    def test_training_stage_table_is_html(self):
        _, questions = load_bank()
        question = next(q for q in questions if q['id'] == 'VLM-028')
        rendered = answer_html(question['answer'])
        self.assertIn('<table>', rendered)
        self.assertEqual(rendered.count('<th>'), 4)
        self.assertEqual(rendered.count('<td>'), 16)

    def test_diagrams_are_embedded_for_offline_reading(self):
        _, questions = load_bank()
        diagrams = [q for q in questions if q.get('figures')]
        self.assertEqual(len(diagrams), 6)
        for question in diagrams:
            rendered = answer_html(question['answer'], question['figures'])
            self.assertIn('data:image/svg+xml;base64,', rendered)
            self.assertNotIn('<img src="http', rendered)


if __name__ == '__main__':
    unittest.main()
