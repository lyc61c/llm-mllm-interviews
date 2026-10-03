"""Regression checks for formulas, tables and local diagrams in offline answers."""
import re
import json
import unittest
from xml.etree import ElementTree
from common import ROOT, load_bank
from presentation import answer_html


class PresentationTests(unittest.TestCase):
    def test_generated_answers_match_canonical_data(self):
        _, questions = load_bank()
        html = (ROOT / 'index.html').read_text(encoding='utf-8')
        payload = json.loads(re.search(r'<script id="bank" type="application/json">([\s\S]*?)</script>', html)[1])
        published = {q['id']:q for q in payload['questions']}
        self.assertEqual(set(published), {q['id'] for q in questions})
        for question in questions:
            with self.subTest(question=question['id']):
                self.assertEqual(published[question['id']]['answer_html'], answer_html(question['answer'], question.get('figures', [])))
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
