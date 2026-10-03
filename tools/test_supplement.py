"""Guard coverage and the distinction between a user paste and a read webpage."""
import copy
import unittest
from unittest.mock import patch
from common import load_bank
from supplement import load_request, validate_supplement


class SupplementCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = load_bank()[1]
        cls.request = load_request()

    def validate_with(self, request, questions=None):
        with patch('supplement.load_request', return_value=request):
            return validate_supplement(self.questions if questions is None else questions)

    def test_real_bank_covers_both_supplied_lists(self):
        self.assertEqual(len(self.request['items']), 15)
        self.assertEqual(len(self.request['article']['items']), 30)
        self.assertEqual(self.validate_with(self.request), [])

    def test_missing_original_question_is_reported(self):
        questions = copy.deepcopy(self.questions)
        for question in questions:
            question['article_supplement_keys'] = [
                key for key in question.get('article_supplement_keys', []) if key != 'Z27'
            ]
        self.assertTrue(any('Z27:' in error for error in self.validate_with(self.request, questions)))

    def test_blocked_webpage_cannot_claim_direct_merge(self):
        request = copy.deepcopy(self.request)
        request['article']['status'] = 'merged'
        self.assertTrue(any('blocked article' in error for error in self.validate_with(request)))

    def test_user_paste_must_disclose_unverified_webpage(self):
        for change in ({'origin': 'webpage'}, {'original_page_verified': True}):
            with self.subTest(change=change):
                request = copy.deepcopy(self.request)
                request['article']['text_provenance'].update(change)
                self.assertTrue(any('user-text merge' in error for error in self.validate_with(request)))

    def test_bad_metadata_returns_errors_instead_of_crashing(self):
        for value in (None, ['Z01', {}], ['Z01', 'Z01'], ['Z99']):
            with self.subTest(value=value):
                questions = copy.deepcopy(self.questions)
                questions[0]['article_supplement_keys'] = value
                self.assertTrue(self.validate_with(self.request, questions))

    def test_bad_request_structure_returns_errors(self):
        for field, value in (('text_provenance', None), ('items', None), ('items', [None])):
            with self.subTest(field=field, value=value):
                request = copy.deepcopy(self.request)
                request['article'][field] = value
                self.assertTrue(self.validate_with(request))

    def test_thirty_keys_cannot_replace_an_original_question(self):
        request = copy.deepcopy(self.request)
        request['article']['items'][0]['key'] = 'Z99'
        questions = copy.deepcopy(self.questions)
        for question in questions:
            question['article_supplement_keys'] = [
                'Z99' if key == 'Z01' else key for key in question.get('article_supplement_keys', [])
            ]
        self.assertTrue(any('exactly Z01' in error for error in self.validate_with(request, questions)))

    def test_request_cannot_upgrade_catalog_access(self):
        request = copy.deepcopy(self.request)
        request['article'].update(access='full', status='merged')
        self.assertTrue(any('source catalog' in error for error in self.validate_with(request)))


if __name__ == '__main__':
    unittest.main()
