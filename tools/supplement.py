"""Trace explicitly supplied questions to merged canonical answers."""
import json
from collections import defaultdict
from common import ROOT, load_bank

REQUEST = ROOT / 'research' / 'supplement-request-2026-10-02.json'


def load_request():
    return json.loads(REQUEST.read_text(encoding='utf-8'))


def mapped_questions(questions, field='user_supplement_keys'):
    result = defaultdict(list)
    for question in questions:
        keys = question.get(field, [])
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            continue
        for key in keys:
            result[key].append(question)
    return result


def validate_supplement(questions, sources=None):
    request = load_request()
    errors = []
    if not isinstance(request, dict):
        return ['supplement request must be an object']
    article = request.get('article')
    if not isinstance(article, dict):
        return ['requested article must be an object']
    def valid_items(items, label):
        if not isinstance(items, list):
            errors.append(f'{label} must be a list of question objects')
            return False
        valid = True
        for item in items:
            if (not isinstance(item, dict) or not isinstance(item.get('key'), str)
                    or not isinstance(item.get('question'), str) or not item['question'].strip()):
                errors.append(f'{label} entries require string key and nonempty question')
                valid = False
        return valid
    user_items = request.get('items')
    article_items = article.get('items', [])
    if not valid_items(user_items, 'user items') or not valid_items(article_items, 'article items'):
        return errors
    def check_mapping(items, field):
        expected = {item['key'] for item in items}
        if len(expected) != len(items):
            errors.append(f'{field}: request has duplicate keys')
        for question in questions:
            keys = question.get(field, [])
            if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
                errors.append(f"{question['id']}: {field} must be a list of strings")
            elif len(keys) != len(set(keys)):
                errors.append(f"{question['id']}: repeated {field}")
        mapping = mapped_questions(questions, field)
        for key in expected:
            if not mapping[key]:
                errors.append(f'{key}: supplied question has no canonical answer')
        for key in mapping.keys() - expected:
            errors.append(f'{key}: question references an unknown supplied question')
    check_mapping(user_items, 'user_supplement_keys')
    if sources is None:
        sources = load_bank()[0]
    source = next((s for s in sources if s['id'] == article.get('source_id')), None)
    if source is None:
        errors.append('requested article has no matching source record')
    elif source.get('url') != article.get('url') or source.get('access') != article.get('access'):
        errors.append('requested article URL/access disagrees with source catalog')
    status = article.get('status')
    if status not in {'pending_source_text', 'merged', 'merged_from_user_text'}:
        errors.append('invalid requested article status')
    if article.get('access') == 'blocked' and (status == 'merged' or (article.get('items') and status != 'merged_from_user_text')):
        errors.append('blocked article must not claim directly extracted questions')
    if status == 'pending_source_text' and article.get('items'):
        errors.append('unread article must not claim extracted questions')
    if status in {'merged', 'merged_from_user_text'} and not article.get('items'):
        errors.append('merged article must retain its actual question mapping')
    if status == 'merged_from_user_text':
        provenance = article.get('text_provenance', {})
        if not isinstance(provenance, dict):
            errors.append('user-text provenance must be an object')
            provenance = {}
        if provenance.get('origin') != 'user_message' or provenance.get('original_page_verified') is not False:
            errors.append('user-text merge must disclose origin and unverified original page')
        if len(article_items) != 30 or provenance.get('question_count') != 30:
            errors.append('core30 user-text merge must retain exactly 30 supplied questions')
        if {item['key'] for item in article_items} != {f'Z{i:02d}' for i in range(1, 31)}:
            errors.append('core30 keys must be exactly Z01 through Z30')
    check_mapping(article_items, 'article_supplement_keys')
    return errors
