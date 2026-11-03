"""Lossless research views. They are not semantic paraphrases or training targets."""
import copy
import hashlib
import json
import re

PHASES = ('canonical', 'characters', 'words', 'turns')


def identity(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def transform(record, phase):
    if phase not in PHASES: raise ValueError('Unknown reversible phase')
    result = copy.deepcopy(record)
    messages = result.get('messages')
    if not isinstance(messages, list) or len(messages) < 2 or any(not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant') or not isinstance(m.get('content'), str) for m in messages):
        raise ValueError('Expected text chat messages')
    if messages[-1]['role'] != 'assistant': raise ValueError('The last assistant target must remain separate')
    offset = 1 if messages[0]['role'] == 'system' else 0
    if any(m['role'] == 'system' for m in messages[offset:]): raise ValueError('System context must be first')
    if phase == 'turns':
        messages[offset:-1] = reversed(messages[offset:-1])
    elif phase in ('characters', 'words'):
        for message in messages[offset:-1]:
            text = message['content']
            message['content'] = text[::-1] if phase == 'characters' else ''.join(reversed(re.findall(r'\s+|\S+', text)))
    return result


def make_view(record, phase):
    view = {'schema': 'xow-reversible-context/1', 'phase': phase,
            'canonical_sha256': identity(record), 'payload': transform(record, phase)}
    if restore(view) != record: raise AssertionError('Phase did not preserve the canonical record')
    return view


def restore(view):
    if view.get('schema') != 'xow-reversible-context/1': raise ValueError('Unknown view format')
    restored = transform(view['payload'], view['phase'])
    if identity(restored) != view['canonical_sha256']: raise ValueError('View recovery checksum mismatch')
    return restored
