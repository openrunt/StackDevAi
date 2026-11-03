import json

import pytest

from vox.podcast_cast_curriculum import EPISODES, prepare, records_for_episode
from vox.public_baseline import conversation_splits
from vox.response_targets import rolling_assistant_examples


class Tokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        roles = {'system': 1, 'user': 2, 'assistant': 3}
        ids = []
        for message in messages:
            ids.extend([roles[message['role']], *map(ord, message['content']), 4])
        return ids + ([3] if add_generation_prompt else [])


def test_actor_views_keep_self_and_peer_roles_with_complete_targets():
    assert len(EPISODES) == 24 and len({row[0] for row in EPISODES}) == 24
    assert len({row[3] for row in EPISODES}) == 4
    for topic, split, premise, order, replies in EPISODES:
        rows, events = records_for_episode(topic, split, premise, order, replies)
        assert len(rows) == 4 and len({r['id'] for r in rows}) == 4
        for row in rows:
            actor = row['actor'].capitalize()
            expected = [event['text'] for event in events if event['name'] == actor]
            assert [m['content'] for m in row['messages'] if m['role'] == 'assistant'] == expected
            assert premise in row['messages'][0]['content']
            examples = rolling_assistant_examples(Tokenizer(), row['messages'], 8192)
            assert len(examples) == 2
            assert all(int(labels[0, -1]) == 4 for _, labels in examples)


def test_frozen_curriculum_holds_entire_topics_out_and_preserves_originals(tmp_path):
    output = tmp_path / 'cast-data'
    manifest = prepare(output)
    groups = conversation_splits(output)
    assert {key: len(value) for key, value in groups.items()} == {'train': 64, 'val': 16, 'test': 16}
    topics = [set(row['topic'] for row in rows) for rows in groups.values()]
    assert all(not a & b for i, a in enumerate(topics) for b in topics[i + 1:])
    originals = json.loads((output / 'original-cast-exchanges.json').read_text())
    assert len(originals) == 24 and all(len(row['events']) == 9 for row in originals)
    assert manifest['timing'] == 'No audio or fabricated timestamps'
    before = (output / 'conversations.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        prepare(output)
    assert (output / 'conversations.jsonl').read_bytes() == before
