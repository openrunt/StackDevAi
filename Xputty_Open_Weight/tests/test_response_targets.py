import pytest
from vox.response_targets import rolling_assistant_examples


class Tokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        role = {'system': 1, 'user': 2, 'assistant': 3}
        ids = []
        for m in messages: ids.extend([role[m['role']], *map(ord, m['content']), 4])
        return ids + ([3] if add_generation_prompt else [])


def test_all_replies_keep_complete_targets_and_previous_context():
    messages = [{'role': 'system', 'content': 'Scene'}, {'role': 'user', 'content': 'Question'},
                {'role': 'assistant', 'content': 'Answer'}, {'role': 'user', 'content': 'Followup'},
                {'role': 'assistant', 'content': 'Respond'}]
    rows = rolling_assistant_examples(Tokenizer(), messages, 80)
    assert len(rows) == 2
    x, y = rows[-1]
    assert x.tolist()[0][-1] == 4 and y.tolist()[0][-1] == 4
    assert sum(v != -100 for v in y.tolist()[0]) == len('Respond') + 1
    assert x.tolist()[0].count(3) == 2, 'The second target must see the earlier assistant reply'
    assert all(v == -100 for v in y.tolist()[0][:len(x[0]) - len('Respond') - 1])


def test_rolling_drops_whole_old_pairs_and_never_clips_targets():
    messages = [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'OLD' * 10},
                {'role': 'assistant', 'content': 'reply'}, {'role': 'user', 'content': 'new'},
                {'role': 'assistant', 'content': 'done'}]
    rows = rolling_assistant_examples(Tokenizer(), messages, 20)
    assert len(rows) == 1
    x, y = rows[0]
    assert x.shape[1] <= 20 and x.tolist()[0][:2] == [1, ord('S')]
    assert y.tolist()[0][-5:] == [*map(ord, 'done'), 4]
    with pytest.raises(ValueError, match='alternating'):
        rolling_assistant_examples(Tokenizer(), [{'role': 'assistant', 'content': 'wrong'}, {'role': 'assistant', 'content': 'end'}], 20)


def test_social_source_roles_follow_actual_speakers():
    from tools.prepare_social import normalize
    row = {'dialogue': ['Hi there.', 'Hello there.', 'What happened?', 'A small problem.', 'Tell me.', 'My radio broke.', 'That is frustrating.', 'Can you help?', 'Yes please.'],
           'speakers': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'B']}
    messages = normalize(row)
    assert messages[0]['role'] == 'system' and 'Portray B' in messages[0]['content']
    assert messages[-1] == {'role': 'assistant', 'content': 'Can you help?\nYes please.'}
    assert normalize({**row, 'speakers': row['speakers'][:-1] + ['C']}) is None


def test_actor_chat_history_distinguishes_self_from_other_hosts():
    from vox.dialogue_policy import cast_messages
    events = [{'name': 'Lyra', 'text': 'A question.'}, {'name': 'Orion', 'text': 'My answer.'},
              {'name': 'Nova', 'text': 'Another angle.'}, {'name': 'Kai', 'text': 'A practical point.'}]
    messages = cast_messages(events, 'Orion')
    assert messages == [{'role': 'user', 'content': 'Lyra: A question.'},
                        {'role': 'assistant', 'content': 'My answer.'},
                        {'role': 'user', 'content': 'Nova: Another angle.\nKai: A practical point.'}]
    with pytest.raises(ValueError, match='another speaker'):
        cast_messages(events, 'Kai')
