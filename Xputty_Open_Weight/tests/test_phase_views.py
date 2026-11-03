import copy
import json
import pytest
from vox.phase_views import PHASES, make_view, restore
from tools.prepare_phase_views import prepare


def example():
    return {'id': 'one', 'episode_id': 'first', 'split': 'train', 'messages': [
        {'role': 'system', 'content': 'Context.'},
        {'role': 'user', 'content': '  Lyra: café 👩🏽‍🚀  isn’t  empty.\n', 'at_ms': 125},
        {'role': 'assistant', 'content': 'No, Orion left it.', 'speaker': 'Kai'},
        {'role': 'user', 'content': 'Where is the cup?'},
        {'role': 'assistant', 'content': 'Under the lamp.'}]}


def test_lossless_views_preserve_target_system_and_metadata():
    record = example(); original = copy.deepcopy(record)
    for phase in PHASES:
        view = make_view(record, phase)
        assert restore(json.loads(json.dumps(view, ensure_ascii=False))) == record
        assert view['payload']['messages'][-1] == record['messages'][-1]
        assert view['payload']['messages'][0] == record['messages'][0]
        view['payload']['messages'][-1]['content'] = 'Changed answer'
        with pytest.raises(ValueError, match='checksum'): restore(view)
    assert record == original


def test_phase_package_retains_holdouts_and_rejects_leakage(tmp_path):
    one = example(); two = example(); two.update(id='two', episode_id='second', split='test')
    two['messages'][1]['content'] = 'A different conversation'
    source = tmp_path / 'source.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in [one, two]), encoding='utf-8')
    result = prepare(source, tmp_path / 'views')
    assert result['views'] == 8 and result['split_counts'] == {'train': 1, 'val': 0, 'test': 1}
    with pytest.raises(ValueError, match='new output'): prepare(source, tmp_path / 'views')
    two['episode_id'] = one['episode_id']
    source.write_text('\n'.join(json.dumps(r) for r in [one, two]), encoding='utf-8')
    with pytest.raises(ValueError, match='holdout'): prepare(source, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
