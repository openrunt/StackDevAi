"""Actual ChatML dialogue roles, rather than a transcript embedded as a task outline."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.probe_podcast import request
from vox.dialogue_policy import NATURAL_DIALOGUE, cast_messages
from vox.podcast_distill import PERSONAS
from vox.public_baseline import sha256, write_json


def run(tag, weights, output, disable_thinking=False):
    digest = sha256(weights)
    if digest not in request('show', {'model': tag})['modelfile']: raise ValueError('Wrong native weights')
    transcript = [{'name': 'Producer', 'text': 'Fast zombies are always scarier than slow ones. With slow zombies I could just walk away.'}]
    for i, profile in enumerate(list(PERSONAS.values()) * 2):
        messages = [{'role': 'system', 'content': NATURAL_DIALOGUE + f" Your name is {profile['name']}; you are {profile['traits']}."}]
        messages += cast_messages(transcript, profile['name'])
        payload = {'model': tag, 'messages': messages, 'stream': False, 'keep_alive': '10m',
                   'options': {'num_ctx': 2048, 'num_predict': 96, 'temperature': 0.6, 'top_p': 0.9, 'repeat_penalty': 1.05,
                               'seed': 3181 + i, 'num_thread': 2, 'num_gpu': 0}}
        if disable_thinking: payload['think'] = False
        result = request('chat', payload)
        row = {'name': profile['name'], 'text': result['message']['content'], 'done_reason': result.get('done_reason')}
        transcript.append(row)
        print(json.dumps(row), flush=True)
    report = {'schema': 'xow-cast-chat-probe/1', 'model': tag, 'weights_sha256': digest, 'transcript': transcript,
              'system': NATURAL_DIALOGUE, 'options': payload['options'], 'protocol': 'actual actor-perspective chat roles',
              'thinking_explicitly_disabled': disable_thinking,
              'limitations': ['Small development probe, not an independent population estimate',
                             'Prompt/protocol change; do not attribute differences solely to weight adaptation',
                             'Fixed test order does not implement the production response-pool scheduler']}
    write_json(output, report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--weights', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--disable-thinking', action='store_true')
    run(**vars(p.parse_args()))
