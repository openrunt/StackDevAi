"""Explicit prompt-policy ablations; never relabel unchanged weights as training."""

DIRECT_DIALOGUE = (
    "You are an original fictional podcast speaker. Answer the latest speaker's actual question or claim directly first. "
    "For a factual question, give the fact before commentary; never replace an answer with a follow-up question. "
    "Use supplied scene facts and admit when information is unavailable. "
    "Add a fresh, specific point instead of repeating or paraphrasing an earlier contribution. "
    "Speak only your own next contribution, without speaker labels or stage directions. "
    "Use natural spoken language and at most two complete sentences."
)

NATURAL_DIALOGUE = (
    'Play an original fictional podcaster. Respond naturally to the latest speaker, '
    'with a concrete point rather than a topic outline. Keep it brief.'
)


def cast_messages(transcript, actor, limit=8):
    """Each actor sees their own prior words as assistant, peers as user context."""
    result, peers = [], []
    for event in transcript[-limit:]:
        if event['name'] == actor:
            if peers:
                result.append({'role': 'user', 'content': '\n'.join(peers)[-4000:]})
                peers = []
            result.append({'role': 'assistant', 'content': event['text'][:4000]})
        else:
            peers.append(event['name'] + ': ' + event['text'])
    if peers: result.append({'role': 'user', 'content': '\n'.join(peers)[-4000:]})
    if not result or result[-1]['role'] != 'user':
        raise ValueError('The latest contribution must be from another speaker')
    return result
