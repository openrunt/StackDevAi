"""Complete assistant targets from every dialogue exchange, with whole-pair rolling context."""


def rolling_assistant_examples(tokenizer, messages, context=384):
    import torch
    if not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('A training dialogue must end with a known assistant response')
    system = [messages[0]] if messages[0]['role'] == 'system' else []
    dialogue = messages[len(system):]
    if any(m['role'] != ('user' if i % 2 == 0 else 'assistant') for i, m in enumerate(dialogue)):
        raise ValueError('Known alternating dialogue roles are required')
    result = []
    for index in range(1, len(dialogue), 2):
        history = list(dialogue[:index])
        response = dialogue[index]
        while True:
            prefix = tokenizer.apply_chat_template(system + history, tokenize=True, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(system + history + [response], tokenize=True, add_generation_prompt=False)
            if full[:len(prefix)] != prefix:
                raise ValueError('Training chat template must be prefix-stable')
            if len(full) <= context:
                labels = [-100] * len(prefix) + full[len(prefix):]
                result.append((torch.tensor([full]), torch.tensor([labels])))
                break
            if len(history) <= 1:
                # Never teach an unfinished response or a left-truncated question.
                break
            del history[:2]
    return result
