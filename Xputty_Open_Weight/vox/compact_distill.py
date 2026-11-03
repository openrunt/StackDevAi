"""Actually reduce decoder depth, then compare CE with teacher-guided recovery.

The teacher remains immutable. This is dense LLaMA-compatible compression, not
an attention-mask experiment or a claim of retained general knowledge.
"""
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

from .public_baseline import conversation_splits, sha256, write_json
from .public_control import atomic_torch_save, check_example_separation
from .run_lock import single_writer


def layer_indices(count, keep):
    if type(keep) is not int or not 1 <= keep <= count: raise ValueError('Invalid retained layer count')
    return [count - 1] if keep == 1 else [round(i * (count - 1) / (keep - 1)) for i in range(keep)]


def make_student(teacher, keep):
    from transformers import AutoModelForCausalLM
    if teacher.config.model_type != 'llama': raise ValueError('Only dense LLaMA-family models are supported')
    indices = layer_indices(len(teacher.model.layers), keep)
    config = copy.deepcopy(teacher.config); config.num_hidden_layers = keep
    student = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    source = teacher.state_dict(); mapped = {}
    for name in student.state_dict():
        parts = name.split('.')
        if name.startswith('model.layers.'):
            parts[2] = str(indices[int(parts[2])])
        mapped[name] = source['.'.join(parts)]
    student.load_state_dict(mapped, strict=True)
    return student, indices


def target_logits(model, inputs, labels):
    positions = (labels[0, 1:] != -100).nonzero().flatten()
    if not len(positions): raise ValueError('No prediction targets')
    hidden = model.model(input_ids=inputs[:, :-1], use_cache=False).last_hidden_state[0, positions]
    return model.lm_head(hidden).float(), labels[0, positions + 1]


def coarse_teacher(logits, top_k, temperature):
    import torch
    logp = torch.log_softmax(logits.float() / temperature, dim=-1)
    values, indices = logp.topk(min(top_k, logp.shape[-1]), dim=-1)
    tail = (1 - values.exp().sum(-1)).clamp_min(0)
    with torch.inference_mode(False), torch.no_grad():
        return {'indices': indices.cpu().clone(), 'logp': values.cpu().clone(), 'tail': tail.cpu().clone()}


def teacher_kl(logits, cached, temperature):
    """KL on top-k teacher categories plus ONE aggregated other-token category."""
    import torch
    logp = torch.log_softmax(logits.float() / temperature, dim=-1)
    chosen = logp.gather(-1, cached['indices'])
    p = cached['logp'].exp()
    top = (p * (cached['logp'] - chosen)).sum(-1)
    if chosen.shape[-1] == logp.shape[-1]: return top.mean() * temperature**2
    tail = cached['tail']
    student_tail = (1 - chosen.exp().sum(-1)).clamp_min(1e-7)
    other = torch.where(tail > 0, tail * (tail.clamp_min(1e-30).log() - student_tail.log()), torch.zeros_like(tail))
    return (top + other).mean() * temperature**2


def examples_for(tokenizer, dataset, context, train_samples, eval_samples):
    from .response_targets import rolling_assistant_examples
    groups = conversation_splits(dataset); examples = {}
    for split, rows in groups.items():
        all_examples = [e for row in rows for e in rolling_assistant_examples(tokenizer, row['messages'], context)]
        examples[split] = all_examples[:train_samples if split == 'train' else eval_samples]
    if min(map(len, examples.values())) < 2: raise ValueError('Too few complete targets in a partition')
    check_example_separation(examples['train'], examples['val'], examples['test'])
    return examples


@single_writer
def prepare_teacher_cache(teacher, examples, identity, output, temperature, top_k):
    import torch
    from .limits import memory_guard
    output = Path(output); manifest = output.with_suffix('.json')
    if output.exists():
        if not manifest.exists() or json.loads(manifest.read_text())['sha256'] != sha256(output):
            raise ValueError('Teacher cache integrity check failed')
        cache = torch.load(output, map_location='cpu', weights_only=True)
        if cache['identity'] != identity: raise ValueError('Teacher cache inputs changed')
        return cache
    cache = {'identity':identity,'targets':{},'teacher_nll':{}}
    began = time.perf_counter()
    with torch.inference_mode():
        for split, rows in examples.items():
            cached,total,tokens=[],0.,0
            for i,(inputs,labels) in enumerate(rows):
                logits,targets=target_logits(teacher,inputs,labels)
                total+=float(torch.nn.functional.cross_entropy(logits,targets,reduction='sum'));tokens+=len(targets)
                cached.append(coarse_teacher(logits,top_k,temperature))
                if i%16==0:print(json.dumps({'phase':'teacher_targets','split':split,'completed':i+1,'rss':memory_guard(4)}),flush=True)
            cache['targets'][split]=cached;cache['teacher_nll'][split]=total/tokens
    cache['preparation_seconds']=time.perf_counter()-began
    output.parent.mkdir(parents=True,exist_ok=True);atomic_torch_save(torch,cache,output)
    write_json(manifest,{'sha256':sha256(output),'identity':identity})
    return cache


@single_writer
def train(base, dataset, output, cache_file, steps=128, keep=5, alpha=.5, temperature=2., top_k=64,
          context=384, train_samples=128, eval_samples=16, eval_every=16, resume=False):
    from .limits import limit_cpu, memory_guard, environment
    torch = limit_cpu(2)
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as hf_version
    if not 1 <= steps <= 10000 or not 0 <= alpha <= 1 or not 0 < temperature <= 10 or not 1 <= top_k <= 256:
        raise ValueError('Invalid distillation budget')
    if context not in (128,256,384,512) or not 1 <= train_samples <= 2048 or not 1 <= eval_samples <= 64 or not 1 <= eval_every <= 100:
        raise ValueError('Invalid context/example budget')
    base, dataset, output, cache_file = map(Path, (base, dataset, output, cache_file))
    if output.exists() and not resume: raise FileExistsError('Choose a new compact-model run')
    if resume and not (output/'checkpoint.pt').exists(): raise ValueError('No compact checkpoint to resume')
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    examples = examples_for(tokenizer, dataset, context, train_samples, eval_samples)
    inputs_hash = hashlib.sha256(b''.join(x.numpy().tobytes()+y.numpy().tobytes() for rows in examples.values() for x,y in rows)).hexdigest()
    cache_identity = {'base_weights': sha256(base/'model.safetensors'), 'config':sha256(base/'config.json'),
                      'tokenizer':sha256(base/'tokenizer.json'), 'template':hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
                      'examples':inputs_hash,'temperature':temperature,'top_k':top_k,'code':sha256(Path(__file__)),
                      'torch':str(torch.__version__),'transformers':hf_version}
    cache_identity['dependencies']={name:sha256(Path(__file__).with_name(name)) for name in ('response_targets.py','public_baseline.py','public_control.py')}
    teacher = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, trust_remote_code=False,
                                                use_safetensors=True, torch_dtype=torch.float32).eval()
    if teacher.config.attention_dropout: raise ValueError('This deterministic pilot requires zero attention dropout')
    teacher.requires_grad_(False)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    cache=prepare_teacher_cache(teacher,examples,cache_identity,cache_file,temperature,top_k)
    torch.manual_seed(739)
    student, selected_layers = make_student(teacher, keep)
    del teacher; gc.collect()
    student.requires_grad_(True); student.model.embed_tokens.requires_grad_(False); student.lm_head.requires_grad_(False)
    trainable = {n:p for n,p in student.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(),lr=.00005,weight_decay=.01)
    identity = {**cache_identity,'keep':selected_layers,'alpha':alpha,'lr':.00005,'eval_every':eval_every,'seed':739,
                'cache_sha256':sha256(cache_file),'train_samples':train_samples,'eval_samples':eval_samples,'context':context}
    output.mkdir(parents=True,exist_ok=True)
    if resume:
        if json.loads((output/'run-config.json').read_text()) != identity: raise ValueError('Compact resume configuration changed')
        state=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=True)
        if steps < state['step']: raise ValueError('Total updates cannot precede the saved step')
        with torch.no_grad():
            for n,p in trainable.items():p.copy_(state['weights'][n])
        optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
        (output/'stop-request').unlink(missing_ok=True)
    else:
        write_json(output/'run-config.json',identity);shutil.copy2(Path(__file__),output/'TRAINER-SNAPSHOT.py')
        state={'step':0,'best_step':0,'best_loss':float('inf'),'best':{},'history':[]}
    def weights():return {n:p.detach().cpu().clone() for n,p in trainable.items()}
    def score(split):
        total=divergence=tokens=0;student.eval()
        with torch.inference_mode():
            for i,(x,y) in enumerate(examples[split]):
                logits,target=target_logits(student,x,y);n=len(target)
                total+=float(torch.nn.functional.cross_entropy(logits,target,reduction='sum'))
                divergence+=float(teacher_kl(logits,cache['targets'][split][i],temperature))*n;tokens+=n
        return {'assistant_nll':total/tokens,'temperature_scaled_teacher_kl':divergence/tokens,'target_tokens':tokens}
    before=score('val')
    if not resume:state.update(best_loss=before['assistant_nll'],best=weights(),before=before)
    began=time.perf_counter()
    for step in range(state['step'],steps):
        if (output/'stop-request').exists():break
        index=random.Random(739000003+step).randrange(len(examples['train']))
        x,y=examples['train'][index];student.train();optimizer.zero_grad(set_to_none=True)
        logits,target=target_logits(student,x,y)
        ce=torch.nn.functional.cross_entropy(logits,target)
        kd=teacher_kl(logits,cache['targets']['train'][index],temperature) if alpha else logits.new_tensor(0.)
        loss=(1-alpha)*ce+alpha*kd;loss.backward();torch.nn.utils.clip_grad_norm_(trainable.values(),1.);optimizer.step()
        state['step']=step+1;memory_guard(4)
        if state['step']%eval_every==0 or state['step']==steps or (output/'stop-request').exists():
            value=score('val')
            if value['assistant_nll']<state['best_loss']:state.update(best_loss=value['assistant_nll'],best_step=state['step'],best=weights())
            row={'step':state['step'],'ce':float(ce.detach()),'kd':float(kd.detach()),'validation':value,'training_seconds':time.perf_counter()-began}
            state['history'].append(row);print(json.dumps(row),flush=True)
            atomic_torch_save(torch,{**state,'weights':weights(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state()},output/'checkpoint.pt')
    if not state['history'] or state['history'][-1]['step'] != state['step']:
        value=score('val')
        if value['assistant_nll']<state['best_loss']:state.update(best_loss=value['assistant_nll'],best_step=state['step'],best=weights())
        state['history'].append({'step':state['step'],'validation':value,'safe_stop':True})
    atomic_torch_save(torch,{**state,'weights':weights(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state()},output/'checkpoint.pt')
    with torch.no_grad():
        for name,p in trainable.items():p.copy_(state['best'][name])
    after,testing=score('val'),score('test')
    exported=output/'export';student.save_pretrained(exported,safe_serialization=True);tokenizer.save_pretrained(exported)
    for name in ('LICENSE','README.md'):
        if (base/name).exists():shutil.copy2(base/name,exported/('UPSTREAM-MODEL-CARD.md' if name=='README.md' else name))
    for name in ('public-sources.json','SODA-DATASET-CARD.md'):
        if (dataset/name).exists():shutil.copy2(dataset/name,exported/name)
    root=Path(__file__).resolve().parents[1];shutil.copy2(root/'export_chat.py',exported/'chat.py');shutil.copy2(root/'LICENSE',exported/'APPLICATION-LICENSE.txt')
    params=sum(p.numel() for p in student.parameters())
    source_metadata=json.loads((dataset/'public-sources.json').read_text(encoding='utf-8')) if (dataset/'public-sources.json').exists() else {}
    report={'name':f'Xputty Compact {params/1e6:.1f}M · '+('teacher-guided' if alpha else 'CE control'),
            'initialization':'Pretrained SmolLM2, selected decoder layers retained; not random scratch weights',
            'instruction_tuned':True,'parameter_count':params,'teacher_parameter_count':teacher_params,
            'selected_layers':selected_layers,'trained_parameters':sum(p.numel() for p in trainable.values()),
            'completed_updates':state['step'],'selected_update':state['best_step'],'before_validation':state['before'],
            'selected_validation':after,'final_test':testing,'teacher_nll':cache['teacher_nll'],'alpha':alpha,
            'temperature':temperature,'top_k':top_k,'weights_sha256':sha256(exported/'model.safetensors'),
            'persona_profiles':source_metadata.get('characters',{}),
            'persona_rules':'You are taking part in a podcast conversation. Reply with one concrete point in at most two sentences.',
            'runtime_profiles_are_prompts_not_separate_models':True,
            'hf_file_bytes':(exported/'model.safetensors').stat().st_size,'verified_hf_transformers':False,
            'selection':'Minimum held-out assistant NLL, including update zero; test never selects weights',
            'environment':environment(),'identity':identity,
            'limitations':['Coarse top-k-plus-tail forward KL, not full-vocabulary KL or MiniLLM reproduction',
                           'Compression is actual layer removal; benchmark answer quality and native response time separately',
                           'Pretrained base knowledge is inherited, not newly trained from scratch',
                           'No natural speech, response pool, VR or general knowledge retention guarantee']}
    write_json(exported/'export_report.json',report);write_json(output/'report.json',report)
    (exported/'README.md').write_text('# '+report['name']+'\n\nExperimental compact model; no conversation-quality acceptance yet.\n\nUpstream base weights retain their Apache-2.0 license. Application tooling is AGPL-3.0.\n',encoding='utf-8')
    return report
