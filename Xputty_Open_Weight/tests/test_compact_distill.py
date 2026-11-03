import json
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from vox.compact_distill import layer_indices,make_student,coarse_teacher,teacher_kl,train


def test_coarse_kl_matches_full_distribution_and_has_valid_gradients():
    torch.manual_seed(7)
    teacher=torch.randn(4,16);student=torch.randn(4,16,requires_grad=True);temperature=2.
    full=coarse_teacher(teacher,16,temperature)
    actual=teacher_kl(student,full,temperature)
    expected=torch.nn.functional.kl_div(torch.log_softmax(student/temperature,-1),torch.softmax(teacher/temperature,-1),reduction='batchmean')*temperature**2
    assert torch.allclose(actual,expected,atol=1e-6)
    actual.backward();assert torch.isfinite(student.grad).all()
    with torch.inference_mode(): sparse=coarse_teacher(teacher,4,temperature)
    student.grad=None;value=teacher_kl(student,sparse,temperature);value.backward()
    assert value >= 0 and torch.isfinite(student.grad).all()


def test_layer_pruning_copies_exact_weights_and_full_depth_matches():
    teacher=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
                    num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,tie_word_embeddings=True)).eval()
    before={n:p.clone() for n,p in teacher.state_dict().items()}
    student,indices=make_student(teacher,2)
    assert indices==[0,3] and sum(p.numel() for p in student.parameters())<sum(p.numel() for p in teacher.parameters())
    assert torch.equal(student.model.layers[1].mlp.up_proj.weight,teacher.model.layers[3].mlp.up_proj.weight)
    same,_=make_student(teacher,4);same.eval();ids=torch.tensor([[1,4,8]])
    with torch.no_grad():assert torch.equal(same(ids).logits,teacher(ids).logits)
    assert all(torch.equal(before[n],p) for n,p in teacher.state_dict().items())
    assert layer_indices(30,5)==[0,7,14,22,29]
    with pytest.raises(ValueError):layer_indices(30,31)


def test_actual_distillation_resume_is_exact_and_guards_identity(tmp_path):
    base=tmp_path/'base';base.mkdir();torch.manual_seed(41)
    words=['[UNK]','[PAD]','[EOS]','user','assistant']+[f'q{i}' for i in range(8)]+[f'a{i}' for i in range(8)]
    tk=Tokenizer(WordLevel({w:i for i,w in enumerate(words)},unk_token='[UNK]'));tk.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tk,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
    tokenizer.chat_template="{% for m in messages %}{{m['role']+' '+m['content']+' [EOS] '}}{% endfor %}{% if add_generation_prompt %}{{'assistant '}}{% endif %}"
    tokenizer.save_pretrained(base)
    teacher=LlamaForCausalLM(LlamaConfig(vocab_size=len(words),hidden_size=16,intermediate_size=32,num_hidden_layers=3,
             num_attention_heads=2,num_key_value_heads=1,tie_word_embeddings=True,eos_token_id=2,pad_token_id=1))
    teacher.save_pretrained(base,safe_serialization=True)
    (base/'LICENSE').write_text('Fixture only')
    dataset=tmp_path/'data';(dataset/'data').mkdir(parents=True)
    all_rows=[]
    for split,start in [('train',0),('val',2),('test',4)]:
        rows=[{'id':str(i),'messages':[{'role':'user','content':f'q{i}'},{'role':'assistant','content':f'a{i}'}]} for i in range(start,start+2)]
        all_rows+=rows;(dataset/'data'/(split+'.jsonl')).write_text(''.join(json.dumps({'episode_id':r['id']})+'\n' for r in rows))
    (dataset/'conversations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows))
    kwargs=dict(base=base,dataset=dataset,cache_file=tmp_path/'teacher.pt',keep=2,alpha=.5,context=128,
                train_samples=2,eval_samples=2,eval_every=1,top_k=8)
    one=train(output=tmp_path/'resumed',steps=1,**kwargs)
    resumed=train(output=tmp_path/'resumed',steps=2,resume=True,**kwargs)
    full=train(output=tmp_path/'full',steps=2,**kwargs)
    a=torch.load(tmp_path/'resumed/checkpoint.pt',weights_only=True);b=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True)
    assert a['step']==b['step']==2 and all(torch.equal(a['weights'][n],v) for n,v in b['weights'].items())
    assert resumed['selected_update']==full['selected_update']
    assert resumed['parameter_count']<resumed['teacher_parameter_count']
    with pytest.raises(ValueError,match='configuration changed'):
        train(output=tmp_path/'resumed',steps=3,resume=True,**{**kwargs,'alpha':.25})
