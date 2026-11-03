"""Compare real native bytes, bounded answer checks and streamed response latency."""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vox.public_baseline import sha256,write_json

# Original finite checks, held out of this compact adaptation. They are not a
# broad knowledge benchmark; forced-choice decoding is distinct from free speech.
CHECKS = [
 ('jupiter','Which planet is the largest in our solar system?','Jupiter',['Earth','Mars','Venus'],'recall','https://science.nasa.gov/jupiter/jupiter-facts/'),
 ('moon','Where does visible moonlight primarily come from?','Reflected sunlight',['Nuclear fusion inside the Moon','Burning gas','Electric lighting'],'recall','https://science.nasa.gov/moon/facts/'),
 ('water','What is the chemical formula of water?','H2O',['CO2','O2','NaCl'],'recall','Elementary chemistry'),
 ('triangle','How many sides does a triangle have?','Three',['Two','Four','Five'],'recall','Definition'),
 ('cup','Mira placed a cup inside a green box. Leon carried the unopened box into a van. Where is the cup now?','Inside the green box in the van',['On the table','In Mira\'s pocket','Outside the van'],'state tracking','Original fictional scene'),
 ('badge','Asha gave a badge to Ben. Ben gave that same badge to Cora. Who holds the badge now?','Cora',['Asha','Ben','Nobody'],'state tracking','Original fictional scene'),
 ('lamp','Two lamps are initially off. The left one is switched on, then off. The right one is never touched. Which lamps are on?','Neither lamp',['Only the left lamp','Only the right lamp','Both lamps'],'state tracking','Original fictional scene'),
 ('message','Only Rowan read the sealed note. Tessa saw the envelope but did not read its contents. Who knows what the note says?','Rowan',['Tessa','Both Rowan and Tessa','Neither person'],'state tracking','Original fictional scene'),
 ('generator','In this fictional machine, an empty fuel tank stops the generator. Its tank is empty and all wires are intact. Which stated fault explains why it stopped?','The empty fuel tank',['A broken wire','Too much coolant','A blocked microphone'],'causality','Original fictional rule'),
 ('bridge','A fictional bridge opens only when BOTH switches are on. The north switch is on; the south switch is off. Is the bridge open?','No, because both switches must be on',['Yes, one switch is enough','Yes, because north is on','Its color determines this'],'causality','Original fictional rule'),
 ('alarm','A fictional alarm sounds whenever the red door opens. The red door opens. What follows from the stated rule?','The alarm sounds',['The alarm stays silent','All doors lock','The lights turn blue'],'causality','Original fictional rule'),
 ('negation','The note says: Do NOT open the blue box. Open the yellow box. Which box does it instruct you to open?','The yellow box',['The blue box','Both boxes','Neither box'],'negation','Original instruction'),
 ('count','There are seven lanterns. Two are removed and three are added. How many lanterns are there now?','Eight',['Five','Seven','Ten'],'arithmetic','7-2+3=8'),
 ('sleep','In this story, slow creatures never sleep, humans require sleep, and nobody can keep moving forever. Which danger follows even if humans walk faster?','Humans eventually need to rest while the creatures can continue',['The creatures must also rest','Humans can escape forever without rest','Walking speed makes sleep unnecessary'],'causality','Original fictional premises'),
 ('unavailable','The room description names two chairs and a desk. It gives no audience count. What can you conclude about the number of viewers?','The description does not establish it',['Exactly two viewers','Exactly three viewers','There are no viewers'],'grounding','Original incomplete scene description'),
 ('ownership','Inez owns a red book. Pavel borrows a different blue book from Omar. Whose book did Pavel borrow?','Omar\'s',['Inez\'s','His own','It belongs to both Inez and Omar'],'speaker/role','Original fictional scene'),
]

FREE = [
 'Fast zombies are obviously scarier; you can just walk away from slow ones. What do you think?',
 'In our fictional story, an old radio repeats a sentence nobody has said yet. Give me one interesting possibility for what happens next.',
 'Our VR studio has a blue door, two stools and a wall screen. Suggest one small interaction that would reward a curious visitor.',
 'A friend says a funny host should make jokes constantly. Offer a specific counterpoint without giving a list.',
]


def post(route,body):
    req=urllib.request.Request('http://127.0.0.1:11434/api/'+route,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as response:return json.load(response)


def choices_for(index,record):
    key,prompt,correct,wrong,category,source=record
    # Precisely balanced correct positions; the same question/order for every model.
    options=list(wrong);random.Random(739+index).shuffle(options);answer='ABCD'[index%4];options.insert(index%4,correct)
    return {'id':key,'category':category,'source':source,'prompt':prompt+'\n'+'\n'.join(letter+'. '+value for letter,value in zip('ABCD',options)),
            'expected':answer,'options':options}


def stream(tag,messages,limit=96):
    body={'model':tag,'messages':messages,'stream':True,'keep_alive':'5m',
          'options':{'num_ctx':2048,'num_predict':limit,'num_thread':2,'num_gpu':0,'temperature':0,'seed':739,'repeat_penalty':1.05}}
    req=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    began=time.perf_counter();first=None;parts=[];last={}
    with urllib.request.urlopen(req,timeout=240) as response:
        for line in response:
            row=json.loads(line)
            if row.get('error'):raise ValueError(row['error'])
            text=row.get('message',{}).get('content','')
            if text:
                if first is None:first=time.perf_counter()-began
                parts.append(text)
            if row.get('done'):last=row
    if not last:raise ValueError('Native stream ended without completion metadata')
    return {'text':''.join(parts),'first_text_seconds':first,'total_seconds':time.perf_counter()-began,
            'output_tokens':last.get('eval_count',0),'decode_seconds':last.get('eval_duration',0)/1e9,
            'load_seconds':last.get('load_duration',0)/1e9,'done_reason':last.get('done_reason')}


def run(models,output):
    if output.exists():raise ValueError('Choose a new benchmark file')
    specs=json.loads(models.read_text(encoding='utf-8'));results=[]
    for spec in specs:
        file=Path(spec['file']);digest=sha256(file)
        if digest not in post('show',{'model':spec['tag']})['modelfile']:raise ValueError('Wrong native model: '+spec['tag'])
        post('generate',{'model':spec['tag'],'keep_alive':0})
        cold=stream(spec['tag'],[{'role':'user','content':'What makes a good conversation? Answer briefly.'}],48)
        checks=[]
        for i,item in enumerate(CHECKS):
            question=choices_for(i,item)
            payload={'model':spec['tag'],'messages':[{'role':'system','content':'Choose the best answer. Return a JSON object with its letter.'},
                        {'role':'user','content':question['prompt']}],'stream':False,'keep_alive':'5m',
                     'format':{'type':'object','properties':{'answer':{'type':'string','enum':['A','B','C','D']}},'required':['answer'],'additionalProperties':False},
                     'options':{'num_ctx':2048,'num_predict':24,'temperature':0,'num_thread':2,'num_gpu':0,'seed':739}}
            reply=post('chat',payload);raw=reply['message']['content']
            try:answer=json.loads(raw).get('answer')
            except (ValueError,AttributeError):answer=None
            checks.append({**question,'answer':answer,'correct':answer==question['expected'],'raw':raw,'done_reason':reply.get('done_reason')})
        # These unconstrained replies determine latency and need complete manual review.
        free=[]
        for prompt in FREE:
            row=stream(spec['tag'],[{'role':'system','content':'Answer the speaker directly in one or two conversational sentences.'},
                                   {'role':'user','content':prompt}]);free.append({'prompt':prompt,**row})
        result={**spec,'file_bytes':file.stat().st_size,'weights_sha256':digest,'cold':cold,'checks':checks,'free_responses':free,
                'bounded_checks_correct':sum(r['correct'] for r in checks),'bounded_checks_total':len(checks),
                'invalid_choice_responses':sum(r['answer'] not in ('A','B','C','D') for r in checks),
                'warm_first_text_median_seconds':statistics.median(r['first_text_seconds'] for r in free if r['first_text_seconds'] is not None),
                'median_decode_tokens_per_second':statistics.median(r['output_tokens']/max(r['decode_seconds'],1e-9) for r in free)}
        results.append(result);report={'schema':'xow-compact-native-comparison/1','models':results,
             'conditions':'Sequential native CPU requests; two inference threads, 2048 context, greedy. Run after local training/tests stop.',
             'limitations':['Sixteen authored checks are a small development sample, not general knowledge equivalence',
                            'Forced-choice decoding constrains answer format; free responses are unconstrained and reviewed separately',
                            'Same precision is required for a parameter-reduction claim; actual file hashes are recorded',
                            'First text is not first audio; four replies do not establish podcast performance',
                            'Pretraining contamination is unknown; no unattended-host acceptance is implied']}
        output.parent.mkdir(parents=True,exist_ok=True);write_json(output,report)
        print(json.dumps({k:result[k] for k in ('label','file_bytes','bounded_checks_correct','bounded_checks_total','warm_first_text_median_seconds','median_decode_tokens_per_second')}),flush=True)
        post('generate',{'model':spec['tag'],'keep_alive':0})
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(**vars(p.parse_args()))
