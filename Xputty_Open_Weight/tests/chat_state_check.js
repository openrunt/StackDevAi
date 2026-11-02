// Exercise the actual exported chat client without accessing a user's browser.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('export_chat.py', 'utf8');
const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];
function element() { return { value:'', disabled:false, children:[], textContent:'', scrollHeight:0,
  append(...items){ this.children.push(...items); }, replaceChildren(){ this.children=[]; }, remove(){} }; }
const ids={};
for(const id of ['messages','form','send','clear','mode','context','reply','temp','prompt','status','persona']) ids[id]=element();
ids.context.options=[256,512,1024,2048].map(value=>({value:String(value),remove(){}}));
ids.mode.value='chat';ids.reply.value='96';ids.temp.value='0.3';ids.persona.value='nova';
let settle, fail, payload;
const context={ document:{getElementById:id=>ids[id],querySelector:()=>({dataset:{maxContext:32768}}),createElement:element},
  location:{hash:'#key=ephemeral-test',pathname:'/'},history:{replaceState(){}},sessionStorage:{getItem(){return ''},setItem(){}},
  URLSearchParams, Date, setInterval(){return 1},clearInterval(){},
  fetch:async(url,options)=>{payload=JSON.parse(options.body);return new Promise((resolve,reject)=>{settle=resolve;fail=reject;});} };
vm.runInNewContext(script,context);
(async()=>{
  ids.prompt.value='Give us a fresh story idea.';
  const pending=ids.form.onsubmit({preventDefault(){}});
  assert.equal(ids.messages.children.length,1,'Show the user message immediately, before inference returns');
  assert.equal(ids.prompt.value,'');assert.equal(ids.persona.disabled,true);
  assert.equal(payload.persona_id,'nova');assert.equal(payload.max_new_tokens,96);
  ids.prompt.value='My next question is already drafted.';
  settle({ok:true,json:async()=>({text:'A radio whispers tomorrow’s weather.',character:'Nova',elapsed_seconds:2,input_tokens:50,
    context_tokens:2048,truncated_tokens:0,weights_sha256:'abc123abc123',response_complete:false})});
  await pending;
  assert.equal(ids.messages.children.length,2);assert.equal(ids.persona.disabled,false);
  assert.equal(ids.prompt.value,'My next question is already drafted.','A completed reply must not erase a new draft');
  assert.match(ids.status.textContent,/token limit/,'A clipped reply must be visible as clipped');
  ids.prompt.value='Another question';
  ids.persona.value='orion';
  const rejected=ids.form.onsubmit({preventDefault(){}});
  assert.equal(payload.messages[1].role,'user','A peer character’s words are not the new actor’s own assistant history');
  assert.match(payload.messages[1].content,/^Nova: /);
  fail(new Error('Worker unavailable'));
  await rejected;
  assert.equal(ids.prompt.value,'Another question','Preserve the failed input for retry');
  assert.equal(ids.send.disabled,false);assert.match(ids.status.textContent,/Worker unavailable/);
  ids.clear.onclick();assert.equal(ids.messages.children.length,0);
  let released=false;
  const streamContext={...context,TextDecoder,document:{...context.document,querySelector:()=>({dataset:{maxContext:32768,stream:'1'}})}};
  const events=[{event:'start',character:'Nova'},{event:'delta',text:'Hello 🌙 '},{event:'delta',text:'there.'},
    {event:'complete',result:{text:'Hello 🌙 there.',character:'Nova',elapsed_seconds:1,input_tokens:20,context_tokens:2048,
      truncated_tokens:0,weights_sha256:'abc123abc123',response_complete:true}}];
  const bytes=Buffer.from(events.map(v=>JSON.stringify(v)).join('\n')+'\n');
  const cut=bytes.indexOf(Buffer.from('🌙'))+2;
  const chunks=[bytes.subarray(0,cut),bytes.subarray(cut,cut+17),bytes.subarray(cut+17)];
  streamContext.fetch=async(url)=>{assert.equal(url,'/api/chat/stream');return {ok:true,body:{getReader:()=>({
    read:async()=>chunks.length?{done:false,value:chunks.shift()}:{done:true},releaseLock(){released=true}})}}};
  vm.runInNewContext(script,streamContext);
  ids.persona.value='nova';
  ids.prompt.value='An actual streamed reply';
  await ids.form.onsubmit({preventDefault(){}});
  assert.equal(ids.messages.children.length,2,'Do not append the final reply twice after streaming');
  assert.equal(ids.messages.children[1].children[1].textContent,'Nova: Hello 🌙 there.');
  assert.equal(released,true,'Release the streaming reader');
  console.log('Chat client checks passed: immediate feedback, correct persona, completion status, retry and clear.');
})().catch(error=>{console.error(error);process.exitCode=1;});
