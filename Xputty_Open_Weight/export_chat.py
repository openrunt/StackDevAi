"""Local chat/continuation using ONLY the exported Safetensors and tokenizer.

This file is copied into each model ZIP as chat.py. No training checkpoint or
network model download is required. UI framing is not instruction training.
"""

import argparse
import hashlib
import hmac
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def weights_identity(folder):
    """Hash actual local weights, including every indexed Safetensors shard."""
    folder = Path(folder).resolve()
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
        return h.hexdigest()
    single = folder / 'model.safetensors'
    if single.exists(): return digest(single)
    index = folder / 'model.safetensors.index.json'
    data = json.loads(index.read_text(encoding='utf-8'))
    files = []
    for name in sorted(set(data['weight_map'].values())):
        path = (folder / name).resolve()
        if path.parent != folder or path.suffix != '.safetensors':
            raise ValueError('Safetensors shards must stay inside the model folder')
        files.append((name, digest(path)))
    return hashlib.sha256(json.dumps({'index_sha256': digest(index), 'files': files}, sort_keys=True).encode()).hexdigest()


def generation_end_ids(configured, extra=None):
    values = list(configured) if isinstance(configured, (list, tuple)) else [configured]
    values.append(extra)
    return list(dict.fromkeys(value for value in values if type(value) is int and value >= 0))


def format_prompt(messages, mode="chat"):
    if mode not in ("chat", "continue"):
        raise ValueError("Choose chat or continue")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 40:
        raise ValueError("Send 1–40 messages")
    clean = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in (
            "user",
            "assistant",
        ):
            raise ValueError("Invalid message role")
        text = message.get("content")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Message must contain 1–4000 characters")
        clean.append((message["role"], text.strip()))
    if clean[-1][0] != "user":
        raise ValueError("The last message must be yours")
    if mode == "continue":
        return clean[-1][1]
    return (
        "\n".join(
            ("Visitor: " if role == "user" else "Companion: ") + text
            for role, text in clean
        )
        + "\nCompanion:"
    )


class ExportSession:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.lock = threading.Lock()
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        torch.set_num_threads(2)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.folder, local_files_only=True
        )
        self.weights_sha256 = weights_identity(self.folder)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.folder, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype='auto'
        ).eval()
        self.context = int(self.model.config.max_position_embeddings)
        report = self.folder / "export_report.json"
        metadata = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
        self.representation = metadata.get("representation")
        self.instruction_tuned = metadata.get("instruction_tuned") is True
        self.model_name = metadata.get("name", "Xputty scratch research model")
        self.initialization = metadata.get("initialization", "scratch-trained")
        self.persona_profiles = metadata.get("persona_profiles", {})
        self.persona_rules = metadata.get("persona_rules", "")
        self.default_temperature = float(metadata.get('default_temperature', 0.8))
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|eod|>") if "<|eod|>" in self.tokenizer.get_vocab() else None

    def respond(self, body):
        import time

        mode = body.get("mode", "chat")
        prompt = format_prompt(body.get("messages"), mode)
        if self.instruction_tuned and mode == "chat":
            messages, persona = framed_messages(body, self.persona_profiles, self.persona_rules)
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            persona = None
        count = int(body.get("max_new_tokens", 96))
        temperature = float(body.get("temperature", 0.8))
        if not 8 <= count <= 256 or not 0.1 <= temperature <= 1.5:
            raise ValueError("Use 8–256 response tokens and temperature 0.1–1.5")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        window = int(body.get("context_tokens", self.context))
        if (
            window not in (128, 256, 512, 1024, 2048)
            or window > self.context
            or window <= count
        ):
            raise ValueError(
                "Context must fit this model and exceed the response token budget"
            )
        budget = window - count
        original_tokens = len(ids)
        if self.instruction_tuned and mode == "chat":
            messages, kept, original_tokens = budget_messages(self.tokenizer, messages, budget)
        else:
            kept = ids[-budget:]
        if not kept:
            raise ValueError("Prompt is empty after tokenization")
        eos = generation_end_ids(self.model.config.eos_token_id, self.end_id)
        started = time.perf_counter()
        with self.lock, self.torch.inference_mode():
            inputs = self.torch.tensor([kept])
            output = self.model.generate(
                inputs,
                attention_mask=self.torch.ones_like(inputs),
                max_new_tokens=count,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=eos,
            )
        raw = self.tokenizer.decode(output[0, len(kept) :], skip_special_tokens=False)
        text = self.tokenizer.decode(output[0, len(kept) :], skip_special_tokens=True)
        for marker in ("<|eod|>", "\nVisitor:", "\nCompanion:"):
            text = text.split(marker)[0]
        complete = len(output[0]) - len(kept) < count or int(output[0, -1]) in eos
        return dict(
            text=text.strip(),
            raw=raw,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            input_tokens=len(kept),
            truncated_tokens=max(0, original_tokens - len(kept)),
            output_tokens=len(output[0]) - len(kept),
            response_complete=complete,
            done_reason="stop" if complete else "length",
            context_tokens=window,
            model_max_context=self.context,
            weights_sha256=self.weights_sha256,
            backend="exported-safetensors-local-cpu",
            instruction_tuned=self.instruction_tuned,
            model_name=self.model_name,
            initialization=self.initialization,
            character=persona.get("name") if persona else None,
            representation=self.representation,
        )


PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Xputty · Model chat</title><style>
:root{color-scheme:dark}body{background:#090d16;color:#eff4ff;font:16px system-ui;margin:0}main{max-width:820px;margin:auto;padding:32px 20px}h1{font-size:36px;margin:10px 0}p,small{color:#a8b4c8;line-height:1.6}.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}button,select,input,textarea{background:#141e2e;color:inherit;font:inherit;padding:12px;border:1px solid #ffffff26;border-radius:12px}button{cursor:pointer}button[type=submit]{background:#a7ffda;color:#08251d;font-weight:700}button:disabled{opacity:.5}#messages{min-height:34vh;max-height:52vh;overflow:auto;padding:14px 0}.message{white-space:pre-wrap;padding:16px;background:#141e2e;border-radius:16px;margin:12px 0}.user{margin-left:15%;background:#15332e}.assistant{margin-right:12%}.label{font-size:11px;color:#a7ffda;margin-bottom:8px}textarea{width:100%;box-sizing:border-box;resize:vertical}#status{min-height:24px;font-size:12px}a{color:#a7ffda}</style>
<main data-max-context="__MAX_CONTEXT__" data-stream="__STREAM__" data-endpoint="__CHAT_ENDPOINT__"><small>XPUTTY / EXPORTED WEIGHTS</small><h1>Your model. Your conversation.</h1>
<p>__MODEL_DESCRIPTION__</p>
<div class="bar">__PERSONA_CONTROL__<select id="mode" aria-label="Mode"><option value="chat">Chat · experimental</option><option value="continue">Continue a passage</option></select><label>Context <select id="context"><option value="256">256</option><option value="512">512</option><option value="1024">1,024</option><option value="2048">2,048</option></select></label><label>Reply <select id="reply"><option value="32">32 tokens</option><option value="64">64 tokens</option><option value="96" selected>96 tokens</option><option value="128">128 tokens</option><option value="192">192 tokens</option><option value="256">256 tokens</option></select></label><label>Creativity <input id="temp" type="range" min="0.1" max="1.5" step="0.1" value="0.8"></label><button id="clear">New conversation</button></div>
<div id="messages" role="log" aria-live="polite"></div><form id="form"><textarea id="prompt" rows="3" maxlength="4000" placeholder="What occupies your thoughts this evening?" required aria-label="Your message"></textarea><div class="bar"><button id="send" type="submit">Send ↗</button><small>Rolling context · oldest tokens fall out when the model window fills.</small></div></form><p id="status" role="status"></p></main>
<script>const $=id=>document.getElementById(id);let key=new URLSearchParams(location.hash.slice(1)).get('key')||sessionStorage.getItem('history-chat-key')||'';if(key)sessionStorage.setItem('history-chat-key',key);history.replaceState(null,'',location.pathname);let messages=[];const maxContext=Number(document.querySelector('main').dataset.maxContext)||512;for(const option of [...$('context').options])if(Number(option.value)>maxContext)option.remove();$('context').value=$('context').options[$('context').options.length-1].value;
function add(role,text){const d=document.createElement('div');d.className='message '+role;const label=document.createElement('div');label.className='label';label.textContent=role==='user'?'YOU':'MODEL';const content=document.createElement('div');content.textContent=text;d.append(label,content);$('messages').append(d);$('messages').scrollTop=$('messages').scrollHeight;return content}
const streaming=document.querySelector('main').dataset.stream==='1';const endpoint=document.querySelector('main').dataset.endpoint||'/api/chat';
async function readReply(response){if(!response.ok){const error=await response.json();throw Error(error.error||'Chat failed')}if(!streaming)return response.json();const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='',parts='',live=null,actor='',result=null;function consume(line){if(!line.trim())return;const event=JSON.parse(line);if(event.event==='error')throw Error(event.error);if(event.event==='start')actor=event.character||'';if(event.event==='delta'){parts+=event.text;if(!live)live=add('assistant','');live.textContent=(actor?actor+': ':'')+parts;$('messages').scrollTop=$('messages').scrollHeight}if(event.event==='complete')result=event.result}try{while(true){const chunk=await reader.read();if(chunk.done)break;buffer+=decoder.decode(chunk.value,{stream:true});const lines=buffer.split('\n');buffer=lines.pop();for(const line of lines)consume(line)}buffer+=decoder.decode();if(buffer.trim())consume(buffer);if(!result)throw Error('The response stream ended before completion.');return {...result,_rendered:!!live}}finally{reader.releaseLock()}}
$('clear').onclick=()=>{messages=[];$('messages').replaceChildren();$('status').textContent='Conversation cleared.'};$('mode').onchange=()=>$('clear').click();
$('form').onsubmit=async e=>{e.preventDefault();const text=$('prompt').value.trim();if(!text)return;$('send').disabled=true;$('clear').disabled=true;$('mode').disabled=true;if($('persona'))$('persona').disabled=true;const chosen=$('persona')?.value||'';const pending=[...messages.slice(-8),{role:'user',content:text}];const wire=pending.map(m=>({role:m.role==='assistant'&&m.persona_id!==chosen?'user':m.role,content:m.role==='assistant'&&m.persona_id!==chosen?(m.speaker||'Guest')+': '+m.content:m.content}));add('user',text);$('prompt').value='';const began=Date.now();const waiting=setInterval(()=>{$('status').textContent='Generating from your exported weights… '+Math.floor((Date.now()-began)/1000)+'s'},1000);$('status').textContent='Generating from your exported weights…';try{const r=await fetch(streaming?endpoint+'/stream':endpoint,{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+key},body:JSON.stringify({messages:wire,persona_id:chosen,mode:$('mode').value,temperature:Number($('temp').value),context_tokens:Number($('context').value),max_new_tokens:Number($('reply').value)})});const result=await readReply(r);if(!result._rendered)add('assistant',(result.character?result.character+': ':'')+(result.text||'[No reply text generated. Try a shorter prompt or a larger reply budget.]'));messages=result.text?[...pending,{role:'assistant',content:result.text,persona_id:chosen,speaker:result.character}]:pending;$('status').textContent=`${result.elapsed_seconds}s · ${result.input_tokens}/${result.context_tokens} context tokens · ${result.truncated_tokens} older tokens omitted · weights ${result.weights_sha256.slice(0,12)}${result.response_complete===false?' · Reply reached its token limit; increase Reply.':''}`}catch(error){$('status').textContent=error.message;if(!$('prompt').value)$('prompt').value=text}finally{clearInterval(waiting);if($('persona'))$('persona').disabled=false;$('send').disabled=false;$('clear').disabled=false;$('mode').disabled=false}};</script></html>"""


def render_page(session, endpoint='/api/chat'):
    import html

    description = (
        session.model_name + ". " + getattr(session, 'initialization', 'Pretrained comparison model') + ". This is separate from the 48M scratch experiment. Responses use only these exported weights."
        if session.instruction_tuned else
        "Experimental chat framing for a scratch-trained model. It may continue your words or produce fragments. Responses use only these exported weights."
    )
    profiles = getattr(session, "persona_profiles", {})
    choices = "" if not profiles else '<label>Character <select id="persona">' + ''.join(
        '<option value="' + html.escape(key, quote=True) + '">' + html.escape(value['name']) + '</option>'
        for key, value in profiles.items()
    ) + '</select></label>'
    page = PAGE.replace("__MAX_CONTEXT__", str(session.context)).replace("__MODEL_DESCRIPTION__", html.escape(description)).replace("__PERSONA_CONTROL__", choices).replace('__STREAM__', '1' if hasattr(session, 'native_name') else '0')
    page = page.replace('__CHAT_ENDPOINT__', html.escape(endpoint, quote=True))
    if profiles:
        page = page.replace('Chat · experimental', 'Character chat')
    if hasattr(session, "native_name"):
        page = page.replace('<option value="continue">Continue a passage</option>', '')
    temperature = max(0.1, min(1.5, float(getattr(session, 'default_temperature', 0.8))))
    page = page.replace('step="0.1" value="0.8"', 'step="0.1" value="' + str(temperature) + '"')
    page = page.replace('What occupies your thoughts this evening?', 'Give the cast a topic, or ask a question…')
    return page


def framed_messages(body, profiles, rules):
    format_prompt(body.get("messages"), body.get("mode", "chat"))
    persona_id = body.get("persona_id") or (next(iter(profiles)) if profiles else "")
    if persona_id and persona_id not in profiles:
        raise ValueError("Choose a character supplied with this model")
    persona = profiles.get(persona_id)
    system = rules + (f" Your name is {persona['name']}; you are {persona['traits']}." if persona else "")
    return ([{"role": "system", "content": system}] if system else []) + body["messages"], persona


def budget_messages(tokenizer, messages, budget, template_kwargs=None):
    messages = [dict(message) for message in messages]
    def encode():
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **(template_kwargs or {}))
    original = len(encode())
    offset = 1 if messages[0]["role"] == "system" else 0
    while len(encode()) > budget and len(messages) - offset > 1:
        removable = len(messages) - offset - 1
        del messages[offset:offset + min(2, removable)]
    ids = encode()
    while len(ids) > budget:
        last = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
        drop = len(ids) - budget + 1
        if drop >= len(last):
            raise ValueError("Character instructions exceed this context budget; select a larger window")
        messages[-1] = {"role": "user", "content": tokenizer.decode(last[drop:])}
        ids = encode()
    return messages, ids, original


class OllamaSession:
    """Local native inference, pinned to this export's exact GGUF content hash."""
    def __init__(self, folder, model_name):
        import urllib.request
        from transformers import AutoTokenizer
        self.folder = Path(folder).resolve()
        self.lock = threading.Lock()
        self.native_name = model_name
        metadata = json.loads((self.folder / "export_report.json").read_text(encoding="utf-8"))
        native = metadata.get("native_weights")
        if native:
            file = (self.folder / native["file"]).resolve()
            if file.parent != self.folder or file.suffix != '.gguf':
                raise ValueError("Native weights must be a GGUF inside this export")
            expected = native['sha256']
        else:
            conversion = json.loads((self.folder / "conversion-q8_0.json").read_text(encoding="utf-8"))
            file, expected = self.folder / "model-q8_0.gguf", conversion['output_sha256']
        checksum = hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''): checksum.update(block)
        self.weights_sha256 = checksum.hexdigest()
        if expected != self.weights_sha256:
            raise ValueError("GGUF checksum differs from its conversion record")
        request = urllib.request.Request("http://127.0.0.1:11434/api/show", data=json.dumps({"model": model_name}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=15) as response:
            shown = json.load(response)
        if self.weights_sha256 not in shown.get("modelfile", ""):
            raise ValueError("The named Ollama model does not use this exported GGUF")
        config = json.loads((self.folder / "config.json").read_text())
        self.context = int(config.get('max_position_embeddings') or config.get('text_config', {}).get('max_position_embeddings', 2048))
        self.model_name = metadata.get("name", model_name)
        self.initialization = metadata.get("initialization", "pretrained")
        self.instruction_tuned = metadata.get("instruction_tuned") is True
        self.persona_profiles = metadata.get("persona_profiles", {})
        self.persona_rules = metadata.get("persona_rules", "")
        self.default_temperature = float(metadata.get('default_temperature', 0.6))
        self.disable_thinking = metadata.get('disable_thinking') is True
        self.generation_options = {k: v for k, v in metadata.get('generation_options', {}).items()
                                   if k in ('top_p', 'top_k', 'min_p', 'presence_penalty', 'repeat_penalty')}
        self.tokenizer = AutoTokenizer.from_pretrained(self.folder, local_files_only=True, trust_remote_code=False)

    def _prepare_request(self, body):
        if body.get("mode", "chat") != "chat":
            raise ValueError("Native character chat uses chat mode")
        messages, persona = framed_messages(body, self.persona_profiles, self.persona_rules)
        count, window = int(body.get("max_new_tokens", 64)), int(body.get("context_tokens", 2048))
        temperature = float(body.get("temperature", self.default_temperature))
        if not 8 <= count <= 256 or not 0.1 <= temperature <= 1.5 or window not in (256, 512, 1024, 2048) or window > self.context or window <= count:
            raise ValueError("Invalid native response/context budget")
        messages, ids, original = budget_messages(self.tokenizer, messages, window - count,
            {'enable_thinking': False} if self.disable_thinking else None)
        payload = {"model": self.native_name, "messages": messages, "stream": False, "keep_alive": "10m",
                   "options": {"num_ctx": window, "num_predict": count, "temperature": temperature, "num_thread": 2, "num_gpu": 0}}
        payload['options'].update(self.generation_options)
        if self.disable_thinking: payload['think'] = False
        return payload, persona, ids, original, window

    def _check_identity(self):
        import urllib.request
        check = urllib.request.Request("http://127.0.0.1:11434/api/show", data=json.dumps({"model": self.native_name}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(check, timeout=15) as response:
            shown = json.load(response)
        if self.weights_sha256 not in shown.get("modelfile", ""):
            raise ValueError("Ollama model changed since this chat opened; re-import this exact GGUF")

    def _result(self, result, text, elapsed, persona, ids, original, window):
        return {"text": text.strip(), "elapsed_seconds": round(elapsed, 3),
                "input_tokens": len(ids), "context_tokens": window, "model_max_context": self.context,
                "truncated_tokens": max(0, original - len(ids)), "output_tokens": result.get("eval_count"),
                "response_complete": result.get("done_reason") != "length", "done_reason": result.get("done_reason"),
                "weights_sha256": self.weights_sha256, "backend": "exported-GGUF-local-Ollama",
                "instruction_tuned": self.instruction_tuned, "character": persona.get("name") if persona else None,
                "model_name": self.model_name, "initialization": self.initialization}

    def respond(self, body):
        import time
        import urllib.request
        payload, persona, ids, original, window = self._prepare_request(body)
        started = time.perf_counter()
        self._check_identity()
        request = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with self.lock, urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        return self._result(result, result['message']['content'], time.perf_counter() - started, persona, ids, original, window)

    def stream(self, body):
        import time
        import urllib.request
        payload, persona, ids, original, window = self._prepare_request(body)
        self._check_identity()
        payload['stream'] = True
        started = time.perf_counter()
        yield {'event': 'start', 'character': persona.get('name') if persona else None}
        request = urllib.request.Request('http://127.0.0.1:11434/api/chat', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        parts, first = [], None
        with self.lock, urllib.request.urlopen(request, timeout=180) as response:
            for line in response:
                item = json.loads(line)
                if item.get('error'): raise ValueError(item['error'])
                text = item.get('message', {}).get('content', '')
                if text:
                    if first is None: first = time.perf_counter() - started
                    parts.append(text)
                    yield {'event': 'delta', 'text': text}
                if item.get('done'):
                    result = self._result(item, ''.join(parts), time.perf_counter() - started, persona, ids, original, window)
                    result['first_token_seconds'] = round(first, 3) if first is not None else None
                    yield {'event': 'complete', 'result': result}
                    return
        raise ValueError('Native stream ended without a completion record')


def serve(folder, port=8790, ollama_model=None):
    session = OllamaSession(folder, ollama_model) if ollama_model else ExportSession(folder)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, code=200, mime="application/json"):
            blob = (
                value.encode() if isinstance(value, str) else json.dumps(value).encode()
            )
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(blob)

        def do_GET(self):
            if urlsplit(self.path).path == "/":
                self.reply(
                    render_page(session),
                    mime="text/html; charset=utf-8",
                )
            else:
                self.reply({"error": "Not found"}, 404)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and origin != f"http://127.0.0.1:{port}":
                return self.reply({"error": "Origin not allowed"}, 403)
            if not hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token
            ):
                return self.reply(
                    {
                        "error": "Open the complete local chat link printed in the terminal"
                    },
                    401,
                )
            try:
                route = urlsplit(self.path).path
                if route not in ("/api/chat", "/api/chat/stream"):
                    return self.reply({"error": "Not found"}, 404)
                n = int(self.headers.get("Content-Length", 0))
                if not 0 < n <= 65536:
                    raise ValueError("Request too large")
                body = json.loads(self.rfile.read(n))
                if route == '/api/chat/stream':
                    if not hasattr(session, 'stream'): raise ValueError('This backend does not provide native streaming')
                    records = session.stream(body)
                    first = next(records)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    try:
                        for record in (first,):
                            self.wfile.write((json.dumps(record, ensure_ascii=False) + '\n').encode()); self.wfile.flush()
                        for record in records:
                            self.wfile.write((json.dumps(record, ensure_ascii=False) + '\n').encode()); self.wfile.flush()
                    except (ValueError, TypeError, KeyError, OSError) as error:
                        try:
                            self.wfile.write((json.dumps({'event': 'error', 'error': str(error)}) + '\n').encode()); self.wfile.flush()
                        except OSError: pass
                    finally:
                        records.close()
                    return
                self.reply(session.respond(body))
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.reply({"error": str(error)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Chat: http://127.0.0.1:{port}/#key={token}", flush=True)
    print("Loaded exported weights " + session.weights_sha256, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--ollama-model", help="Use the exact GGUF registered under this local name; no fallback")
    args = parser.parse_args()
    serve(args.model, args.port, args.ollama_model)
