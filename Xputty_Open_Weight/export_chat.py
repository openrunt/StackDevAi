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
        from transformers import AutoTokenizer, LlamaForCausalLM

        torch.set_num_threads(2)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.folder, local_files_only=True
        )
        self.model = LlamaForCausalLM.from_pretrained(
            self.folder, local_files_only=True
        ).eval()
        self.context = int(self.model.config.max_position_embeddings)
        self.weights_sha256 = hashlib.sha256(
            (self.folder / "model.safetensors").read_bytes()
        ).hexdigest()
        report = self.folder / "export_report.json"
        self.representation = (
            json.loads(report.read_text()).get("representation")
            if report.exists()
            else None
        )
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|eod|>")

    def respond(self, body):
        import time

        prompt = format_prompt(body.get("messages"), body.get("mode", "chat"))
        count = int(body.get("max_new_tokens", 96))
        temperature = float(body.get("temperature", 0.8))
        if not 8 <= count <= 128 or not 0.1 <= temperature <= 1.5:
            raise ValueError("Use 8–128 response tokens and temperature 0.1–1.5")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        budget = self.context - count
        kept = ids[-budget:]
        if not kept:
            raise ValueError("Prompt is empty after tokenization")
        eos = list(set([self.model.config.eos_token_id, self.end_id]))
        eos = [n for n in eos if isinstance(n, int) and n > 0]
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
        return dict(
            text=text.strip(),
            raw=raw,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            input_tokens=len(kept),
            truncated_tokens=max(0, len(ids) - len(kept)),
            output_tokens=len(output[0]) - len(kept),
            context_tokens=self.context,
            weights_sha256=self.weights_sha256,
            backend="exported-safetensors-local-cpu",
            instruction_tuned=False,
            representation=self.representation,
        )


PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Xputty · Historical chat</title><style>
:root{color-scheme:dark}body{background:#090d16;color:#eff4ff;font:16px system-ui;margin:0}main{max-width:820px;margin:auto;padding:32px 20px}h1{font-size:36px;margin:10px 0}p,small{color:#a8b4c8;line-height:1.6}.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}button,select,input,textarea{background:#141e2e;color:inherit;font:inherit;padding:12px;border:1px solid #ffffff26;border-radius:12px}button{cursor:pointer}button[type=submit]{background:#a7ffda;color:#08251d;font-weight:700}button:disabled{opacity:.5}#messages{min-height:34vh;max-height:52vh;overflow:auto;padding:14px 0}.message{white-space:pre-wrap;padding:16px;background:#141e2e;border-radius:16px;margin:12px 0}.user{margin-left:15%;background:#15332e}.assistant{margin-right:12%}.label{font-size:11px;color:#a7ffda;margin-bottom:8px}textarea{width:100%;box-sizing:border-box;resize:vertical}#status{min-height:24px;font-size:12px}a{color:#a7ffda}</style>
<main><small>XPUTTY / EXPORTED WEIGHTS</small><h1>A conversation across time.</h1>
<p>Experimental chat framing for a prose-trained model. It may continue your words or produce fragments. Responses come only from this export.</p>
<div class="bar"><select id="mode" aria-label="Mode"><option value="chat">Chat · experimental</option><option value="continue">Continue a passage</option></select><label>Creativity <input id="temp" type="range" min="0.1" max="1.5" step="0.1" value="0.8"></label><button id="clear">New conversation</button></div>
<div id="messages" role="log" aria-live="polite"></div><form id="form"><textarea id="prompt" rows="3" maxlength="4000" placeholder="What occupies your thoughts this evening?" required aria-label="Your message"></textarea><div class="bar"><button id="send" type="submit">Send ↗</button><small>Rolling context · oldest tokens fall out when the model window fills.</small></div></form><p id="status" role="status"></p></main>
<script>const $=id=>document.getElementById(id);let key=new URLSearchParams(location.hash.slice(1)).get('key')||sessionStorage.getItem('history-chat-key')||'';if(key)sessionStorage.setItem('history-chat-key',key);history.replaceState(null,'',location.pathname);let messages=[];
function add(role,text){const d=document.createElement('div');d.className='message '+role;const label=document.createElement('div');label.className='label';label.textContent=role==='user'?'YOU':'MODEL';const content=document.createElement('div');content.textContent=text;d.append(label,content);$('messages').append(d);$('messages').scrollTop=$('messages').scrollHeight}
$('clear').onclick=()=>{messages=[];$('messages').replaceChildren();$('status').textContent='Conversation cleared.'};$('mode').onchange=()=>$('clear').click();
$('form').onsubmit=async e=>{e.preventDefault();const text=$('prompt').value.trim();if(!text)return;$('send').disabled=true;$('clear').disabled=true;$('mode').disabled=true;const pending=[...messages.slice(-8),{role:'user',content:text}];$('status').textContent='Generating from your exported weights…';try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+key},body:JSON.stringify({messages:pending,mode:$('mode').value,temperature:Number($('temp').value)})});const result=await r.json();if(!r.ok)throw Error(result.error||'Chat failed');add('user',text);add('assistant',result.text||'[No readable text generated. Try continuation mode.]');messages=result.text?[...pending,{role:'assistant',content:result.text}]:pending;$('prompt').value='';$('status').textContent=`${result.elapsed_seconds}s · ${result.input_tokens}/${result.context_tokens} context tokens · ${result.truncated_tokens} older tokens omitted · weights ${result.weights_sha256.slice(0,12)}`}catch(error){$('status').textContent=error.message}finally{$('send').disabled=false;$('clear').disabled=false;$('mode').disabled=false}};</script></html>"""


def serve(folder, port=8790):
    session = ExportSession(folder)
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
                self.reply(PAGE, mime="text/html; charset=utf-8")
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
                if urlsplit(self.path).path != "/api/chat":
                    return self.reply({"error": "Not found"}, 404)
                n = int(self.headers.get("Content-Length", 0))
                if not 0 < n <= 65536:
                    raise ValueError("Request too large")
                self.reply(session.respond(json.loads(self.rfile.read(n))))
            except (ValueError, TypeError, KeyError) as error:
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
    args = parser.parse_args()
    serve(args.model, args.port)
