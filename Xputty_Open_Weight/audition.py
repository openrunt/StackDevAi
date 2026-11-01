import json, sys
from pathlib import Path
from vox.limits import limit_cpu
from vox.model import TinyLlama, ModelConfig, generate
from vox.dialogue_format import raw_prompt, parse_raw_generation
import sentencepiece as spm

torch = limit_cpu(2)
root = Path(sys.argv[1])
checkpoint = torch.load(
    root / "dense/checkpoint.pt", map_location="cpu", weights_only=True
)
model = TinyLlama(ModelConfig(**checkpoint["config"]))
model.load_state_dict(checkpoint["weights"])
sp = spm.SentencePieceProcessor(model_file=str(root / "data/vox.model"))
prompt = sp.encode(raw_prompt(sys.stdin.read()), out_type=int)
ids = generate(
    model, prompt, max_new_tokens=96, temperature=0.8, stop_id=sp.piece_to_id("<|eod|>")
)
text = sp.decode(ids)
print(json.dumps({"raw": text, **parse_raw_generation(text)}))
