"""Isolated, resumable browser job. Never modifies the donor lab or its weights."""

import argparse, hashlib, json, os, subprocess, sys, time, zipfile
from pathlib import Path


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run(root, steps=400):
    for name in ("DEVICE", "NATIVE_PYTHON", "OLLAMA_VERIFY"):
        if "XOW_" + name in os.environ:
            os.environ["VOX_" + name] = os.environ["XOW_" + name]
    from vox.intake import ingest_file
    from vox.raw_baseline import build
    from vox.tokenizer import train as tokenize
    from vox.data import prepare
    from vox.train import train
    from vox.evaluate import evaluate
    from vox.export import export

    root = Path(root)
    state = json.loads((root / "state.json").read_text())
    ledger = root / "events.jsonl"

    def phase(label, **details):
        state.update(status="running", phase=label, **details)
        write(root / "state.json", state)
        previous = "0" * 64
        if ledger.exists():
            previous = json.loads(ledger.read_text().splitlines()[-1])["hash"]
        event = {
            "phase": label,
            "time_ns": time.time_ns(),
            "details": details,
            "previous": previous,
        }
        event["hash"] = hashlib.sha256(
            json.dumps(event, sort_keys=True).encode()
        ).hexdigest()
        with ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        print(label, flush=True)

    try:
        sources = []
        warnings = []
        from materials import source_path

        for i, item in enumerate(state["files"]):
            source = source_path(root, item)
            phase("Reading material", file=source.name)
            out = root / "intake" / str(i)
            out.parent.mkdir(exist_ok=True)
            if not (out / "candidate_turns.jsonl").exists():
                ingest_file(source, out)
            rows = [
                json.loads(line)
                for line in (out / "candidate_turns.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            # Captured dialogue has unknown speakers; retain source timing without inventing turns.
            if len(rows) < 2:
                warnings.append(
                    source.name
                    + ": fewer than two extracted segments; excluded from training"
                )
                continue
            folder = root / "normalized"
            folder.mkdir(exist_ok=True)
            target = folder / (source.stem + ".txt")
            lines = []
            for row in rows:
                if row.get("start_ms") is not None:
                    ms = int(row["start_ms"])
                    s, ms = divmod(ms, 1000)
                    m, s = divmod(s, 60)
                    h, m = divmod(m, 60)
                    lines.append(f"{h:02}:{m:02}:{s:02}.{ms:03}")
                lines.append(row["text"])
            target.write_text("\n".join(lines), encoding="utf-8")
            sources.append(target)
        phase("Preparing episodes", warnings=warnings)
        if not (root / "data/prepared_report.json").exists():
            if (root / "data").exists():
                import shutil

                shutil.rmtree(root / "data")  # owned derived directory only
            report = build(sources, root / "data")
            tokenize(root / "data", 4096)
            prepare(root / "data")
        report = json.loads((root / "data/corpus_report.json").read_text())
        phase("Training", episodes=report["episodes"], steps=steps)
        options = dict(
            seq_len=512, dim=128, layers=3, heads=4, ffn_dim=320, batch=2, cores=2
        )
        checkpoint = root / "dense/checkpoint.pt"
        if not checkpoint.exists():
            train(root / "data", root / "dense", steps=0, **options)
            write(
                root / "before.json",
                evaluate(root / "data", checkpoint, split="val", samples=32, cores=2),
            )
        train(
            root / "data",
            root / "dense",
            steps=steps,
            resume=True,
            save_every=25,
            log_interval=25,
            **options,
        )
        phase("Evaluating")
        metrics = {
            s: evaluate(root / "data", checkpoint, split=s, samples=32, cores=2)
            for s in ["val", "test"]
        }
        write(root / "metrics.json", metrics)
        phase("Exporting")
        export(root / "data", checkpoint, root / "export", cores=2)
        phase("Checking Transformers")
        subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                "tools/verify_hf_export.py",
                "--run",
                str(root),
            ],
            check=True,
        )
        hf = json.loads((root / "hf-verification.json").read_text())
        if not hf["logits_match"] or not hf["all_tokenizer_probes_match"]:
            raise ValueError("Export parity check failed")
        checks = {
            "transformers": "passed",
            "llama_cpp": "not installed",
            "ollama": "not checked",
        }
        import importlib.util

        native_python = os.environ.get("VOX_NATIVE_PYTHON")
        if native_python or importlib.util.find_spec("llama_cpp"):
            subprocess.run(
                [
                    native_python or sys.executable,
                    "-X",
                    "utf8",
                    "tools/verify_native_export.py",
                    "--run",
                    str(root),
                ],
                check=True,
            )
            native = json.loads((root / "native-verification.json").read_text())
            if (
                not native["all_tokenizer_probes_match"]
                or not native["logits_match_f16"]
            ):
                raise ValueError("Native export parity check failed")
            checks["llama_cpp"] = "passed"
        if os.environ.get("VOX_OLLAMA_VERIFY", "auto") != "0":
            import urllib.request

            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:11434/api/version", timeout=2
                ):
                    pass
            except OSError:
                checks["ollama"] = "service unavailable"
            else:
                subprocess.run(
                    [
                        sys.executable,
                        "-X",
                        "utf8",
                        "tools/register_ollama_baseline.py",
                        "--run",
                        str(root),
                        "--name",
                        "vox-browser:" + root.name,
                    ],
                    check=True,
                )
                ollama = json.loads((root / "ollama-verification.json").read_text())
                if not ollama.get("done"):
                    raise ValueError("Ollama generation did not complete")
                checks["ollama"] = "passed"
        phase("Packaging", checks=checks)
        export_report = json.loads((root / "export/export_report.json").read_text())
        export_report.update(
            verified_hf_transformers=True,
            verified_llama_cpp=checks["llama_cpp"] == "passed",
            verified_ollama=checks["ollama"] == "passed",
        )
        write(root / "export/export_report.json", export_report)
        card = root / "export/README.md"
        text = card.read_text(encoding="utf-8")
        text += (
            "\nBrowser build verification: Transformers tokenizer/logit parity passed. Native llama.cpp: "
            + checks["llama_cpp"]
            + ". Ollama: "
            + checks["ollama"]
            + ". See export_report.json.\n"
        )
        card.write_text(text, encoding="utf-8")
        with zipfile.ZipFile(root / "model.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for file in (root / "export").iterdir():
                if file.is_file():
                    z.write(file, "model/" + file.name)
            for name in [
                "metrics.json",
                "before.json",
                "hf-verification.json",
                "events.jsonl",
            ]:
                z.write(root / name, "reports/" + name)
            for name in ["native-verification.json", "ollama-verification.json"]:
                if (root / name).exists():
                    z.write(root / name, "reports/" + name)
        phase("Ready", metrics=metrics, checks=checks)
        state.update(status="complete", file=None)
        write(root / "state.json", state)
    except Exception as error:
        state.update(status="failed", error=str(error))
        write(root / "state.json", state)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--steps", type=int, default=400)
    a = p.parse_args()
    run(a.root.resolve(), a.steps)
