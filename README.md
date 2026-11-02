# Xputty Open Weight — 48M Research

A browser workspace for reviewing source material, training an original small language model, testing exported weights in chat, and downloading GGUF/Safetensors.

This development branch adds a 48,116,224-parameter model and mixed 256/512/1,024/2,048-token training windows. It is a research build; runtime acceptance does not establish coherent podcast-host behavior.

## Test in Chrome

On this branch, choose **Code → Codespaces → Create codespace**. The Python 3.11 container installs the app and starts its private worker. Open the forwarded port **8787** using the complete connection link printed in the terminal.

## Test locally

Open `Xputty_Open_Weight/` and use `START-WINDOWS.cmd`, `START-MAC.command` or `sh start.sh`. The local canvas uses port **8791**. Runtime dependencies install at first launch.

New projects default to 48M. The 1.6M quick-test preset remains available; existing projects retain their original architecture. Review **Context mix & learning schedule** before building. Small update targets test the pipeline; substantive language training requires more data and compute.

[Application guide](Xputty_Open_Weight/README.md) · [Verification record](Xputty_Open_Weight/VERIFICATION.json)

This branch includes no private transcripts, access keys, project storage or trained weights. The original `xputty-open-weight` branch remains separate.
