#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [ "$(uname -s)" != "Darwin" ]; then
  .venv/bin/python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -r requirements.txt
printf '\nXputty Open Weight is ready. Run ./start.sh\n'
