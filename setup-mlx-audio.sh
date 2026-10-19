#!/usr/bin/env bash
# Separate native MLX-Audio environment. Never modifies a legacy/remote venv.
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "Local (MLX-Audio) requires Apple Silicon macOS." >&2
  exit 1
fi
macos_version="$(sw_vers -productVersion)"
macos_major="${macos_version%%.*}"
if [[ ! "$macos_major" =~ ^[0-9]+$ ]] || (( macos_major < 14 )); then
  echo "The pinned MLX/Metal wheels require macOS 14 or newer (found $macos_version)." >&2
  echo "No environment has been created or modified." >&2
  exit 1
fi
venv_dir="${MLX_AUDIO_VENV:-venv-mlx-audio}"
if [[ -e "$venv_dir" ]]; then
  echo "Refusing to modify existing environment: $venv_dir" >&2
  echo "Use MLX_AUDIO_VENV=/path/to/a/new/venv to install another isolated copy." >&2
  exit 1
fi
command -v uv >/dev/null || { echo "Install uv or enter the tts-lab Nix shell first." >&2; exit 1; }
uv venv --python 3.12 "$venv_dir"
uv pip install --python "$venv_dir/bin/python" -r requirements-mlx-audio.txt
"$venv_dir/bin/python" -c 'import importlib.metadata as m; import tts_audiobook_tool.tts; print("MLX-Audio",m.version("mlx-audio"),"MLX",m.version("mlx")); print("App imports OK; select an MLX-Audio family from Model settings.")'
printf 'Launch from this checkout: %s/bin/python -m tts_audiobook_tool\n' "$venv_dir"
