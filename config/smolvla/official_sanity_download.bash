set -euo pipefail
export LIBERO_CONFIG_PATH=/content/libero-official-config
/content/libero-official-env/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('HuggingFaceVLA/smolvla_libero',
    revision='6721902bc4d61e50a3bfdb11dfb4cb626f05d102',
    local_dir='/content/smolvla-libero-official-checkpoint',
    allow_patterns=['*.json','*.safetensors','README.md'])
# Small metadata download only; no demonstration dataset download.
snapshot_download('HuggingFaceVLA/libero', repo_type='dataset',
    revision='86958911c0f959db2bbbdb107eb3e17c5f9c798e',
    local_dir='/content/libero-official-task-metadata',
    allow_patterns=['meta/info.json','meta/tasks.parquet'])
from libero.libero import get_libero_path, get_assets_path
print('BDDL:', get_libero_path('bddl_files'))
print('Initial states:', get_libero_path('init_states'))
from pathlib import Path
assets=Path(get_assets_path())
assert assets.is_dir(), f'LIBERO asset download failed: {assets}'
print('Assets:', assets)
PY
