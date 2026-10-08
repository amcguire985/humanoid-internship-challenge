set -euo pipefail
nvidia-smi
mountpoint -q /content/drive || { echo "Run the Drive mount cell first."; exit 1; }
apt-get update -qq
apt-get install -y -qq libegl1 libgl1 libosmesa6 ffmpeg
python -m pip install -q uv
python -m uv venv --python 3.12 --seed --allow-existing /content/libero-official-env
python -m uv pip install --python /content/libero-official-env/bin/python \
  'lerobot[smolvla,libero,evaluation] @ git+https://github.com/huggingface/lerobot.git@8c920c4270460851cedd2737657584586d3dc66f' \
  'hf-libero==0.1.4' 'mujoco==3.3.2'
export LIBERO_CONFIG_PATH=/content/libero-official-config
printf 'n\n' | /content/libero-official-env/bin/python -c 'import libero.libero'
/content/libero-official-env/bin/python -c 'import torch; from lerobot.scripts.lerobot_eval import main; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
