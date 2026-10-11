set -euo pipefail
cd /content/humanoid-internship-challenge
mountpoint -q /content/drive || { echo "Run the Drive mount cell first."; exit 1; }
PYTHON=/content/libero-official-env/bin/python
OUTPUT=/content/drive/MyDrive/humanoid_results/bowl_human_reference
mkdir -p "$OUTPUT"
# User-confirmed data_003 geometry: top ID6; ID0 flat/up; matching camera calibration.
cp config/smolvla/bowl_human_calibration.json "$OUTPUT/calibration.json"
MPLBACKEND=Agg "$PYTHON" scripts/bowl_human_reference.py \
  --detections results/data_003_raw/detections.csv \
  --centers results/data_003_raw/object/cube_center/trajectory.csv \
  --calibration "$OUTPUT/calibration.json" \
  --start 5.9 --end 16.7 --output "$OUTPUT/reference.csv"
printf 'Review human reference preview before simulation: %s/reference.png\n' "$OUTPUT"
