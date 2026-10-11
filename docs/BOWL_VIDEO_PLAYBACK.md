# Bowl rollout video playback fix

Inspection of the supplied bowl_synthetic_results.zip found the custom
agentview.mp4 and wrist.mp4 contain mp4v (MPEG-4 Part 2), with their moov index at
the end. The official eval_episode_0.mp4 contains avc1 (H.264). All three archived
files have their indexes, so this sample does not establish unfinished/corrupt
recordings. Codec compatibility is the leading explanation; latest failing
video files have not been supplied. Google Drive may also need processing time.
FFmpeg +faststart moves the MP4 index to the front for playback:
https://ffmpeg.org/ffmpeg-formats.html#mov_002c-mp4_002c-ismv

Future completed rollouts now close custom camera writers immediately at terminal
steps rather than relying only on worker close/reset. Post-rollout preparation
creates *_playback.mp4 copies of custom and official videos, with H.264,
yuv420p, faststart, and a full-frame decoding check. Encoding/validation happens
on local temporary disk before copying to Drive. Source recordings are preserved.
Failures are recorded in video_playback_report.json and do not invalidate task
metrics. A missing MP4 index from an interrupted recording cannot be reliably
recovered by transcoding; errors will identify such files.

Simulation cell wrappers ensure ffmpeg exists without changing Python or policy
packages. No controller, camera transforms, frame rate, prompts, gains, metrics
or model behavior changes. Use *_playback.mp4 for browser/Drive playback.

## Repair the latest saved phone-constraints run without simulation

```bash
%%bash
set -euo pipefail
cd /content/humanoid-internship-challenge
git fetch origin
git switch upright-mug-orientation
git pull --ff-only origin upright-mug-orientation
command -v ffmpeg >/dev/null || { apt-get update -qq; apt-get install -y -qq ffmpeg; }
OUTPUT="$(cat /content/bowl_phone_constraints_last_output.txt)"
/content/libero-official-env/bin/python scripts/prepare_bowl_videos.py --root "$OUTPUT"
printf 'Playback copies and video_playback_report.json: %s\n' "$OUTPUT"
```

If the runtime restarted and the pointer file is absent, replace OUTPUT with the
exact saved .../bowl_hybrid_phone_constraints_<ID>/C directory on mounted Drive.
Do not run a simulation to repair an existing video. Conversion preserves originals.

Modified files: scripts/prepare_bowl_videos.py (new), scripts/evaluate_bowl_hybrid.py
(finalization/postprocessing), both Colab simulation wrappers and the synchronized
synthetic notebook/documentation. Local encoder verification uses an isolated
FFmpeg binary in ignored results; no project Python dependency was added.

Validation: all 60 bowl regression tests passed. All three supplied archived videos converted and fully decoded: agent/wrist 159 frames each, official evaluator 158 frames. No simulation was run. Latest Drive videos and actual Drive browser playback remain untested locally.
