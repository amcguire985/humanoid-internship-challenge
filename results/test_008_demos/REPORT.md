# test_008 shutter-speed-500 tracking test

Input: videos/test__008_shutterspeed500.MOV. One complete object -> target transfer confirmed in half-second video review, including release before the clip ends.

Raw cube-centre coverage: 131/185 (70.8%). Cleaned coverage: 152/185 (82.2%).

| Metric | Result |
|---|---|
| Start-to-end distance | 24.8 cm |
| World-XY heading | -126.2 degrees |
| Maximum observed world-Z lift | 9.0 cm |
| Observed path length (does not cross gaps) | 44.9 cm |
| Pickup / transport start / transport end / release | 2.167 / 3.500 / 5.125 / 6.750 s |
| Missing carry frames | 28 |
| Longest missing carry interval | 0.333 s |
| Retargeting decision | Reject: tracking gaps |

Individual tracking gaps:

| Start (s) | End (s) | Missing frames | Missing duration (s) |
|---:|---:|---:|---:|
| 4.708 | 5.000 | 8 | 0.333 |
| 5.292 | 5.417 | 4 | 0.167 |
| 5.500 | 5.625 | 4 | 0.167 |
| 5.708 | 5.833 | 4 | 0.167 |
| 6.000 | 6.125 | 4 | 0.167 |
| 6.208 | 6.333 | 4 | 0.167 |

Diagnostic: [compact trajectory/height/speed plot](demo_001/diagnostic.png). Whole-video motion: [overview](full_overview.png). Video review: [half-second contact sheet](phase_review.jpg). Original annotated tracking: [video](../test_008_raw/annotated.mp4).

Exported demo_001 contains processed_demo.csv (smoothed positions with validity/provenance), transport.csv (pickup through release, missing rows retained), raw_trajectory.csv, cleaned_trajectory.csv (accepted measurements before filling/smoothing), normalized_task_trajectory.csv, metadata.json, and diagnostic.png. Resting endpoints are robust measured medians; the fixed target-tag anchor is retained separately.

Automatic stationary detection did not confirm a stable final rest because final poses are missing/noisy. The complete transfer and direction were verified in video and an explicit split was saved. Pickup is motion onset; release is an approximate annotation. Inner transport estimates remain automatic height heuristics. Edit ../../config/test_008_demos.json to override timing.

Reproduce:

```powershell
.\.venv\Scripts\python.exe scripts/extract_transfer_demos.py results/test_008_raw --config config/test_008_demos.json --output results/test_008_demos
```

Tracking and cleaning use unchanged data_001/data_002 settings. No shutter-speed setting was inferred from camera metadata; 500 is the filename label. Motion is never invented across large tracking gaps. Interpolation is bounded by 3 missing frames and 0.15-second endpoint span; SG7 smoothing resets at unresolved gaps. Normalized coordinates retain the same missing rows.

Lift is relative to the start in configured signed world-Z, whose gravity alignment is unverified. Calibration/crop validation remains pending. Observed maximum lift can miss a peak hidden by gaps; observed path length is partial. This short clip cannot isolate the effect of shutter speed because recording length, motion and visible tags differ from the prior videos.

Validation: export audit recomputed validity, missing carry counts and no-gap path sums, and verified all artifacts. No LIBERO execution or training.
