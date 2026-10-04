"""Fill bounded position-only trajectory gaps without smoothing or outlier rejection."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def fill_position_gaps(times, positions, max_frames=10, max_span=None):
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (len(times), 3) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Need Nx3 positions and finite, strictly increasing timestamps')
    if max_frames < 0 or (max_span is not None and (not np.isfinite(max_span) or max_span <= 0)):
        raise ValueError('Gap frames must be nonnegative and optional span finite and positive')
    result = positions.copy()
    valid = np.isfinite(positions).all(axis=1)
    labels = np.where(valid, 'observed', 'missing').astype(object)
    gaps = []
    anchors = np.flatnonzero(valid)
    for left, right in zip(anchors[:-1], anchors[1:]):
        length = int(right - left - 1)
        if not length:
            continue
        span = float(times[right] - times[left])
        filled = length <= max_frames and (max_span is None or span <= max_span)
        gaps.append(dict(first_missing_index=int(left + 1), last_missing_index=int(right - 1),
                         missing_frames=length, endpoint_span_s=span, filled=bool(filled)))
        if filled:
            fractions = (times[left+1:right] - times[left]) / span
            result[left+1:right] = ((1 - fractions[:, None]) * positions[left]
                                    + fractions[:, None] * positions[right])
            labels[left+1:right] = 'interpolated'
    return result, labels, gaps


def run(source, output, max_frames=10, max_span=None):
    from track_apriltags import plot_trajectory
    source, output = Path(source), Path(output)
    if output.resolve() == source.parent.resolve():
        raise ValueError('Use a separate output directory to preserve raw results')
    with source.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        rows = list(reader)
    times = np.array([float(r['time_s']) for r in rows])
    positions = np.array([[float(r[a+'_m']) if r.get(a+'_m') else np.nan for a in 'xyz'] for r in rows])
    filled, labels, gaps = fill_position_gaps(times, positions, max_frames, max_span)
    output.mkdir(parents=True, exist_ok=True)
    processed = []
    for i, row in enumerate(rows):
        record = dict(row, source_status=row['status'], pose_source=labels[i])
        if labels[i] == 'interpolated':
            record.update(status='interpolated', distance_m=float(np.linalg.norm(filled[i])))
            record.update({a+'_m':float(filled[i,j]) for j,a in enumerate('xyz')})
        processed.append(record)
    with (output/'interpolated.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields,'source_status','pose_source'])
        writer.writeheader()
        writer.writerows(processed)
    report = dict(source=str(source), frames=len(rows), max_missing_frames=max_frames,
                  max_endpoint_span_s=max_span, smoothing=False, outlier_rejection=False,
                  observed_frames=int(np.sum(labels=='observed')),
                  interpolated_frames=int(np.sum(labels=='interpolated')),
                  remaining_missing_frames=int(np.sum(labels=='missing')),
                  available_frames=int(np.sum(labels!='missing')), gaps=gaps,
                  notes=['Linear position interpolation uses decoder timestamps and future observations.',
                         'Leading/trailing gaps and bounded gaps longer than the limit remain blank.',
                         'Original observations and source provenance are preserved; no orientation is invented.'])
    (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')
    plotting = [dict(r,time_s=float(r['time_s']),**{a+'_m':float(r[a+'_m']) if r.get(a+'_m') else np.nan for a in 'xyz'}) for r in processed]
    plot_trajectory(plotting,output,source.parent.name.replace('_',' ')+' (gap-filled)','ID0 world')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,1,figsize=(12,8),sharex=True)
    for i,axis in enumerate(axes):
        axis.plot(times,filled[:,i]*100,label='Gap-filled (no smoothing)',linewidth=1,color='tab:orange')
        axis.plot(times,positions[:,i]*100,label='Raw',linewidth=1,color='tab:blue')
        interpolated = labels=='interpolated'
        axis.scatter(times[interpolated],filled[interpolated,i]*100,s=9,color='tab:orange',label='Interpolated samples')
        axis.set_ylabel('xyz'[i]+' (cm)')
        axis.grid(alpha=.3)
    axes[0].legend()
    axes[0].set_title(f'{source.parent.name}: up to {max_frames} missing frames, no smoothing')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=160)
    plt.close(fig)
    print(f"{source.parent.name}: {report['observed_frames']} observed + {report['interpolated_frames']} filled = {report['available_frames']}/{len(rows)} available; {output}",flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-gap-frames',type=int,default=10)
    parser.add_argument('--max-gap-span',type=float,help='Optional endpoint time cap; default uses frame limit only')
    args = parser.parse_args()
    run(args.trajectory,args.output or args.trajectory.parent/'gap_filled_10_frames',args.max_gap_frames,args.max_gap_span)


if __name__=='__main__':
    main()
