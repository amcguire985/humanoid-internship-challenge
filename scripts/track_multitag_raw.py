"""Raw multi-tag observations; stationary registration is fixed, never gap-filled."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def register_world(frames, world_ids=(0, 1, 2, 3)):
    """First co-observation on each graph edge registers stationary tags to ID0."""
    edges = {}
    for observations in frames:
        visible = {i: observations[i][0]['transform'] for i in world_ids
                   if len(observations.get(i, [])) == 1 and observations[i][0] is not None}
        for i in visible:
            for j in visible:
                if i != j and (i, j) not in edges:
                    edges[i, j] = np.linalg.inv(visible[i]) @ visible[j]
    layout = {0: np.eye(4)}
    while True:
        previous = len(layout)
        for (i, j), transform in edges.items():
            if i in layout and j not in layout:
                layout[j] = layout[i] @ transform
        if len(layout) == previous:
            return layout


def camera_world(observations, layout):
    candidates = [(observations[i][0]['error'], i, observations[i][0]['transform'])
                  for i in layout if len(observations.get(i, [])) == 1 and observations[i][0] is not None]
    if not candidates:
        return None, None
    _, tag_id, pose = min(candidates, key=lambda value: (value[0], value[1]))
    return pose @ np.linalg.inv(layout[tag_id]), tag_id


def run(a):
    from track_apriltags import Standard41Detector, estimate_pose, fit_intrinsics, pose_fields, plot_trajectory
    if a.cube_edge is not None and (not np.isfinite(a.cube_edge) or a.cube_edge <= 0):
        raise ValueError('Cube edge must be finite and positive')
    if not np.isfinite(a.hand_edge) or a.hand_edge <= 0:
        raise ValueError('Hand edge must be finite and positive')
    if a.reference_id != 0:
        raise ValueError('Multi-tag raw mode uses ID0 as the origin')
    groups = {'world': a.world_ids, 'object': a.cube_ids, 'hand': a.hand_ids,
              'target': [] if a.target_id is None else [a.target_id]}
    all_ids = [i for ids in groups.values() for i in ids]
    if len(set(all_ids)) != len(all_ids) or any(i < 0 for i in all_ids):
        raise ValueError('Tag groups must have distinct nonnegative IDs')
    if 0 not in a.world_ids:
        raise ValueError('World IDs must include ID0 as the origin')
    sizes = {i: a.world_size if group == 'world' else
             (a.target_size if a.target_size is not None else a.world_size) if group == 'target'
             else a.body_size for group, ids in groups.items() for i in ids}
    if any(not np.isfinite(size) or size <= 0 for size in sizes.values()):
        raise ValueError('Tag sizes must be positive')
    if a.tag_size_convention == 'full-pattern':
        if a.family != 'Standard41h12':
            raise ValueError('Full-pattern conversion requires Standard41h12')
        sizes = {i: size * 5 / 9 for i, size in sizes.items()}
    calibration = json.loads(a.calibration.read_text(encoding='utf-8-sig'))
    cap = cv2.VideoCapture(str(a.video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, frame = cap.read()
    if not ok or not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError(f'Cannot decode video with usable frame rate: {a.video}')
    height, width = frame.shape[:2]
    matrix = fit_intrinsics(calibration['camera_matrix'], calibration['image_size'], (width, height), a.calibration_fit)
    distortion = np.array(calibration['distortion_coefficients'], dtype=float)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = Standard41Detector() if a.family == 'Standard41h12' else cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, 'DICT_APRILTAG_' + a.family)), params)
    a.output.mkdir(parents=True, exist_ok=True)
    frames, times, raw = [], [], []
    writer = None
    if not a.no_video:
        writer = cv2.VideoWriter(str(a.output / 'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError('Cannot open annotated video writer')
    try:
        while ok:
            index = len(frames)
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            observations = {}
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                instances = {}
                for corner, tag_id in zip(corners, ids.ravel()):
                    tag_id = int(tag_id)
                    if tag_id not in sizes:
                        continue
                    instance = instances.get(tag_id, 0)
                    instances[tag_id] = instance + 1
                    pose = estimate_pose(corner, sizes[tag_id], matrix, distortion)
                    record = dict(frame=index, time_s=timestamp, tag_id=tag_id, instance=instance,
                                  status='observed' if pose is not None else 'invalid_pose')
                    observations.setdefault(tag_id, []).append(pose)
                    if pose is not None:
                        record.update(error_px=pose['error'], ambiguity_gap_px=pose['ambiguity_gap'])
                        record.update({'camera_' + k: v for k, v in pose_fields(pose['transform']).items()})
                        cv2.drawFrameAxes(frame, matrix, distortion, pose['rvec'], pose['tvec'], sizes[tag_id] * .6)
                    raw.append(record)
            frames.append(observations)
            times.append(timestamp)
            cv2.putText(frame, f'{timestamp:.2f}s | world {a.world_ids} object {a.cube_ids} target {a.target_id}', (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
            if index == 0:
                cv2.imwrite(str(a.output / 'preview.jpg'), frame)
            if writer is not None:
                writer.write(frame)
            if len(frames) % 100 == 0:
                print(f'Processed {len(frames)} frames', flush=True)
            ok, frame = cap.read()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    layout = register_world(frames, a.world_ids)
    trajectory_fields = ['frame', 'time_s', 'status', 'world_source_id', 'detection_count',
                         'error_px', 'ambiguity_gap_px', *pose_fields(np.eye(4))]
    def write_csv(path, rows, fields):
        with path.open('w', newline='', encoding='utf-8') as handle:
            out = csv.DictWriter(handle, fieldnames=fields)
            out.writeheader()
            out.writerows(rows)
    # All detections, including repeated IDs, retain their own camera-frame pose.
    raw_fields = ['frame', 'time_s', 'tag_id', 'instance', 'status', 'error_px', 'ambiguity_gap_px',
                  *['camera_' + key for key in pose_fields(np.eye(4))], 'world_source_id',
                  *['world_' + key for key in pose_fields(np.eye(4))]]
    for record in raw:
        world, source_id = camera_world(frames[record['frame']], layout)
        if world is not None and record['status'] == 'observed':
            # Reconstruct the transform from exported position and Rodrigues vector.
            camera = np.eye(4)
            camera[:3, 3] = [record['camera_' + key + '_m'] for key in 'xyz']
            camera[:3, :3] = cv2.Rodrigues(np.array([record['camera_relative_' + key + '_rad'] for key in ('rx', 'ry', 'rz')]))[0]
            record.update(world_source_id=source_id)
            record.update({'world_' + k: v for k, v in pose_fields(np.linalg.inv(world) @ camera).items()})
    write_csv(a.output / 'detections.csv', raw, raw_fields)
    coverage = {}
    for tag_id in groups['object'] + groups['hand'] + groups['target']:
        rows = []
        for index, observations in enumerate(frames):
            world, source_id = camera_world(observations, layout)
            poses = observations.get(tag_id, [])
            row = dict(frame=index, time_s=times[index], detection_count=len(poses), world_source_id=source_id,
                       status='missing_tag' if not poses else 'duplicate_id' if len(poses) > 1 else 'invalid_pose' if poses[0] is None else 'missing_world')
            if world is not None and len(poses) == 1 and poses[0] is not None:
                row.update(status='tracked', error_px=poses[0]['error'], ambiguity_gap_px=poses[0]['ambiguity_gap'])
                row.update(pose_fields(np.linalg.inv(world) @ poses[0]['transform']))
            rows.append(row)
        group = next(group for group, ids in groups.items() if tag_id in ids)
        output = a.output / group / f'id{tag_id}'
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / 'trajectory.csv', rows, trajectory_fields)
        plot_trajectory(rows, output, f'ID{tag_id} tag centre (raw)', 'ID0 world')
        coverage[tag_id] = sum(row['status'] == 'tracked' for row in rows)
    summary = dict(video=str(a.video), calibration=str(a.calibration), frames_processed=len(frames),
                   tag_groups=groups, family=a.family, tag_size_convention=a.tag_size_convention, pose_tag_sizes_m=sizes,
                   input_world_size_m=a.world_size, input_body_size_m=a.body_size,
                   registered_world_ids=sorted(layout), world_tag_transforms={i: t.tolist() for i, t in layout.items()},
                   tracked_frames_by_id=coverage, gap_filling=False, temporal_filtering=False,
                   reprojection_threshold_rejection=False, hand_object_inference=False,
                   video_size=[width, height], fps=fps, calibration_fit=a.calibration_fit,
                   input_target_size_m=a.target_size if a.target_size is not None else a.world_size,
                   effective_camera_matrix=matrix.tolist(), distortion_coefficients=distortion.tolist(),
                   world_registration='First co-visible positive-depth poses per edge; fixed registration to ID0. Lowest-error current world tag supplies camera pose; no temporal averaging.',
                   limitations=['Metric accuracy depends on calibrated recording mode, crop model and measured tag sizes.',
                                'Cube outputs are face-tag centres, not a fused cube centre. Opposite faces share IDs and cannot be identified from ID alone.',
                                'Wrist outputs are individual tag centres, not a shared wrist frame. Curvature violates planar PnP.',
                                'Raw pose errors and planar ambiguity are retained. World-registration errors and source switches can cause offsets.',
                                'Missing observations remain blank. Disconnected world tags cannot supply an ID0 reference.'])
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if a.cube_edge is not None and a.cube_ids:
        export_body_center(a.output, a.cube_edge, a.cube_ids, f'object/id{a.cube_ids[0]}', 'object/cube_center', 'cube_center', 'Cube')
    if a.hand_ids:
        export_body_center(a.output, a.hand_edge, a.hand_ids, f'hand/id{a.hand_ids[0]}', 'hand/hand_center', 'hand_center', 'Hand')
    print(f'Raw tracking saved to {a.output}; coverage {coverage}; world IDs {sorted(layout)}', flush=True)


def cube_center(transform, edge):
    """Move inward from a centred face tag along its outward decoded z axis."""
    if not np.isfinite(edge) or edge <= 0:
        raise ValueError('Cube edge must be finite and positive')
    return transform[:3, 3] - transform[:3, 2] * (edge / 2)


def export_cube_center(output, edge):
    return export_body_center(output, edge, (4, 5, 6), 'object/id4', 'object/cube_center', 'cube_center', 'Cube')


def export_hand_center(output, edge=0.045):
    return export_body_center(output, edge, (7, 8), 'hand/id7', 'hand/hand_center', 'hand_center', 'Hand')


def export_body_center(output, edge, tag_ids, timing_path, destination_path, summary_key, label):
    """Derive raw centre positions from cached detections without rerunning detection."""
    from track_apriltags import plot_trajectory
    if not np.isfinite(edge) or edge <= 0:
        raise ValueError('Cube edge must be finite and positive')
    output = Path(output)
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    candidates = {}
    with (output / 'detections.csv').open(newline='', encoding='utf-8') as handle:
        for record in csv.DictReader(handle):
            if int(record['tag_id']) not in tag_ids or not record.get('world_x_m'):
                continue
            transform = np.eye(4)
            transform[:3, 3] = [float(record['world_' + axis + '_m']) for axis in 'xyz']
            rotation = np.array([float(record['world_relative_' + axis + '_rad']) for axis in ('rx', 'ry', 'rz')])
            transform[:3, :3] = cv2.Rodrigues(rotation)[0]
            candidates.setdefault(int(record['frame']), []).append((record, cube_center(transform, edge)))
    # Use an existing per-frame table to preserve decoder timestamps, including missing frames.
    with (output / timing_path / 'trajectory.csv').open(newline='', encoding='utf-8') as handle:
        timing = list(csv.DictReader(handle))
    rows = []
    for sample in timing:
        frame = int(sample['frame'])
        visible = candidates.get(frame, [])
        row = dict(frame=frame, time_s=float(sample['time_s']),
                   status=('missing_hand' if summary_key == 'hand_center' else 'missing_object') if not visible else 'tracked',
                   candidate_count=len(visible),
                   candidate_tag_ids=';'.join(str(item[0]['tag_id']) for item in visible))
        if not visible and sample.get('world_source_id', '') == '':
            row['status'] = 'missing_world'
        if visible:
            record, centre = min(visible, key=lambda item: (float(item[0]['error_px']), int(item[0]['tag_id']), int(item[0]['instance'])))
            row.update(source_tag_id=int(record['tag_id']), source_instance=int(record['instance']),
                       world_source_id=record['world_source_id'], error_px=record['error_px'],
                       ambiguity_gap_px=record['ambiguity_gap_px'],
                       x_m=float(centre[0]), y_m=float(centre[1]), z_m=float(centre[2]),
                       distance_m=float(np.linalg.norm(centre)),
                       max_candidate_disagreement_m=max(float(np.linalg.norm(a[1] - b[1])) for a in visible for b in visible))
        rows.append(row)
    destination = output / destination_path
    destination.mkdir(parents=True, exist_ok=True)
    fields = ['frame', 'time_s', 'status', 'candidate_count', 'candidate_tag_ids',
              'source_tag_id', 'source_instance', 'world_source_id', 'error_px',
              'ambiguity_gap_px', 'max_candidate_disagreement_m', 'x_m', 'y_m', 'z_m', 'distance_m']
    with (destination / 'trajectory.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plot_trajectory(rows, destination, f'{label} centre (raw)', 'ID0 world')
    summary[summary_key] = dict(edge_m=edge, frames_tracked=sum(r['status'] == 'tracked' for r in rows),
                                  method=f'Face centre minus {edge * 500:g} mm outward normal; select lowest-reprojection-error current face, no averaging or temporal filtering.',
                                  face_labels=({'4': 'initial camera-facing face', '5': 'initial top face', '6': 'third orthogonal face'}
                                               if summary_key == 'cube_center' else {'7': 'wrist cube face', '8': 'wrist cube face'}),
                                  assumptions=['Tags centred on each face and flush with the cube surface.',
                                               'Decoded tag z axis points outward on every face, including opposite faces.'],
                                  orientation_exported=False,
                                  orientation_reason='Decoded in-plane mounting rotations and face association are unspecified.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f"{label} centre: {summary[summary_key]['frames_tracked']}/{len(rows)} frames; {destination}")
