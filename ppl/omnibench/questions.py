"""Encoder-observable QA. Oracle metadata is used only for grading/provenance."""
import hashlib
import re
import numpy as np

VERSION = "omnibench-qa-v3"


def window_indices(seed, n):
    pairs = [(i, j) for i in [0, n // 4, n // 2, 3 * n // 4]
             for j in [n // 4, n // 2, 3 * n // 4, n - 1] if j > i]
    value = int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], "big")
    return pairs[value % len(pairs)]


def make_questions(path, task, state, physics, tactile, audio):
    ts = state['timestamp_s']
    i, j = window_indices(task['seed'], len(ts))
    full = [float(ts[0]), float(ts[-1])]
    interval = [float(ts[i]), float(ts[j])]
    wording = f"from {ts[i]:.3f} s to {ts[j]:.3f} s"
    out = []

    def add(template, branch, text, answer, fields, evidence=full, tolerance=None,
            primary=True, note=None):
        out.append(dict(template_id=template, category=branch, encoder_branch=branch,
                        question=text, expert_answer=str(answer),
                        answer_type='number' if tolerance is not None else 'string',
                        numeric_tolerance=tolerance, answer_source=fields,
                        input_fields=fields, input_modalities=[branch],
                        evidence_modalities=[branch], evidence_time_range_s=evidence,
                        primary_candidate=primary, diagnostic_reason=note,
                        protocol=VERSION, input_mode='isolated_encoder',
                        evidence_independence_claimed=False))

    # Presence check only; the TTS transcript and style are never RGB model inputs.
    with np.load(path / 'multiview_rgb_depth_seg_flow.npz', allow_pickle=False) as z:
        rgb = z['front_rgb'][::12].astype(np.float32) / 255
        source = np.asarray(task['style']['target'][:3])
        chroma = rgb / np.maximum(rgb.sum(-1, keepdims=True), .01)
        target_chroma = source / source.sum()
        visible = bool(np.any((np.linalg.norm(chroma-target_chroma, axis=-1) < .12)
                              & (rgb.max(-1)-rgb.min(-1) > .15)))
        add('rgb_cube_color', 'vision',
            'What color is the cube being manipulated in the front-camera video? Answer with one color word.',
            task['style']['target_name'], ['multiview_rgb_depth_seg_flow.npz:front_rgb'],
            primary=visible, note=None if visible else 'target color not verified in sampled RGB')
        depth = z['front_depth_mm'][0, 60:84, 84:108].astype(np.float32)
        valid = depth[depth > 0]
        if valid.size:
            add('depth_central_patch', 'depth',
                'At the first front-camera frame, what is the median optical-axis depth in centimeters in pixel rows 60:84 and columns 84:108 (zero-based, end-exclusive)? Ignore zero depths.',
                f'{np.median(valid)/10:.1f}', ['multiview_rgb_depth_seg_flow.npz:front_depth_mm'],
                [0., 0.], tolerance=1.)
        flow = z['front_optical_flow_rgb_estimated_px'][i:j+1, 48:112, 48:144].astype(np.float32)
        magnitude = np.linalg.norm(flow, axis=-1)
        moving = flow[magnitude > .2]
        if moving.size:
            dx, dy = np.median(moving, axis=0)
            answer = ('right' if dx > 0 else 'left') if abs(dx) >= abs(dy) else ('down' if dy > 0 else 'up')
        else:
            answer = 'still'
        add('flow_patch_direction', 'flow',
            f'For estimated front-camera flow in rows 48:112 and columns 48:144 {wording}, what is the dominant median direction among vectors larger than 0.2 pixels: left, right, up, down, or still if none?',
            answer, ['multiview_rgb_depth_seg_flow.npz:front_optical_flow_rgb_estimated_px'], interval)

    words = re.findall(r'\b(blue|green|orange|magenta|red|yellow|gray|grey)\b', task['instruction'].lower())
    if words:
        add('speech_first_color', 'speech',
            'What is the first color word spoken in the instruction recording? Answer with one color word.',
            words[0], ['spoken_instruction.wav'], [0., None],
            note='Gold from source TTS transcript; not an independent ASR verification.')
        out[-1]['answer_source'] = ['task.json:instruction (TTS source transcript)']

    with np.load(path / 'pointcloud_world.npz', allow_pickle=False) as z:
        points = np.concatenate([z[f'{v}_world_xyz_m'][0] for v in ['front','overhead','side']])
        keep = ((points[:,0] >= .25) & (points[:,0] <= .75)
                & (abs(points[:,1]) <= .35) & (points[:,2] >= .02) & (points[:,2] <= .5))
        if keep.any():
            add('pointcloud_roi_centroid_x', 'points',
                'At the first frame, combine all three observed point clouds. Within world X=[0.25,0.75], Y=[-0.35,0.35], Z=[0.02,0.50] meters, what is the sampled-surface centroid X in centimeters?',
                f'{points[keep,0].mean()*100:.1f}',
                [f'pointcloud_world.npz:{v}_world_xyz_m' for v in ['front','overhead','side']], [0.,0.], tolerance=1.)
    vertices = np.asarray([[float(v) for v in line.split()[1:4]]
                           for line in (path/'target_cube_local.obj').read_text().splitlines() if line.startswith('v ')])
    add('mesh_longest_extent', 'mesh',
        'What is the longest axis-aligned bounding-box dimension of the supplied local object mesh, in centimeters?',
        f'{np.ptp(vertices,axis=0).max()*100:.2f}', ['target_cube_local.obj'], tolerance=.15,
        note='Procedural cube family only; this is a geometry probe, not diverse-mesh recognition.')
    ee = state['ee_position_world_m']
    add('ee_window_xy_displacement', 'timeseries',
        f"What is the end-effector's straight-line XY displacement in centimeters {wording}?",
        f'{np.linalg.norm(ee[j,:2]-ee[i,:2])*100:.1f}',
        ['robot_object_state.npz:ee_position_world_m'], interval, tolerance=1.)
    add('ee_window_height_change', 'timeseries',
        f"What is the end-effector's signed Z change in centimeters {wording} (positive upward)?",
        f'{(ee[j,2]-ee[i,2])*100:.1f}', ['robot_object_state.npz:ee_position_world_m'], interval, tolerance=1.)
    add('ee_initial_x', 'proprio', 'What is the first recorded end-effector world X position in centimeters?',
        f'{ee[0,0]*100:.1f}', ['robot_object_state.npz:ee_position_world_m'], [0.,0.], tolerance=1.)
    add('gripper_window_opening', 'proprio',
        f'What is the minimum sum of both observed finger joint positions in millimeters {wording}?',
        f'{state["finger_position_m"][i:j+1].sum(-1).min()*1000:.1f}',
        ['robot_object_state.npz:finger_position_m'], interval, tolerance=1.)
    pressure = tactile['pressure_2pads_PROXY'][i:j+1]
    add('tactile_window_peak', 'tactile',
        f'What is the maximum recorded tactile-proxy value across both pads {wording}, in proxy units?',
        f'{pressure.max():.2f}', ['tactile_proxy.npz:pressure_2pads_PROXY'], interval, tolerance=.05)
    bursts = audio['bursts']
    add('audio_window_bursts', 'impact', f'How many audible synthetic bursts begin {wording}?',
        sum(interval[0] <= b[0] <= interval[1] for b in bursts), ['impact_audio_SYNTHETIC.wav'], interval)
    if bursts:
        add('audio_first_onset', 'impact', 'At what time in seconds does the first audible synthetic burst begin?',
            f'{bursts[0][0]:.2f}', ['impact_audio_SYNTHETIC.wav'], bursts[0], tolerance=.03)
    select = (physics['time_s'] >= interval[0]-1e-6) & (physics['time_s'] <= interval[1]+1e-6)
    force = physics['robot_object_contact_N'][select]
    add('recorded_contact_window', 'force',
        f'Does the recorded 240 Hz robot-object contact signal exceed 0.000001 N at any sample {wording}?',
        'yes' if np.any(force > 1e-6) else 'no', ['physics_240hz.npz:robot_object_contact_N'], interval)
    add('recorded_force_peak', 'force',
        f'What is the maximum reported robot-object contact force in newtons {wording}? This signal is uncalibrated simulation contact and excludes assisted attachment force.',
        f'{force.max():.1f}', ['physics_240hz.npz:robot_object_contact_N'], interval, tolerance=25.)
    active=np.flatnonzero(np.any(tactile['pressure_2pads_PROXY']>1e-6,axis=(1,2,3)))
    if active.size and bursts:
        onset=float(ts[active[0]])
        add('audio_tactile_onset_difference','cross_modal',
            'What is the first audible synthetic burst onset minus the first nonzero tactile-proxy frame time, in seconds? Positive means the sound begins later.',
            f'{bursts[0][0]-onset:.2f}',
            ['impact_audio_SYNTHETIC.wav','tactile_proxy.npz:pressure_2pads_PROXY'],
            [min(onset,bursts[0][0]),max(onset,bursts[0][0])],tolerance=.1,
            note='Both are synthetic event-derived signals; this does not establish independent physical sensing.')
        out[-1]['input_modalities']=['impact','tactile']
        out[-1]['evidence_modalities']=['impact','tactile']
        out[-1]['input_mode']='restricted_encoder_pair'
    for q in out:
        if q['evidence_time_range_s'][1] is None:
            q['evidence_time_range_s'][1] = audio.get('speech_duration_s', 8.)
        q['timestamp_fields'] = ['timestamp_s or time_s from the same recorded sensor stream']
    return out
