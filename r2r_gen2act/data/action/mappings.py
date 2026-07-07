from __future__ import annotations

import numpy as np


def _camera_extrinsic_pose(payload: dict, mapping_cfg: dict | None = None) -> np.ndarray:
    mapping_cfg = mapping_cfg or {}
    extrinsics = payload.get("calibration", {}).get("extrinsics", {})
    key = str(mapping_cfg.get("camera_extrinsics_key", "") or "")
    if key:
        if key not in extrinsics:
            raise ValueError(f"camera_extrinsics_key={key!r} not found in calibration.extrinsics")
        pose = extrinsics[key]
    else:
        candidates = [v for v in extrinsics.values() if isinstance(v, list) and len(v) == 6]
        if len(candidates) != 1:
            raise ValueError(
                "Expected exactly one 6D camera extrinsic in calibration.extrinsics; "
                "set action.mapping.camera_extrinsics_key to disambiguate"
            )
        pose = candidates[0]
    return np.asarray(pose, dtype=np.float64)


def _base_to_camera_rotation(payload: dict, mapping_cfg: dict | None = None) -> np.ndarray:
    from r2r_gen2act.data.action.rotation import euler_to_matrix

    mapping_cfg = mapping_cfg or {}
    pose = _camera_extrinsic_pose(payload, mapping_cfg)
    r = euler_to_matrix(pose[3:6])
    convention = str(mapping_cfg.get("extrinsics_convention", "camera_pose_in_base"))
    if convention == "camera_pose_in_base":
        # The 6D extrinsic is the camera pose in robot-base coordinates, so invert rotation.
        return r.T
    if convention == "base_to_camera":
        return r
    raise ValueError(f"Unknown extrinsics_convention={convention!r}")


def droid_action(payload: dict, step: int, mapping_type: str = "droid_actions_first6_plus_gripper", future_horizon: int = 0, chunk_size: int = 1, mapping_cfg: dict | None = None) -> np.ndarray:
    if mapping_type == "droid_actions_first6_plus_gripper":
        arr = np.asarray(payload["actions"][step], dtype=np.float32)
        if arr.shape[0] < 7:
            raise ValueError(f"DROID actions[{step}] has shape {arr.shape}, expected at least 7")
        return arr[:7]
    if mapping_type in ("droid_observation_cartesian_future_delta_pose6d", "droid_observation_cartesian_future_delta_pose6d_camera"):
        # Future-delta action chunk with 6D continuous rotation. For step k (0..N-1) the target is
        # the cumulative delta from the current pose to step + (k+1)*future_horizon:
        #   [dx, dy, dz, r00,r10,r20, r01,r11,r21] + gripper.
        # The `_camera` variant expresses both translation and relative rotation in the source
        # camera frame using calibration.extrinsics; the legacy variant stays in robot-base frame.
        from r2r_gen2act.data.action.rotation import euler_delta_to_6d, euler_to_matrix, matrix_to_6d

        obs = np.asarray(payload["observations"]["cartesian_position"], dtype=np.float32)
        grip_seq = payload.get("observations", {}).get("gripper_position")
        if grip_seq is None:
            grip_seq = payload.get("action_dict", {}).get("gripper_position", [[0.0]])
        grip_seq = np.asarray(grip_seq, dtype=np.float32).reshape(len(obs), -1)
        use_camera_frame = mapping_type.endswith("_camera")
        r_base_to_cam = _base_to_camera_rotation(payload, mapping_cfg) if use_camera_frame else None
        n = max(1, int(chunk_size))
        rows = []
        for k in range(n):
            future_step = min(int(step) + (k + 1) * max(1, int(future_horizon)), len(obs) - 1)
            dxyz = (obs[future_step, :3] - obs[step, :3]).astype(np.float32)
            if use_camera_frame:
                assert r_base_to_cam is not None
                dxyz = (r_base_to_cam @ dxyz.astype(np.float64)).astype(np.float32)
                r_cur = euler_to_matrix(obs[step, 3:6])
                r_fut = euler_to_matrix(obs[future_step, 3:6])
                r_delta_cam = r_base_to_cam @ (r_fut @ r_cur.T) @ r_base_to_cam.T
                d6 = matrix_to_6d(r_delta_cam)
            else:
                d6 = euler_delta_to_6d(obs[step, 3:6], obs[future_step, 3:6])
            grip = grip_seq[future_step, 0]
            rows.append(np.concatenate([dxyz, d6, np.asarray([grip], dtype=np.float32)]))
        chunk = np.stack(rows, axis=0).astype(np.float32)
        return chunk[0] if n == 1 else chunk
    if mapping_type == "droid_observation_cartesian_future_delta_pose":
        # Full 6-DOF future delta: [dx, dy, dz, droll, dpitch, dyaw] + gripper.
        # Rotation is a euler-angle delta wrapped to [-pi, pi]; naive subtraction is
        # broken because roll wraps at +/-pi (raw deltas would jump by ~2*pi).
        obs = np.asarray(payload["observations"]["cartesian_position"], dtype=np.float32)
        future_step = min(int(step) + max(1, int(future_horizon)), len(obs) - 1)
        delta = np.empty(6, dtype=np.float32)
        delta[:3] = obs[future_step, :3] - obs[step, :3]
        drot = obs[future_step, 3:6] - obs[step, 3:6]
        delta[3:6] = np.arctan2(np.sin(drot), np.cos(drot))
        grip_seq = payload.get("observations", {}).get("gripper_position")
        if grip_seq is None:
            grip_seq = payload.get("action_dict", {}).get("gripper_position", [[0.0]])
        grip = np.asarray(grip_seq[future_step], dtype=np.float32).reshape(-1)[0]
        return np.concatenate([delta, np.asarray([grip], dtype=np.float32)])
    action_dict = payload.get("action_dict", {})
    if mapping_type == "droid_action_dict_cartesian_position":
        pose = np.asarray(action_dict["cartesian_position"][step], dtype=np.float32)
    elif mapping_type == "droid_action_dict_cartesian_delta_position":
        target_pose = np.asarray(action_dict["cartesian_position"][step], dtype=np.float32)
        obs_pose = np.asarray(payload["observations"]["cartesian_position"][step], dtype=np.float32)
        pose = target_pose.copy()
        pose[:3] = target_pose[:3] - obs_pose[:3]
    elif mapping_type == "droid_observation_cartesian_future_delta_position":
        obs = np.asarray(payload["observations"]["cartesian_position"], dtype=np.float32)
        future_step = min(int(step) + max(1, int(future_horizon)), len(obs) - 1)
        pose = obs[step].copy()
        pose[:3] = obs[future_step, :3] - obs[step, :3]
    elif mapping_type == "droid_action_dict_cartesian_velocity":
        pose = np.asarray(action_dict["cartesian_velocity"][step], dtype=np.float32)
    else:
        raise ValueError(f"Unknown DROID action mapping: {mapping_type}")
    grip = np.asarray(action_dict.get("gripper_position", [[0.0]])[step], dtype=np.float32).reshape(-1)[0]
    return np.concatenate([pose[:6], np.asarray([grip], dtype=np.float32)])


def toto_action(payload: dict, step: int) -> np.ndarray:
    world = np.asarray(payload["world_vector"][step], dtype=np.float32)
    rot = np.asarray(payload["rotation_delta"][step], dtype=np.float32)
    grip = np.asarray([float(payload["open_gripper"][step])], dtype=np.float32)
    return np.concatenate([world[:3], rot[:3], grip])


def terminate_from_payload(payload: dict, step: int, num_steps: int, positive_window: int = 5) -> int:
    if "terminate_episode" in payload:
        return int(float(payload["terminate_episode"][step]) > 0.5)
    if "is_terminal" in payload and bool(payload["is_terminal"][step]):
        return 1
    if "is_last" in payload and bool(payload["is_last"][step]):
        return 1
    return int(step >= max(0, num_steps - positive_window))
