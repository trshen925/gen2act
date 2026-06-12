from __future__ import annotations

import numpy as np


def droid_action(payload: dict, step: int, mapping_type: str = "droid_actions_first6_plus_gripper", future_horizon: int = 0) -> np.ndarray:
    if mapping_type == "droid_actions_first6_plus_gripper":
        arr = np.asarray(payload["actions"][step], dtype=np.float32)
        if arr.shape[0] < 7:
            raise ValueError(f"DROID actions[{step}] has shape {arr.shape}, expected at least 7")
        return arr[:7]
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
