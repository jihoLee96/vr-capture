"""Robust SteamVR motion capture logger.

This script polls the SteamVR runtime directly (background application type)
so it is independent from any specific game's input handling. It captures:
 - HMD pose (position + rotation)
 - Controller pose and full button/trigger/touch state
 - Tracking reference and generic tracker poses

Records are written as JSON lines, one frame per line, to make postprocessing easy.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from typing import Dict, List, Optional

import openvr


def _rotation_to_quaternion(matrix: List[float]) -> Dict[str, float]:
    """Convert a 3x4 row-major pose matrix into a quaternion.

    SteamVR returns 3 rows of 4 values (3x4). The final column is translation.
    """

    m00, m01, m02, _ = matrix[0:4]
    m10, m11, m12, _ = matrix[4:8]
    m20, m21, m22, _ = matrix[8:12]

    trace = m00 + m11 + m22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    return {
        "x": qx,
        "y": qy,
        "z": qz,
        "w": qw,
    }


def _pose_to_dict(pose) -> Optional[Dict]:
    if pose is None or not pose.bDeviceIsConnected:
        return None

    matrix = [val for row in pose.mDeviceToAbsoluteTracking for val in row]
    return {
        "valid": bool(pose.bPoseIsValid),
        "position": {
            "x": pose.mDeviceToAbsoluteTracking[0][3],
            "y": pose.mDeviceToAbsoluteTracking[1][3],
            "z": pose.mDeviceToAbsoluteTracking[2][3],
        },
        "quaternion": _rotation_to_quaternion(matrix),
    }


def _device_class_name(class_enum: int) -> str:
    return {
        openvr.TrackedDeviceClass_HMD: "HMD",
        openvr.TrackedDeviceClass_Controller: "Controller",
        openvr.TrackedDeviceClass_GenericTracker: "Tracker",
        openvr.TrackedDeviceClass_TrackingReference: "BaseStation",
        openvr.TrackedDeviceClass_DisplayRedirect: "DisplayRedirect",
    }.get(class_enum, f"Unknown-{class_enum}")


def _controller_state_to_dict(state: openvr.VRControllerState_t) -> Dict:
    # Boolean buttons
    buttons = {
        "system": bool(state.ulButtonPressed & openvr.ButtonMask.from_id(openvr.k_EButton_System)),
        "application_menu": bool(
            state.ulButtonPressed & openvr.ButtonMask.from_id(openvr.k_EButton_ApplicationMenu)
        ),
        "grip": bool(state.ulButtonPressed & openvr.ButtonMask.from_id(openvr.k_EButton_Grip)),
        "dpad_left": bool(state.ulButtonPressed & openvr.ButtonMask.from_id(openvr.k_EButton_DPad_Left)),
        "dpad_up": bool(state.ulButtonPressed & openvr.ButtonMask.from_id(openvr.k_EButton_DPad_Up)),
        "dpad_right": bool(state.ulButtonPressed & openvr.ButtonMask.from_id(openvr.k_EButton_DPad_Right)),
        "dpad_down": bool(state.ulButtonPressed & openvr.ButtonMask.from_id(openvr.k_EButton_DPad_Down)),
        "a_x": bool(state.ulButtonPressed & openvr.ButtonMask.from_id(openvr.k_EButton_A)),
        "proximity_sensor": bool(
            state.ulButtonPressed & openvr.ButtonMask.from_id(openvr.k_EButton_ProximitySensor)
        ),
        "touchpad_click": bool(state.ulButtonPressed & openvr.ButtonMask.from_id(openvr.k_EButton_SteamVR_Touchpad)),
        "trigger_click": bool(state.ulButtonPressed & openvr.ButtonMask.from_id(openvr.k_EButton_SteamVR_Trigger)),
    }

    touched = {
        "touchpad": bool(state.ulButtonTouched & openvr.ButtonMask.from_id(openvr.k_EButton_SteamVR_Touchpad)),
        "trigger": bool(state.ulButtonTouched & openvr.ButtonMask.from_id(openvr.k_EButton_SteamVR_Trigger)),
        "thumbstick": bool(state.ulButtonTouched & openvr.ButtonMask.from_id(openvr.k_EButton_ApplicationMenu)),
    }

    axes = [
        {"x": state.rAxis[i].x, "y": state.rAxis[i].y}
        for i in range(openvr.k_unControllerStateAxisCount)
    ]

    return {"buttons": buttons, "touched": touched, "axes": axes}


def poll_frame(vr_system: openvr.IVRSystem, compositor: openvr.IVRCompositor) -> Dict:
    # Wait for poses so we sync with compositor; this keeps us independent from any game's loop.
    render_poses = [openvr.TrackedDevicePose_t() for _ in range(openvr.k_unMaxTrackedDeviceCount)]
    game_poses = [openvr.TrackedDevicePose_t() for _ in range(openvr.k_unMaxTrackedDeviceCount)]
    compositor.waitGetPoses(render_poses, game_poses)
    poses = render_poses
    timestamp = time.time()
    frame: Dict = {"timestamp": timestamp, "devices": []}

    for device_index in range(openvr.k_unMaxTrackedDeviceCount):
        pose = poses[device_index]
        if not pose.bDeviceIsConnected:
            continue

        device_class = vr_system.getTrackedDeviceClass(device_index)
        entry: Dict = {
            "index": device_index,
            "class": _device_class_name(device_class),
            "pose": _pose_to_dict(pose),
        }

        if device_class in (openvr.TrackedDeviceClass_HMD, openvr.TrackedDeviceClass_GenericTracker):
            frame["devices"].append(entry)
            continue

        if device_class == openvr.TrackedDeviceClass_Controller:
            try:
                controller_pose = openvr.TrackedDevicePose_t()
                state, controller_pose = vr_system.getControllerStateWithPose(
                    openvr.TrackingUniverseStanding, device_index
                )
                entry["inputs"] = _controller_state_to_dict(state)
                # Prefer the pose from controller state if valid
                if controller_pose.bPoseIsValid:
                    entry["pose"] = _pose_to_dict(controller_pose)
            except openvr.error_code.InitError:
                # If controller is not ready yet, skip inputs but keep pose info.
                pass

        frame["devices"].append(entry)

    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SteamVR motion + input logger")
    parser.add_argument("--output", default="captures.jsonl", help="Destination JSONL file")
    parser.add_argument("--hz", type=float, default=90.0, help="Sampling rate (frames per second)")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration in seconds. Omit to run until Ctrl+C.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    running = True

    def _handle_sigint(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _handle_sigint)

    openvr.init(openvr.VRApplication_Background)
    compositor = openvr.VRCompositor()
    vr_system = openvr.VRSystem()

    frame_time = 1.0 / args.hz if args.hz > 0 else 0.0
    end_time = time.time() + args.duration if args.duration else None

    with open(args.output, "a", encoding="utf-8") as fp:
        while running:
            if end_time and time.time() >= end_time:
                break

            frame = poll_frame(vr_system, compositor)
            fp.write(json.dumps(frame, ensure_ascii=False) + "\n")
            fp.flush()

            if frame_time > 0:
                time.sleep(frame_time)

    openvr.shutdown()


if __name__ == "__main__":
    main()
