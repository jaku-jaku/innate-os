#!/usr/bin/env python3
"""
Arm Draw Image As Waypoints — traces a raster image on paper using IK waypoints.
"""
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import vtracer
from svgpathtools import svg2paths

from brain_client.skill_types import Interface, InterfaceType, Skill, SkillResult

class ActionType(Enum):
    WAYPOINT = 0
    LIFT = 1
    DROP = 2

@dataclass(frozen = True)
class Waypoint:
    x: float
    y: float

@dataclass(frozen = True)
class Action:
    action_type: ActionType
    waypoint: Optional[Waypoint]

@dataclass
class ArmPose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float



FLATTEN_RESOLUTION = 1.0  # smaller = smoother drawing    

def _flatten_path(path, step=1.0):
    """Convert SVG path (curves included) into points."""
    points = []
    for segment in path:
        length = segment.length()
        num_samples = max(int(length / step), 1)

        for i in range(num_samples + 1):
            t = i / num_samples
            point = segment.point(t)
            points.append((point.real, point.imag))
    return points

def _parse_transform_str(transform_str: str) -> Waypoint:
    match = re.match(r"translate\((\-?\d+\.?\d*),\s*(\-?\d+\.?\d*)\)", transform_str)
    if not match:
        raise ValueError(f"Invalid format: {transform_str}")
    
    x, y = map(float, match.groups())
    return Waypoint(x, y)

def _svg_to_paths(svg_path: str) -> List[List[Waypoint]]:
    paths, attributes = svg2paths(svg_path)
    all_paths = []
    if len(paths) == 0:
        return [[]]
    
    ## TODO(rbenefo): First path isn't always the border. Sometimes,
    ## second path is too.
    for path, attr in list(zip(paths, attributes))[1:]:
        ## Skip first path, which is just the border of the image.
        transform_str = attr.get('transform', '')
        if transform_str != '':
            transform = _parse_transform_str(transform_str)
            
        pts = _flatten_path(path, FLATTEN_RESOLUTION)
        if len(pts) > 1:
            all_paths.append([Waypoint(x + transform.x, y + transform.y) for x, y in pts])
    return all_paths

def _points_to_commands(paths: List[List[Waypoint]]) -> List[Action]:
    commands: List[Action] = []
    pen_down = False
    for path in paths:
        # Move to start point (pen up)
        if pen_down:
            commands.append(Action(ActionType.LIFT, None))
            pen_down = False

        start = path[0]
        commands.append(Action(ActionType.WAYPOINT, Waypoint(start.x, start.y)))

        # Drop pen to start drawing
        commands.append(Action(ActionType.DROP, None))
        pen_down = True

        # Draw rest of path
        for point in path[1:]:
            commands.append(Action(ActionType.WAYPOINT, Waypoint(point.x, point.y)))

    # End with pen up
    if pen_down:
        commands.append(Action(ActionType.LIFT, None))

    return commands


def produce_waypoints(input_path: str) -> List[Action]:
    with tempfile.NamedTemporaryFile(suffix=".svg") as tmp:
        vtracer.convert_image_to_svg_py(input_path, tmp.name)
        paths = _svg_to_paths(tmp.name)
    actions = _points_to_commands(paths)
    return actions


_DEFAULT_IMAGE = "draw_waypoints/demo/patrick.png"


def _resolve_image_path(relative: str) -> str:
    """Resolve a demo image path relative to ~/skills or $INNATE_OS_ROOT/skills."""
    candidates = [
        Path.home() / "skills" / relative,
        Path(os.environ.get("INNATE_OS_ROOT", Path.home() / "innate-os")) / "skills" / relative,
        Path(__file__).resolve().parent / relative,
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return str(candidates[0])


class ArmDrawImageAsWaypoints(Skill):
    """Trace a raster image on paper as Cartesian arm waypoints."""

    manipulation = Interface(InterfaceType.MANIPULATION)

    def __init__(self, logger):
        super().__init__(logger)
        self._cancelled = False

    @property
    def name(self):
        return "arm_draw_image_as_waypoints"

    def guidelines(self):
        return (
            "Draw an image using the arm. Converts a raster image to SVG paths "
            "and traces them as Cartesian waypoints on a flat surface beneath the arm. "
            "Automatically dispatched when arm_mode is set to 'draw'."
        )

    def execute(self):
        image_path = _resolve_image_path(_DEFAULT_IMAGE)
        self.logger.info(f"Drawing from image: {image_path}")

        if not os.path.isfile(image_path):
            return f"Image not found: {image_path}", SkillResult.FAILURE

        actions = produce_waypoints(image_path)
        self.logger.info(f"Produced {len(actions)} actions from image")

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return f"Failed to read image: {image_path}", SkillResult.FAILURE

        img_h, img_w = img.shape[:2]

        CAMERA_CAPTURE_X = 0.17
        CAMERA_CAPTURE_Y = 0.0
        CAMERA_CAPTURE_Z = 0.12
        CAMERA_CAPTURE_ROLL = math.pi
        CAMERA_CAPTURE_PITCH = 1.52805058
        CAMERA_CAPTURE_YAW = 3.0171064
        LIFT_HEIGHT = CAMERA_CAPTURE_Z

        HOME_X = 0.23
        HOME_Y = 0.0
        DOWN_Z = 0.08
        DROP_HEIGHT = DOWN_Z

        workspace_dim = 0.21  # metres

        poses: List[ArmPose] = []
        pose = ArmPose(
            CAMERA_CAPTURE_X, CAMERA_CAPTURE_Y, CAMERA_CAPTURE_Z,
            CAMERA_CAPTURE_ROLL, CAMERA_CAPTURE_PITCH, CAMERA_CAPTURE_YAW,
        )
        poses.append(pose)

        for i, action in enumerate(actions):
            if self._cancelled:
                return "Drawing cancelled", SkillResult.CANCELLED

            if action.action_type == ActionType.LIFT:
                pose = ArmPose(
                    pose.x, pose.y, LIFT_HEIGHT,
                    CAMERA_CAPTURE_ROLL, CAMERA_CAPTURE_PITCH, CAMERA_CAPTURE_YAW,
                )
            elif action.action_type == ActionType.DROP:
                pose = ArmPose(pose.x, pose.y, DROP_HEIGHT, 0.0, 0.0, 0.0)
            elif action.action_type == ActionType.WAYPOINT:
                norm = Waypoint(
                    x=action.waypoint.x / img_w,
                    y=action.waypoint.y / img_h,
                )
                pose = ArmPose(
                    x=HOME_X + norm.x * workspace_dim,
                    y=HOME_Y + norm.y * workspace_dim,
                    z=pose.z,
                    roll=0.0, pitch=0.0, yaw=0.0,
                )

            if action.action_type == ActionType.WAYPOINT and i % 10 == 0:
                poses.append(pose)
            elif action.action_type != ActionType.WAYPOINT:
                poses.append(pose)

        self.logger.info(f"Sending {len(poses)} IK poses to arm")
        poses_dicts = [asdict(pose) for pose in poses]

        success = self.manipulation.move_cartesian_trajectory(
            poses=poses_dicts,
            segment_duration=0.5,
        )

        if not success:
            return "Failed to solve IK or send arm command", SkillResult.FAILURE

        return "Successfully drew image", SkillResult.SUCCESS

    def cancel(self):
        self._cancelled = True
        return "Drawing cancelled"