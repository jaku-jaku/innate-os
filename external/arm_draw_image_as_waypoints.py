#!/usr/bin/env python3
"""
Arm Move To XYZ Skill - Move arm to a Cartesian position using IK.
"""
import math
from typing import Optional, List
import time
import sys
import os
from dataclasses import dataclass, asdict
from enum import Enum
import re
import tempfile

import numpy as np
import vtracer
import cv2
from svgpathtools import svg2paths
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


from typing import List
from dataclasses import dataclass
from brain_client.skill_types import Skill, SkillResult, Interface, InterfaceType

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
    print(f"Extracted {len(paths)} paths from SVG")
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


class ArmDrawImageAsWaypoints(Skill):
    """Move the arm to a Cartesian position using inverse kinematics."""
    
    manipulation = Interface(InterfaceType.MANIPULATION)
    
    def __init__(self, logger):
        super().__init__(logger)
        self._cancelled = False
    
    @property
    def name(self):
        return "arm_draw_image_as_waypoints"
    
    def guidelines(self):
        return (
            "Move the arm end-effector to a target position in Cartesian space (x, y, z in meters). "
            "Coordinates are relative to the robot base_link. Optionally specify roll, pitch, yaw orientation in radians."
        )
    
    def execute(self):
        """
        Draw image as waypoints.
        
        Args:
        """
        image_path = "/home/jetson1/skills/draw_waypoints_utils/demo/patrick.png"
        self.logger.info(f"Producing waypoints from image: {image_path}")
        actions = produce_waypoints(image_path)
        self.logger.info(f"Produced {len(actions)} actions from image")
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None

        ## Produce XYZ waypoints from actions
        poses: List[ArmPose] = []
        ## TODO(rbenefo): Figure out how to get to 0 pose.
        
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
        DOWN_ROLL = 0.0
        DOWN_PITCH = 0.0
        DOWN_YAW = 0.0
        DROP_HEIGHT = DOWN_Z
        
        image_dim = img.shape[0] # Assuming square image, and shape[0] == shape[1]
        workspace_dim = 0.21 # m

        # while True:
        #     pose = self.manipulation.get_current_end_effector_pose()
        #     self.logger.info(f"Current end-effector pose: {pose}")
        #     time.sleep(0.01)

        ## Define initial position.
        pose = ArmPose(CAMERA_CAPTURE_X, CAMERA_CAPTURE_Y, CAMERA_CAPTURE_Z, CAMERA_CAPTURE_ROLL, CAMERA_CAPTURE_PITCH, CAMERA_CAPTURE_YAW)
        poses.append(pose)
        for i, action in enumerate(actions):
            if action.action_type == ActionType.LIFT:
                pose = ArmPose(pose.x, pose.y, LIFT_HEIGHT, \
                    CAMERA_CAPTURE_ROLL, CAMERA_CAPTURE_PITCH, CAMERA_CAPTURE_YAW)
            elif action.action_type == ActionType.DROP:
                pose = ArmPose(pose.x, pose.y, DROP_HEIGHT,
                    0.0, 0.0, 0.0)
            elif action.action_type == ActionType.WAYPOINT:                
                ## Actions are in pixel coordinates.
                normalized_waypoint = Waypoint(
                    x=action.waypoint.x / image_dim,  # Center at (0, 0)
                    y=action.waypoint.y / image_dim
                )
                self.logger.info(f"Normalized waypoint: ({normalized_waypoint.x}, {normalized_waypoint.y})")
                waypoint = Waypoint(
                    x=normalized_waypoint.x * workspace_dim,  # Scale to fit in 20cm x 20cm area
                    y=normalized_waypoint.y * workspace_dim
                )
                pose = ArmPose(
                    x=HOME_X + waypoint.x,
                    y=HOME_Y + waypoint.y,
                    z=pose.z,
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0
                )
            else:
                raise ValueError("Uh oh. Invalid action??")
            
            if action.action_type == ActionType.WAYPOINT and i % 10 == 0:
                ## Only run every 10th waypoint to make the robot faster.
                self.logger.info(f"Generated pose for action {i}/{len(actions)}: ({pose.x}, {pose.y}, {pose.z})")
                poses.append(pose)
            elif action.action_type != ActionType.WAYPOINT:
                poses.append(pose)
            
        poses_dicts = [asdict(pose) for pose in poses]
        
        success = self.manipulation.move_cartesian_trajectory(
            poses=poses_dicts,
            segment_duration=0.5,
        )

        if not success:
            return "Failed to solve IK or send arm command", SkillResult.FAILURE
     
        return "Successfully drew image", SkillResult.SUCCESS
    
    def cancel(self):
        """Cancel the arm movement."""
        self._cancelled = True
        return "Arm motion cancelled"

def visualize_actions(image: cv2.Mat, actions: List[Action], curr_pos: ArmPose, duration: float, out_path: str):
    plt.figure()
    plt.imshow(image, cmap='gray')
    n = len(actions)

    for i, action in enumerate(actions):
        if action.action_type == ActionType.WAYPOINT:
            xs = [action.waypoint.x]
            ys = [action.waypoint.y]
            plt.plot(xs, ys, 'o', color = "green")
    plt.scatter([curr_pos.x], [curr_pos.y], marker='x', color = "red")
    plt.title("Next action in: {:.1f}s".format(duration))
            
    plt.savefig(out_path)
    
def estimate_duration(target_pos: ArmPose, curr_pos: ArmPose) -> float:
    ## Just focus on translation, not rotation for now.
    max_speed = 0.5 # m/s, super rough estimate
    target_position = np.array([target_pos.x, target_pos.y, target_pos.z])
    curr_position = np.array([curr_pos.x, curr_pos.y, curr_pos.z])
    distance = np.linalg.norm(target_position - curr_position)
    return distance / max_speed