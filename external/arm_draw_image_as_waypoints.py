#!/usr/bin/env python3
"""
Arm Move To XYZ Skill - Move arm to a Cartesian position using IK.
"""
import math
from typing import Optional, List
import time
import sys
import os
from dataclasses import dataclass
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
        tracker_image_path = "/home/jetson1/skills/draw_waypoints_utils/demo/tracker.png"
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
        
        patrick_dim = 225
        workspace_dim = 0.21 # m

        # while True:
        #     pose = self.manipulation.get_current_end_effector_pose()
        #     self.logger.info(f"Current end-effector pose: {pose}")
        #     time.sleep(0.01)

        ## Define initial position.
        pose = ArmPose(CAMERA_CAPTURE_X, CAMERA_CAPTURE_Y, CAMERA_CAPTURE_Z, CAMERA_CAPTURE_ROLL, CAMERA_CAPTURE_PITCH, CAMERA_CAPTURE_YAW)
        poses.append(pose)
        for action in actions:
            if action.action_type == ActionType.LIFT:
                pose = ArmPose(pose.x, pose.y, LIFT_HEIGHT, \
                    CAMERA_CAPTURE_ROLL, CAMERA_CAPTURE_PITCH, CAMERA_CAPTURE_YAW)
            elif action.action_type == ActionType.DROP:
                pose = ArmPose(pose.x, pose.y, DROP_HEIGHT,
                    0.0, 0.0, 0.0)
            elif action.action_type == ActionType.WAYPOINT:                
                ## Actions are in pixel coordinates.
                ## Patrick is 225 x 225
                normalized_waypoint = Waypoint(
                    x=action.waypoint.x / patrick_dim,  # Center at (0, 0)
                    y=action.waypoint.y / patrick_dim
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
            poses.append(pose)
            
        
        for i, pose in enumerate(poses):
            # if i != 0 and i % 5 != 0:
            #     continue ## Skip some waypoints to speed up execution.
            log_info = f"Attempting to move arm to XYZ ({pose.x}, {pose.y}, {pose.z})."
            
            curr_pos = self.manipulation.get_current_end_effector_pose()
            if curr_pos is not None:
                curr_postion = curr_pos['position']
                curr_arm_pos = ArmPose(curr_postion['x'], curr_postion['y'], curr_postion['z'], 0.0, 0.0, 0.0)
                log_info += " Current arm position is XYZ ({:.3f}, {:.3f}, {:.3f}).".format(curr_arm_pos.x, curr_arm_pos.y, curr_arm_pos.z)
                duration=estimate_duration(pose, curr_arm_pos)
            else:
                duration = 5.0
            self.logger.info(log_info)

            duration = max(duration, 0.1)
            if i != 0 and curr_pos is not None:
                ## Hack: Skip the first index, since it's the initial position.
                scaled_arm_pos = ArmPose((curr_arm_pos.x- HOME_X) / workspace_dim * patrick_dim, 
                                         (curr_arm_pos.y- HOME_Y) / workspace_dim * patrick_dim, 
                                         0.0, 0.0, 0.0, 0.0) # Skip z, not needed for image viz.
                visualize_actions(img, actions[:i - 1], scaled_arm_pos, duration, tracker_image_path)

            result = self._execute_pose(
                x=pose.x, y=pose.y, z=pose.z,
                roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw,
                duration=duration,
                timeout=duration * 2.0,
            )
            if result[1] != SkillResult.SUCCESS:
                self.logger.error(
                    f"Failed to move to waypoint ({pose.x}, {pose.y}, {pose.z}) with result: {result[0]}"
                )
     
        return "Successfully drew image", SkillResult.SUCCESS
    
    def cancel(self):
        """Cancel the arm movement."""
        self._cancelled = True
        return "Arm motion cancelled"


    def _execute_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        pos_tolerance: float = 0.2,
        angle_tolerance: float = 0.1, # rad, approx ~5°
        duration: float= 5.0,
        timeout: float = 10.0,
    ):
        """
        Move arm to Cartesian pose using IK.
        
        Args:
            x: Target x position in meters (forward from base)
            y: Target y position in meters (left from base)
            z: Target z position in meters (up from base)
            roll: Target roll orientation in radians
            pitch: Target pitch orientation in radians
            yaw: Target yaw orientation in radians
            duration: Motion duration in seconds
        """
        self._cancelled = False
        
        if self.manipulation is None:
            print("Manipulation interface not available!")
            return "Manipulation interface not available", SkillResult.FAILURE
        
        self.logger.info(
            f"Moving arm to XYZ ({x}, {y}, {z}) with RPY ({roll}, {pitch}, {yaw}) and duration: {duration}"
        )
        
        success = self.manipulation.move_to_cartesian_pose(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            duration=duration
        )

        if not success:
            return "Failed to solve IK or send arm command", SkillResult.FAILURE
        
        # Wait for motion to complete (with cancellation check)
        start_time = time.time()
        while True:
            curr_pos = self.manipulation.get_current_end_effector_pose()
            position = curr_pos['position']
            curr_position = np.array([position['x'], position['y'], position['z']])
            target_pos = np.array([x, y, z])

            rel_pos = target_pos - curr_position
            curr_orientation = curr_pos['orientation']
            curr_quat = R.from_quat([
                curr_orientation['x'],
                curr_orientation['y'],
                curr_orientation['z'],
                curr_orientation['w']
            ])
            target_quat = R.from_euler('xyz', [roll, pitch, yaw])
            rel_angle = compute_rel_angle(target_quat, curr_quat)
            
            time_elapsed = time.time() - start_time
            if time_elapsed > timeout:
                error = f"Timed out after {timeout} seconds while moving. Final pos: ({curr_position[0]}, {curr_position[1]}, {curr_position[2]})"
                self.logger.error(error)
                return error, SkillResult.FAILURE
            
            if self._cancelled:
                return "Arm motion cancelled", SkillResult.CANCELLED

            time.sleep(0.1)
            if (rel_pos < pos_tolerance).all() and rel_angle < angle_tolerance:
                return f"Arm moved to ({x}, {y}, {z})", SkillResult.SUCCESS   
        error = f"Failed to reach target within tolerances. Final pos: ({curr_pos[0]}, {curr_pos[1]}, {curr_pos[2]}), target pos: ({x}, {y}, {z}), rel pos: ({rel_pos[0]}, {rel_pos[1]}, {rel_pos[2]}), rel angle: {rel_angle}"
        self.logger.error(error)
        return error, SkillResult.FAILURE


def compute_rel_angle(angla_a: R, angle_b: R) -> float:
    r_rel = angla_a * angle_b.inv()
    # angle of rotation (in radians)
    angle = r_rel.magnitude()
    return angle


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