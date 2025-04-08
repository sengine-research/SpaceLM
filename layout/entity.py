"""
This code is derived from the SceneScript language sequence and entity parameters.

Reference: https://github.com/facebookresearch/scenescript/blob/main/src/data/language_sequence.py
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

NORMALIZATION_PRESET = {
    "world": (0, 32.0),
    "height": (0.0, 25.6),
    "width": (0.0, 25.6),
    "scale": (0.0, 20.0),
    "angle": (-6.2832, 6.2832),
    "num_bins": 640,
}


@dataclass
class Wall:
    id: int
    ax: float
    ay: float
    az: float
    bx: float
    by: float
    bz: float
    height: float
    thickness: float
    entity_label: str = "wall"

    def __post_init__(self):
        self.id = int(self.id)
        self.ax = float(self.ax)
        self.ay = float(self.ay)
        self.az = float(self.az)
        self.bx = float(self.bx)
        self.by = float(self.by)
        self.bz = float(self.bz)
        self.height = float(self.height)
        self.thickness = float(self.thickness)

    def rotate(self, angle: float):
        wall_start = np.array([self.ax, self.ay, self.az])
        wall_end = np.array([self.bx, self.by, self.bz])
        rotmat = R.from_rotvec([0, 0, angle]).as_matrix()
        wall_start = rotmat @ wall_start
        wall_end = rotmat @ wall_end

        self.ax = wall_start[0]
        self.ay = wall_start[1]
        self.az = wall_start[2]
        self.bx = wall_end[0]
        self.by = wall_end[1]
        self.bz = wall_end[2]

    def translate(self, translation: np.ndarray):
        self.ax += translation[0]
        self.ay += translation[1]
        self.az += translation[2]
        self.bx += translation[0]
        self.by += translation[1]
        self.bz += translation[2]

    def scale(self, scaling: float):
        self.height *= scaling
        self.thickness *= scaling
        self.ax *= scaling
        self.ay *= scaling
        self.az *= scaling
        self.bx *= scaling

    def normalize_and_discretize(self):
        height_min, height_max = NORMALIZATION_PRESET["height"]
        world_min, world_max = NORMALIZATION_PRESET["world"]
        num_bins = NORMALIZATION_PRESET["num_bins"]

        self.height = (self.height - height_min) / (height_max - height_min) * num_bins
        self.thickness = (
            (self.thickness - height_min) / (height_max - height_min) * num_bins
        )
        self.ax = (self.ax - 0) / (world_max - world_min) * num_bins / 2
        self.ay = (self.ay - 0) / (world_max - world_min) * num_bins / 2
        self.az = (self.az - 0) / (world_max - world_min) * num_bins / 2
        self.bx = (self.bx - 0) / (world_max - world_min) * num_bins / 2
        self.by = (self.by - 0) / (world_max - world_min) * num_bins / 2
        self.bz = (self.bz - 0) / (world_max - world_min) * num_bins / 2

        self.height = np.clip(self.height, 0, num_bins - 1)
        self.thickness = np.clip(self.thickness, 0, num_bins - 1)
        self.ax = np.clip(self.ax, -num_bins / 2, num_bins / 2 - 1)
        self.ay = np.clip(self.ay, -num_bins / 2, num_bins / 2 - 1)
        self.az = np.clip(self.az, -num_bins / 2, num_bins / 2 - 1)
        self.bx = np.clip(self.bx, -num_bins / 2, num_bins / 2 - 1)
        self.by = np.clip(self.by, -num_bins / 2, num_bins / 2 - 1)
        self.bz = np.clip(self.bz, -num_bins / 2, num_bins / 2 - 1)

    def undiscretize_and_unnormalize(self):
        num_bins = NORMALIZATION_PRESET["num_bins"]
        height_min, height_max = NORMALIZATION_PRESET["height"]
        world_min, world_max = NORMALIZATION_PRESET["world"]

        # undiscretize
        self.height = self.height / num_bins
        self.thickness = self.thickness / num_bins
        self.ax = self.ax / num_bins * 2
        self.ay = self.ay / num_bins * 2
        self.az = self.az / num_bins * 2
        self.bx = self.bx / num_bins * 2
        self.by = self.by / num_bins * 2
        self.bz = self.bz / num_bins * 2

        # unnormalize
        self.height = self.height * (height_max - height_min) + height_min
        self.thickness = self.thickness * (height_max - height_min) + height_min
        self.ax = self.ax * (world_max - world_min) + 0
        self.ay = self.ay * (world_max - world_min) + 0
        self.az = self.az * (world_max - world_min) + 0
        self.bx = self.bx * (world_max - world_min) + 0
        self.by = self.by * (world_max - world_min) + 0
        self.bz = self.bz * (world_max - world_min) + 0

    def to_language_string(self):
        capitalized_label = self.entity_label.capitalize()
        # wall_0=Wall(a_x,a_y,a_z,b_x,b_y,b_z,height,thickness)
        language_string = f"{self.entity_label}_{self.id}={capitalized_label}({self.ax},{self.ay},{self.az},{self.bx},{self.by},{self.bz},{self.height},{self.thickness})"
        return language_string


@dataclass
class Door:
    id: int
    wall_id: int
    position_x: float
    position_y: float
    position_z: float
    width: float
    height: float
    entity_label: str = "door"

    def __post_init__(self):
        self.id = int(self.id)
        self.wall_id = int(self.wall_id)
        self.position_x = float(self.position_x)
        self.position_y = float(self.position_y)
        self.position_z = float(self.position_z)
        self.width = float(self.width)
        self.height = float(self.height)

    def rotate(self, angle: float):
        center = np.array([self.position_x, self.position_y, self.position_z])
        rotmat = R.from_rotvec([0, 0, angle]).as_matrix()
        new_center = rotmat @ center

        self.position_x = new_center[0]
        self.position_y = new_center[1]
        self.position_z = new_center[2]

    def translate(self, translation: np.ndarray):
        self.position_x += translation[0]
        self.position_y += translation[1]
        self.position_z += translation[2]

    def scale(self, scaling: float):
        self.width *= scaling
        self.height *= scaling
        self.position_x *= scaling
        self.position_y *= scaling
        self.position_z *= scaling

    def normalize_and_discretize(self):
        width_min, width_max = NORMALIZATION_PRESET["width"]
        height_min, height_max = NORMALIZATION_PRESET["height"]
        world_min, world_max = NORMALIZATION_PRESET["world"]
        num_bins = NORMALIZATION_PRESET["num_bins"]

        self.width = (self.width - width_min) / (width_max - width_min) * num_bins
        self.height = (self.height - height_min) / (height_max - height_min) * num_bins
        self.position_x = (
            (self.position_x - 0) / (world_max - world_min) * num_bins / 2
        )
        self.position_y = (
            (self.position_y - 0) / (world_max - world_min) * num_bins / 2
        )
        self.position_z = (
            (self.position_z - 0) / (world_max - world_min) * num_bins / 2
        )

        self.width = np.clip(self.width, 0, num_bins - 1)
        self.height = np.clip(self.height, 0, num_bins - 1)
        self.position_x = np.clip(self.position_x, -num_bins / 2, num_bins / 2 - 1)
        self.position_y = np.clip(self.position_y, -num_bins / 2, num_bins / 2 - 1)
        self.position_z = np.clip(self.position_z, -num_bins / 2, num_bins / 2 - 1)

    def undiscretize_and_unnormalize(self):
        num_bins = NORMALIZATION_PRESET["num_bins"]
        width_min, width_max = NORMALIZATION_PRESET["width"]
        height_min, height_max = NORMALIZATION_PRESET["height"]
        world_min, world_max = NORMALIZATION_PRESET["world"]

        # undiscretize
        self.width = self.width / num_bins
        self.height = self.height / num_bins
        self.position_x = self.position_x / num_bins * 2
        self.position_y = self.position_y / num_bins * 2
        self.position_z = self.position_z / num_bins * 2

        # unnormalize
        self.width = self.width * (width_max - width_min) + width_min
        self.height = self.height * (height_max - height_min) + height_min
        self.position_x = self.position_x * (world_max - world_min) + 0
        self.position_y = self.position_y * (world_max - world_min) + 0
        self.position_z = self.position_z * (world_max - world_min) + 0

    def to_language_string(self):
        capitalized_label = self.entity_label.capitalize()
        self.id = self.id % 1000
        # door_0=Door(wall_id,position_x,position_y,position_z,width,height)
        language_string = f"{self.entity_label}_{self.id}={capitalized_label}(wall_{self.wall_id},{self.position_x},{self.position_y},{self.position_z},{self.width},{self.height})"
        return language_string


@dataclass
class Window(Door):
    entity_label: str = "window"

entity_label_list = ["floor", "wall", "door", "window", "ceiling", "sofa", "bed", "table", "desk", "bookshelf", "cabinet"]

@dataclass
class Bbox:
    id: int
    class_name: str
    position_x: float
    position_y: float
    position_z: float
    angle_z: float
    scale_x: float
    scale_y: float
    scale_z: float
    entity_label: str = "bbox"

    def __post_init__(self):
        self.id = int(self.id)
        self.class_name = str(self.class_name)
        self.position_x = float(self.position_x)
        self.position_y = float(self.position_y)
        self.position_z = float(self.position_z)
        self.angle_z = float(self.angle_z)
        self.scale_x = abs(float(self.scale_x))
        self.scale_y = abs(float(self.scale_y))
        self.scale_z = abs(float(self.scale_z))

    def rotate(self, angle: float):
        augment_rot_mat = R.from_rotvec([0, 0, angle]).as_matrix()
        bbox_rot_mat = R.from_rotvec([0, 0, self.angle_z]).as_matrix()
        new_bbox_rot_mat = augment_rot_mat @ bbox_rot_mat
        new_angle_z = R.from_matrix(new_bbox_rot_mat).as_euler("ZYX")[0]
        new_angle_z = (new_angle_z + np.pi) % (2 * np.pi) - np.pi  # Range: [-pi, pi)

        # Bbox is symmetric
        symmetry = np.pi
        if np.isclose(self.scale_x, self.scale_y, atol=1e-3):
            symmetry = np.pi / 2
        new_angle_z = (new_angle_z + np.pi) % symmetry - np.pi
        self.angle_z = new_angle_z

        bbox_center = np.array([self.position_x, self.position_y, self.position_z])
        bbox_center = augment_rot_mat @ bbox_center
        self.position_x = bbox_center[0]
        self.position_y = bbox_center[1]
        self.position_z = bbox_center[2]

    def translate(self, translation: np.ndarray):
        self.position_x += translation[0]
        self.position_y += translation[1]
        self.position_z += translation[2]

    def scale(self, scaling: float):
        self.scale_x *= scaling
        self.scale_y *= scaling
        self.scale_z *= scaling
        self.position_x *= scaling
        self.position_y *= scaling
        self.position_z *= scaling

    def normalize_and_discretize(self):
        world_min, world_max = NORMALIZATION_PRESET["world"]
        scale_min, scale_max = NORMALIZATION_PRESET["scale"]
        angle_min, angle_max = NORMALIZATION_PRESET["angle"]
        num_bins = NORMALIZATION_PRESET["num_bins"]

        self.position_x = (
            (self.position_x - world_min) / (world_max - world_min) * num_bins
        )
        self.position_y = (
            (self.position_y - world_min) / (world_max - world_min) * num_bins
        )
        self.position_z = (
            (self.position_z - world_min) / (world_max - world_min) * num_bins
        )
        self.angle_z = (self.angle_z - angle_min) / (angle_max - angle_min) * num_bins
        self.scale_x = (self.scale_x - scale_min) / (scale_max - scale_min) * num_bins
        self.scale_y = (self.scale_y - scale_min) / (scale_max - scale_min) * num_bins
        self.scale_z = (self.scale_z - scale_min) / (scale_max - scale_min) * num_bins

        self.position_x = np.int32(np.clip(self.position_x, 0, num_bins - 1))
        self.position_y = np.int32(np.clip(self.position_y, 0, num_bins - 1))
        self.position_z = np.int32(np.clip(self.position_z, 0, num_bins - 1))
        self.angle_z = np.int32(np.clip(self.angle_z, 0, num_bins - 1))
        self.scale_x = np.int32(np.clip(self.scale_x, 0, num_bins - 1))
        self.scale_y = np.int32(np.clip(self.scale_y, 0, num_bins - 1))
        self.scale_z = np.int32(np.clip(self.scale_z, 0, num_bins - 1))

    def undiscretize_and_unnormalize(self):
        world_min, world_max = NORMALIZATION_PRESET["world"]
        scale_min, scale_max = NORMALIZATION_PRESET["scale"]
        angle_min, angle_max = NORMALIZATION_PRESET["angle"]
        num_bins = NORMALIZATION_PRESET["num_bins"]

        self.position_x = self.position_x / num_bins
        self.position_y = self.position_y / num_bins
        self.position_z = self.position_z / num_bins
        self.angle_z = self.angle_z / num_bins
        self.scale_x = self.scale_x / num_bins
        self.scale_y = self.scale_y / num_bins
        self.scale_z = self.scale_z / num_bins

        self.position_x = self.position_x * (world_max - world_min) + world_min
        self.position_y = self.position_y * (world_max - world_min) + world_min
        self.position_z = self.position_z * (world_max - world_min) + world_min
        self.angle_z = self.angle_z * (angle_max - angle_min) + angle_min
        self.scale_x = self.scale_x * (scale_max - scale_min) + scale_min
        self.scale_y = self.scale_y * (scale_max - scale_min) + scale_min
        self.scale_z = self.scale_z * (scale_max - scale_min) + scale_min

    def to_language_string(self):
        capitalized_label = self.entity_label.capitalize()
        self.id = self.id % 1000
        # bbox_0=Bbox(class_name,position_x,position_y,position_z,angle_z,scale_x,scale_y,scale_z)
        language_string = f"{self.entity_label}_{self.id}=({self.class_name},{self.position_x},{self.position_y},{self.position_z},{self.angle_z},{self.scale_x},{self.scale_y},{self.scale_z})"
        return language_string
