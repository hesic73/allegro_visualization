import os
import numpy as np
import trimesh
import transforms3d
from allegro_visualization.generate_hand_mesh import generate_hand_mesh


def create_allegro_scene(urdf_path, hand_state, objects):
    """
    Create a trimesh scene with an Allegro hand and multiple objects.

    Parameters:
        urdf_path (str): Path to the URDF file for the hand model.
        hand_state (dict): Dictionary containing Allegro hand state.
        objects (list): List of object definitions.

    Returns:
        trimesh.Scene: The combined scene with the hand and objects.
    """
    hand_mesh = generate_hand_mesh(
        urdf_path,
        hand_state["translation"],
        hand_state["rotation"],
        hand_state["joint_angles"],
    )

    scene = trimesh.Scene([hand_mesh])

    for obj in objects:
        object_mesh = obj["mesh"]

        if "color" in obj:
            object_mesh.visual.face_colors = obj["color"]

        if "scale" in obj:
            object_mesh.apply_scale(obj["scale"])

        object_transform = np.eye(4)
        object_transform[:3, 3] = obj["pose"]["translation"]
        object_transform[:3, :3] = transforms3d.euler.euler2mat(
            *obj["pose"]["rotation"]
        )

        object_mesh.apply_transform(object_transform)

        scene.add_geometry(object_mesh)

    return scene
