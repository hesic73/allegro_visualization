import torch
import transforms3d
import trimesh
from typing import Dict, List, Optional

from allegro_visualization.utils.hand_model_lite import HandModelURDFLite


def generate_hand_mesh(
    urdf_path: str,
    xyz: List,
    rpy: List,
    qpos: List,
    device="cpu",
    contact_candidates: Optional[Dict[str, List[List[float]]]] = None,
    keypoints: Optional[Dict[str, List[List[float]]]] = None,
) -> trimesh.Trimesh:
    """Generate a mesh for the Allegro hand."""
    hand_model = HandModelURDFLite(
        urdf_path=urdf_path,
        device=device,
        contact_candidates=contact_candidates,
        keypoints=keypoints,
    )

    rotation_matrix = transforms3d.euler.euler2mat(*rpy)
    rotation_vector = rotation_matrix[:, :2].T.ravel().tolist()

    hand_pose = torch.tensor(
        list(xyz) + rotation_vector + list(qpos),
        dtype=torch.float,
        device=device,
    ).unsqueeze(0)

    hand_model.set_parameters(hand_pose)
    return hand_model.get_trimesh_data(0)

