import os
import json
import numpy as np
import cv2
from system import config
from system.logger_setup import setup_logger

logger = setup_logger(__name__)

HOMOGRAPHY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "homographies.json")

def load_all_homographies():
    """Load all saved homography matrices and points from JSON."""
    if not os.path.exists(HOMOGRAPHY_FILE):
        return {}
    try:
        with open(HOMOGRAPHY_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load homographies.json: {e}")
        return {}

def save_all_homographies(data):
    """Save all homography data to JSON."""
    os.makedirs(os.path.dirname(HOMOGRAPHY_FILE), exist_ok=True)
    try:
        with open(HOMOGRAPHY_FILE, "w") as f:
            json.add_space = False
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Failed to save homographies.json: {e}")
        return False

def compute_homography(camera_pts, world_pts):
    """
    Computes a 3x3 homography matrix mapped from camera coordinates to virtual world coordinates.
    camera_pts and world_pts must each be a list of N [x, y] coordinates (N >= 4).
    Returns: (matrix H as list of lists, True/False success)
    """
    if len(camera_pts) < 4 or len(world_pts) != len(camera_pts):
        return None, False
        
    src_pts = np.array(camera_pts, dtype=np.float32)
    dst_pts = np.array(world_pts, dtype=np.float32)
    
    if len(camera_pts) == 4:
        # For exactly 4 points, getPerspectiveTransform provides a direct exact mapping
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    else:
        # For >4 points, findHomography using RANSAC computes the optimal least-squares fit
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
    if H is not None:
        return H.tolist(), True
    return None, False

def load_camera_homography(camera_id):
    """Returns the precomputed H matrix for a camera, or None."""
    data = load_all_homographies()
    cam_data = data.get(camera_id)
    if cam_data and "H" in cam_data:
        return np.array(cam_data["H"], dtype=np.float32)
    return None

def save_camera_homography(camera_id, camera_pts, world_pts):
    """
    Compute H and save the calibration for a specific camera.
    Returns (H_list, True/False)
    """
    H, ok = compute_homography(camera_pts, world_pts)
    if ok:
        data = load_all_homographies()
        data[camera_id] = {
            "camera_pts": camera_pts,
            "world_pts": world_pts,
            "H": H
        }
        success = save_all_homographies(data)
        return H, success
    return None, False
