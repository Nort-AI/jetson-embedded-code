# polygon_zone.py
import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any, Optional

from system.logger_setup import setup_logger

logger = setup_logger(__name__)
UNKNOWN_ZONE = "Unknown"

def load_camera_config(filename: str, client_id: str, store_id: str, camera_id: str) -> Dict[str, Any]:
    """Loads the entire configuration object for a given camera (type, zones, lines)."""
    default_config = {"type": "standard_camera", "zones": []}
    
    if not os.path.exists(filename):
        logger.error(f"Zone definition file not found: {filename}")
        return default_config
    try:
        with open(filename, 'r') as file:
            all_data = json.load(file)
        
        # Navigate through the JSON structure to get the config for the specific camera
        camera_config = all_data.get(client_id, {}).get(store_id, {}).get(camera_id)
        
        if camera_config is None:
            logger.warning(f"No configuration defined for camera_id '{camera_id}' in {filename}.")
            return default_config
            
        return camera_config

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error reading or parsing zone file {filename} for camera {camera_id}: {e}")
        return default_config

def save_camera_zones(filename: str, client_id: str, store_id: str, camera_id: str, zones: List[Dict[str, Any]], camera_type: str = "standard_camera") -> bool:
    """Saves the zone configuration for a specific camera back to the JSON file."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                all_data = json.load(file)
        else:
            all_data = {}
            
        if client_id not in all_data:
            all_data[client_id] = {}
        if store_id not in all_data[client_id]:
            all_data[client_id][store_id] = {}
        if camera_id not in all_data[client_id][store_id]:
            all_data[client_id][store_id][camera_id] = {}
            
        all_data[client_id][store_id][camera_id]["type"] = camera_type
        all_data[client_id][store_id][camera_id]["zones"] = zones
        
        with open(filename, 'w') as file:
            json.dump(all_data, file, indent=4)
        logger.info(f"Successfully saved {len(zones)} zones for camera {camera_id} to {filename}.")
        return True
    except Exception as e:
        logger.error(f"Error saving zones to {filename} for camera {camera_id}: {e}")
        return False

def find_zone(point: Tuple[int, int], zones_for_camera: Optional[List[Dict[str, Any]]]) -> str:
    """Finds the zone a point belongs to from a pre-loaded list of zones."""
    if not zones_for_camera:
        return UNKNOWN_ZONE

    for sector in zones_for_camera:
        vertices = sector.get("polygon_vertices")
        if vertices and len(vertices) >= 3:
            polygon_np = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
            if cv2.pointPolygonTest(polygon_np, point, False) >= 0:
                return sector.get("sector_name", UNKNOWN_ZONE)
    
    return UNKNOWN_ZONE