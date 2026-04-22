"""
video_buffer.py — Rolling video buffer for temporal VLM analysis.

Stores recent video frames per track for behavior-over-time analysis.
Efficient ring buffer implementation with configurable duration and FPS.

Usage:
    from core.video_buffer import TrackVideoBuffer, get_buffer, save_frame, get_clip
    
    # Initialize for a track
    buffer = get_buffer(track_id, camera_id, max_seconds=30, target_fps=5)
    
    # Save frames during tracking (called from camera_processor)
    save_frame(track_id, frame_bgr, bbox)
    
    # Get video clip for analysis
    clip = get_clip(track_id, last_n_seconds=5)
    # Returns: List[{"timestamp": float, "image": np.ndarray, "bbox": tuple}]
"""

import threading
import time
import logging
from collections import deque, OrderedDict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameEntry:
    """Single frame entry in the video buffer."""
    timestamp: float
    image: np.ndarray  # BGR cropped image
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in original frame coords
    frame_number: int


@dataclass
class VideoBufferConfig:
    """Configuration for video buffering."""
    max_duration_seconds: float = 30.0  # How much history to keep
    target_fps: float = 5.0  # Store at 5fps (not full camera fps)
    max_tracks: int = 100  # Max tracks to buffer simultaneously (LRU eviction)
    jpeg_quality: int = 85  # Compression for stored frames
    
    def max_frames(self) -> int:
        return int(self.max_duration_seconds * self.target_fps)


# Global configuration
_CONFIG = VideoBufferConfig()

# Track buffers: Dict[track_id -> VideoRingBuffer]
_buffers: OrderedDict[str, "VideoRingBuffer"] = OrderedDict()
_buffers_lock = threading.Lock()

# Last frame time per track (for FPS throttling)
_last_frame_time: Dict[str, float] = {}
_last_frame_lock = threading.Lock()


class VideoRingBuffer:
    """Ring buffer for a single track's video history."""
    
    def __init__(self, track_id: str, camera_id: str, config: VideoBufferConfig):
        self.track_id = track_id
        self.camera_id = camera_id
        self.config = config
        self.max_frames = config.max_frames()
        
        # Use deque as ring buffer
        self.frames: deque = deque(maxlen=self.max_frames)
        self.frame_count = 0
        self.created_at = time.time()
        self.last_access = time.time()
        
        self._lock = threading.Lock()
    
    def add_frame(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Add a frame to the buffer. Returns True if added, False if skipped."""
        with self._lock:
            entry = FrameEntry(
                timestamp=time.time(),
                image=frame_bgr.copy(),
                bbox=bbox,
                frame_number=self.frame_count
            )
            self.frames.append(entry)
            self.frame_count += 1
            self.last_access = time.time()
        return True
    
    def get_clip(self, last_n_seconds: float, max_frames: Optional[int] = None) -> List[FrameEntry]:
        """Get frames from the last N seconds."""
        with self._lock:
            if not self.frames:
                return []
            
            now = time.time()
            cutoff = now - last_n_seconds
            
            # Get all frames newer than cutoff
            result = [f for f in self.frames if f.timestamp >= cutoff]
            
            # Optionally limit frame count (for analysis efficiency)
            if max_frames and len(result) > max_frames:
                # Evenly sample frames to get max_frames
                step = len(result) / max_frames
                result = [result[int(i * step)] for i in range(max_frames)]
            
            self.last_access = now
            return result
    
    def get_key_frames(self, n_frames: int = 5) -> List[FrameEntry]:
        """Get N evenly distributed key frames from the entire buffer."""
        with self._lock:
            if not self.frames:
                return []
            
            if len(self.frames) <= n_frames:
                return list(self.frames)
            
            # Evenly sample
            step = len(self.frames) / n_frames
            result = [self.frames[int(i * step)] for i in range(n_frames)]
            self.last_access = time.time()
            return result
    
    def get_duration(self) -> float:
        """Get actual duration of stored video in seconds."""
        with self._lock:
            if len(self.frames) < 2:
                return 0.0
            return self.frames[-1].timestamp - self.frames[0].timestamp
    
    def clear(self):
        """Clear all frames."""
        with self._lock:
            self.frames.clear()
            self.frame_count = 0


def configure(max_duration_seconds: float = 30.0, target_fps: float = 5.0, max_tracks: int = 100):
    """Configure global video buffer settings."""
    global _CONFIG
    _CONFIG = VideoBufferConfig(
        max_duration_seconds=max_duration_seconds,
        target_fps=target_fps,
        max_tracks=max_tracks
    )
    logger.info(f"[VideoBuffer] Configured: {max_duration_seconds}s @ {target_fps}fps, max {max_tracks} tracks")


def get_buffer(track_id: str, camera_id: str) -> VideoRingBuffer:
    """Get or create a video buffer for a track."""
    key = f"{camera_id}:{track_id}"
    
    with _buffers_lock:
        if key in _buffers:
            # Move to end (LRU)
            _buffers.move_to_end(key)
            return _buffers[key]
        
        # Create new buffer
        buffer = VideoRingBuffer(track_id, camera_id, _CONFIG)
        _buffers[key] = buffer
        
        # LRU eviction
        while len(_buffers) > _CONFIG.max_tracks:
            oldest_key, oldest_buffer = _buffers.popitem(last=False)
            logger.debug(f"[VideoBuffer] Evicted buffer for {oldest_key}")
        
        logger.debug(f"[VideoBuffer] Created buffer for {key}")
        return buffer


def should_save_frame(track_id: str, target_fps: Optional[float] = None) -> bool:
    """Check if we should save a frame based on target FPS throttling."""
    fps = target_fps or _CONFIG.target_fps
    min_interval = 1.0 / fps
    
    with _last_frame_lock:
        last_time = _last_frame_time.get(track_id, 0)
        now = time.time()
        
        if now - last_time >= min_interval:
            _last_frame_time[track_id] = now
            return True
        return False


def save_frame(track_id: str, camera_id: str, frame_bgr: np.ndarray, 
               bbox: Tuple[int, int, int, int], force: bool = False) -> bool:
    """
    Save a frame to the track's video buffer.
    
    Args:
        track_id: Track identifier
        camera_id: Camera identifier
        frame_bgr: Full BGR frame
        bbox: (x1, y1, x2, y2) bounding box to crop
        force: If True, save even if FPS throttling would skip
    
    Returns:
        True if frame was saved, False if skipped
    """
    key = f"{camera_id}:{track_id}"
    
    # Check FPS throttling
    if not force and not should_save_frame(key):
        return False
    
    try:
        # Crop the frame to bbox
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        cropped = frame_bgr[y1:y2, x1:x2].copy()
        
        # Get or create buffer
        buffer = get_buffer(track_id, camera_id)
        
        # Add frame
        return buffer.add_frame(cropped, bbox)
        
    except Exception as e:
        logger.error(f"[VideoBuffer] Error saving frame for {key}: {e}")
        return False


def get_clip(track_id: str, camera_id: str, last_n_seconds: float = 5.0,
             max_frames: Optional[int] = None) -> List[FrameEntry]:
    """Get video clip for a track."""
    key = f"{camera_id}:{track_id}"
    
    with _buffers_lock:
        buffer = _buffers.get(key)
        if not buffer:
            return []
        return buffer.get_clip(last_n_seconds, max_frames)


def get_key_frames(track_id: str, camera_id: str, n_frames: int = 5) -> List[FrameEntry]:
    """Get key frames for a track."""
    key = f"{camera_id}:{track_id}"
    
    with _buffers_lock:
        buffer = _buffers.get(key)
        if not buffer:
            return []
        return buffer.get_key_frames(n_frames)


def clear_track(track_id: str, camera_id: str):
    """Clear buffer for a specific track."""
    key = f"{camera_id}:{track_id}"
    
    with _buffers_lock:
        if key in _buffers:
            _buffers[key].clear()
            del _buffers[key]
            logger.debug(f"[VideoBuffer] Cleared buffer for {key}")


def clear_all():
    """Clear all buffers."""
    with _buffers_lock:
        for buffer in _buffers.values():
            buffer.clear()
        _buffers.clear()
        logger.info("[VideoBuffer] Cleared all buffers")


def get_stats() -> Dict:
    """Get buffer statistics for monitoring."""
    with _buffers_lock:
        total_tracks = len(_buffers)
        total_frames = sum(len(b.frames) for b in _buffers.values())
        avg_duration = sum(b.get_duration() for b in _buffers.values()) / total_tracks if total_tracks > 0 else 0
        
        return {
            "active_tracks": total_tracks,
            "total_frames": total_frames,
            "avg_duration_seconds": round(avg_duration, 2),
            "max_tracks_config": _CONFIG.max_tracks,
            "target_fps": _CONFIG.target_fps,
            "max_duration_config": _CONFIG.max_duration_seconds
        }
