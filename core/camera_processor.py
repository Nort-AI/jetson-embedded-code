# camera_processor.py - Enhanced with age detection + Cross-Camera Re-ID

import cv2
import os
import time
import numpy as np
import csv
from datetime import datetime
import threading
from collections import deque
from typing import Tuple, Optional, Literal

from system.logger_setup import setup_logger
from system.config import *
from system import config  # Also import module for getattr() usage
from data.data_handler import DataHandler
from core.polygon_zone import find_zone, load_camera_config
from core import utils_body
from core import reid_manager
from data import spatial_logger

occupancy_log_lock = threading.Lock()
OCCUPANCY_LOG_FILE = "occupancy_log.csv"

# ── Live pose skeleton overlay ─────────────────────────────────────────────────
# Used when --live-pose is active.  Draws ONLY on the admin-panel selected person.
# Uses MediaPipe's own draw_landmarks() on the person's ROI in the current frame
# so the rendering is IDENTICAL to the crop panel.  The 33 normalised landmarks
# stored by the background worker are back-projected to the CURRENT bbox so the
# skeleton always tracks the person without drift from stale stored bounds.

def _draw_pose_overlay(frame: np.ndarray, pose_data: dict,
                       cx1: int, cy1: int, cx2: int, cy2: int) -> None:
    """Draw MediaPipe skeleton in-place on frame[cy1:cy2, cx1:cx2].

    The ROI view means modifications go directly into *frame* without a copy.
    Landmarks are normalised [0,1] relative to the person's crop; drawing on
    the ROI that covers the same area maps them correctly onto the full frame.
    """
    if not pose_data or not pose_data.get("detected"):
        return
    landmarks = pose_data.get("landmarks", [])
    if not landmarks:
        return   # YOLO fallback data has no 33-landmark list — skip

    cw, ch = cx2 - cx1, cy2 - cy1
    if cw <= 0 or ch <= 0:
        return

    try:
        import mediapipe as mp
        from mediapipe.framework.formats import landmark_pb2

        # Rebuild NormalizedLandmarkList from the stored list-of-dicts
        lm_list = landmark_pb2.NormalizedLandmarkList()
        for lm_data in landmarks:
            lm       = lm_list.landmark.add()
            lm.x          = float(max(0.0, min(1.0, lm_data["x"])))
            lm.y          = float(max(0.0, min(1.0, lm_data["y"])))
            lm.z          = float(lm_data.get("z", 0.0))
            lm.visibility = float(lm_data.get("vis", 1.0))

        # Draw on the ROI — this is an in-place numpy view, no copy needed
        roi = frame[cy1:cy2, cx1:cx2]
        mp.solutions.drawing_utils.draw_landmarks(
            roi,
            lm_list,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    except Exception:
        pass   # MediaPipe not installed or import error

# Global occupancy tracker shared across all cameras
occupancy_tracker_lock = threading.Lock()
store_occupancy = {}  # Key: (client_id, store_id), Value: current occupancy

def generate_color(track_id):
    if track_id is None: return (150, 150, 150)
    try:
        seed = int(track_id)
    except (ValueError, TypeError):
        seed = abs(hash(track_id))
    np.random.seed(seed % (2**31))
    return tuple(np.random.randint(50, 255, 3).tolist())

class EntranceLineDetector:
    """
    Robust entrance/exit detection using line crossing mathematics
    """
    
    def __init__(self, line_start: Tuple[int, int], line_end: Tuple[int, int], inside_direction: Literal['left', 'right'] = 'left'):
        """
        Initialize entrance line detector
        
        Args:
            line_start: (x1, y1) - Start point of entrance line
            line_end: (x2, y2) - End point of entrance line  
            inside_direction: 'left' or 'right' - Which side of the line is inside the store
                             when looking from line_start to line_end
        """
        self.line_start = np.array(line_start)
        self.line_end = np.array(line_end)
        self.inside_direction = inside_direction
        
        # Calculate line vector and normal vector
        self.line_vector = self.line_end - self.line_start
        self.line_length = np.linalg.norm(self.line_vector)
        
        # Normalize line vector
        if self.line_length > 0:
            self.line_unit_vector = self.line_vector / self.line_length
        else:
            raise ValueError("Line start and end points cannot be the same")
        
        # Calculate normal vector (perpendicular to line)
        # Rotate 90 degrees counterclockwise: (x, y) -> (-y, x)
        self.normal_vector = np.array([-self.line_unit_vector[1], self.line_unit_vector[0]])
        
        # Adjust normal vector based on inside direction
        if inside_direction == 'right':
            self.normal_vector = -self.normal_vector
            
    def get_side_of_line(self, point: Tuple[int, int]) -> Literal['inside', 'outside']:
        """
        Determine which side of the line a point is on
        """
        point_array = np.array(point)
        to_point = point_array - self.line_start
        dot_product = np.dot(to_point, self.normal_vector)
        return 'inside' if dot_product >= 0 else 'outside'
    
    def detect_crossing(self, prev_position: Tuple[int, int], curr_position: Tuple[int, int]) -> Optional[Literal['entry', 'exit']]:
        """
        Detect if a person crossed the entrance line between two positions
        """
        if not self._line_segments_intersect(prev_position, curr_position):
            return None
        
        prev_side = self.get_side_of_line(prev_position)
        curr_side = self.get_side_of_line(curr_position)
        
        if prev_side == 'outside' and curr_side == 'inside':
            return 'entry'
        elif prev_side == 'inside' and curr_side == 'outside':
            return 'exit'
        
        return None
    
    def _line_segments_intersect(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """
        Check if line segment p1->p2 intersects the entrance line
        """
        p1_array = np.array(p1)
        p2_array = np.array(p2)
        
        d1 = p2_array - p1_array
        d2 = self.line_end - self.line_start
        
        cross_product = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross_product) < 1e-6:  # Lines are parallel
            return False
        
        diff = self.line_start - p1_array
        t1 = (diff[0] * d2[1] - diff[1] * d2[0]) / cross_product
        t2 = (diff[0] * d1[1] - diff[1] * d1[0]) / cross_product
        
        return 0 <= t1 <= 1 and 0 <= t2 <= 1

    def get_distance_to_line(self, point: Tuple[int, int]) -> float:
        """Get perpendicular distance from point to entrance line"""
        point_array = np.array(point)
        to_point = point_array - self.line_start
        
        projection_length = np.dot(to_point, self.line_unit_vector)
        projection_length = np.clip(projection_length, 0, self.line_length)
        
        closest_point = self.line_start + projection_length * self.line_unit_vector
        distance = np.linalg.norm(point_array - closest_point)
        
        return distance

class CameraProcessor:
    def __init__(self, camera_id, video_source, data_handler, output_frames, lock, stop_event, barrier, yolo_model, reid_manager: Optional['ReIDManager'] = None, setup_mode: bool = False, shared_attr_session=None):
        self.camera_id = camera_id
        self.video_source = video_source
        self.data_handler = data_handler
        self.output_frames = output_frames
        self.lock = lock
        self.stop_event = stop_event
        self.barrier = barrier
        self.yolo = yolo_model
        self.setup_mode = setup_mode

        # Cross-camera Re-ID manager (shared across all processors)
        self.reid_manager = reid_manager
        
        # Remote management flags
        self.request_snapshot = False
        self.auto_vlm_enabled = False # Gated for resource optimization
        self.logger = setup_logger(f"CamProc-{camera_id}")
        
        self.fps = 0
        self.frame_count = 0
        self.frame_counter = 0   # global frame counter for bbox_renderer animation
        self.fps_start_time = time.time()

        self.yolo = yolo_model

        # ── Privacy Mode Setup ──────────────────────────────────────────────
        self._last_config_check = 0.0
        self._privacy_mode = "disabled"
        try:
            self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            self.logger.warning(f"Failed to load Haar cascade for face blurring: {e}")
            self._face_cascade = None

        # ── VLM (Moondream) integration ─────────────────────────────────────
        from core import vlm_session as _vlm_session_mod
        from core import vlm_analyst as _vlm_analyst_mod
        from core import bbox_renderer as _bbox_renderer_mod
        self._vlm_session    = _vlm_session_mod.get_session(camera_id)
        self._vlm_analyst    = _vlm_analyst_mod
        self._bbox_renderer  = _bbox_renderer_mod
        self.logger.info(f"[VLM] Session created for camera '{camera_id}' "
                         f"(enabled={_vlm_analyst_mod.is_enabled()})")

        # ── DVR ring buffer ─────────────────────────────────────────────────
        from core import frame_buffer as _fb
        self._dvr_buffer = _fb.get_buffer(camera_id)
        
        # Initialize an independent ByteTrack instance for this specific camera stream!
        import supervision as sv
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.40,
            lost_track_buffer=300,  # ~30 s at 10 fps. Keeps the same local track_id for longer
                                    # so the person reappears under their old ID without triggering a new
                                    # register_or_match call — the primary source of ID proliferation.
                                    # 300 frames covers most store occlusion gaps (shelf browsing, etc.)
            minimum_consecutive_frames=3
        )

        self.track_attributes = {}
        # Per-camera recent-activity map: global_id → last_seen unix timestamp.
        # Updated every frame; pruned to 120 s.  Passed as `recent_gids` to
        # register_or_match so the context bonus fires for everyone who was on
        # this camera recently — not just currently-active tracks.
        self._recently_active_gids: dict = {}
        # Deduplication: remember which global_ids were already counted as entered.
        # Prevents the same physical person from being counted multiple times when
        # their local track is lost and re-assigned (new track_id, same global_id).
        # On confirmed exit the gid is removed so they can be counted on re-entry.
        self._counted_entry_global_ids: set = set()
        self.last_homography_check = 0.0       # perf: reload once per frame, not per track
        self._attr_onnx_inp_name = None        # perf: cached ONNX I/O names (set after session loads)
        self._attr_onnx_out_name = None
        self._attr_transforms = None           # perf: cached torchvision transforms pipeline

        # Initialize attribute model (ONNX)
        self.attribute_model = None
        self._attr_onnx_session = None

        if shared_attr_session is not None:
            # ── Shared session: skip per-camera GPU load, reuse the single instance ──
            self._attr_onnx_session = shared_attr_session
            self._attr_onnx_inp_name = shared_attr_session.get_inputs()[0].name
            self._attr_onnx_out_name = shared_attr_session.get_outputs()[0].name
            try:
                from torchvision import transforms as _T
                self._attr_transforms = _T.Compose([
                    _T.Resize((288, 144)),
                    _T.ToTensor(),
                    _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            except ImportError:
                self._attr_transforms = None
            self.logger.info(
                f"[Attr] Using shared ONNX session ({self._attr_onnx_session.get_providers()})"
            )
        else:
            # Per-camera load (fallback when no shared session is passed)
            attr_onnx_path = ATTRIBUTE_MODEL_PATH.replace('.pth', '.onnx').replace('net_last', 'attribute_model')
            if not os.path.exists(attr_onnx_path):
                attr_onnx_path = os.path.join('assets', 'models', 'attribute_model.onnx')

            if os.path.exists(attr_onnx_path):
                try:
                    import platform as _platform
                    import onnxruntime as ort
                    so = ort.SessionOptions()
                    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    # On aarch64 (Jetson) never use TRT/CUDA EP via ORT — GPU is
                    # handled by TRTSession (.engine). ORT TRT EP conflicts with
                    # system libnvinfer and segfaults on Jetson's integrated GPU.
                    if _platform.machine() == 'aarch64':
                        valid = ['CPUExecutionProvider']
                    else:
                        model_dir = os.path.dirname(os.path.abspath(attr_onnx_path))
                        trt_cache = os.path.join(model_dir, 'trt_engine_cache')
                        os.makedirs(trt_cache, exist_ok=True)
                        providers = [
                            ('TensorrtExecutionProvider', {
                                'trt_max_workspace_size': str(4 * 1024 * 1024 * 1024),
                                'trt_fp16_enable': 'True',
                                'trt_engine_cache_enable': 'True',
                                'trt_engine_cache_path': trt_cache,
                            }),
                            'CUDAExecutionProvider',
                            'CPUExecutionProvider',
                        ]
                        available = ort.get_available_providers()
                        valid = [p for p in providers if (p if isinstance(p, str) else p[0]) in available]
                        if not valid:
                            valid = ['CPUExecutionProvider']
                    self._attr_onnx_session = ort.InferenceSession(attr_onnx_path, sess_options=so, providers=valid)
                    self.logger.info(f"Attribute ONNX model loaded: {self._attr_onnx_session.get_providers()}")
                    # Cache I/O names and transforms once — hot path uses these instead of rebuilding every frame
                    self._attr_onnx_inp_name = self._attr_onnx_session.get_inputs()[0].name
                    self._attr_onnx_out_name = self._attr_onnx_session.get_outputs()[0].name
                    try:
                        from torchvision import transforms as _T
                        self._attr_transforms = _T.Compose([
                            _T.Resize((288, 144)),
                            _T.ToTensor(),
                            _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                    except ImportError:
                        self._attr_transforms = None
                except Exception as e:
                    self.logger.error(f"Failed to load attribute ONNX model: {e}")

            elif ATTRIBUTE_MODEL_PATH and os.path.exists(ATTRIBUTE_MODEL_PATH):
                try:
                    self.attribute_model = utils_body.load_network(utils_body.Backbone_nFC(30), ATTRIBUTE_MODEL_PATH)
                    self.attribute_model.to(DEVICE)
                    self.attribute_model.eval()
                    self.logger.info("Attribute classification model loaded (PyTorch fallback).")
                except Exception as e:
                    self.logger.error(f"Failed to load attribute model: {e}")


        if self._attr_onnx_session is None and self.attribute_model is None:
            self.logger.warning(
                "Attribute model failed to load — age/gender inference DISABLED. "
                "Check model path: %s", config.ATTRIBUTE_MODEL_PATH
            )

        # Load camera configuration and homography matrix
        camera_config = load_camera_config(POLYGON_POINTS_FILE, CLIENT_ID, STORE_ID, self.camera_id)
        from core.homography_manager import load_camera_homography
        self.homography_matrix = load_camera_homography(self.camera_id)
        
        self.zones = camera_config.get("zones", [])
        self.is_entrance_camera = camera_config.get("type") == "entrance_camera"
        self.entrance_line = camera_config.get("entrance_line")

        # ENHANCED: Track first zone interactions for entrance analytics
        self.first_zone_tracker = {}  # track_id -> first zone after entrance
        
        # Initialize entrance line detector
        self.entrance_detector = None
        if self.is_entrance_camera and self.entrance_line:
            try:
                inside_direction = camera_config.get("inside_direction", "left")
                
                self.entrance_detector = EntranceLineDetector(
                    line_start=tuple(self.entrance_line[0]),
                    line_end=tuple(self.entrance_line[1]),
                    inside_direction=inside_direction
                )
                self.logger.debug(f"ENTRANCE camera active. Line: {self.entrance_line}, Inside direction: {inside_direction}")
                self.logger.debug(f"Configured zones for first interaction analysis: {[z['sector_name'] for z in self.zones]}")
                
                # Initialize occupancy log
                with occupancy_log_lock:
                    with open(OCCUPANCY_LOG_FILE, 'a', newline='') as f:
                        if f.tell() == 0:
                            csv.writer(f).writerow(['timestamp', 'event_type', 'track_id', 'camera_id'])
                            
            except Exception as e:
                self.logger.error(f"Failed to initialize entrance detector: {e}")
                self.is_entrance_camera = False
        elif self.is_entrance_camera:
            self.logger.error("Camera is type 'entrance_camera' but 'entrance_line' is not defined in JSON!")
            self.is_entrance_camera = False

        # C1-fix: GCS snapshot uploads must NOT block the camera thread.
        # A single-worker executor keeps uploads sequential (no burst) while
        # the camera thread returns immediately after submitting the task.
        from concurrent.futures import ThreadPoolExecutor
        self._snapshot_executor = ThreadPoolExecutor(max_workers=1,
                                                     thread_name_prefix="SnapUpload")

    def update_store_occupancy(self, change):
        """Update global store occupancy counter"""
        global store_occupancy
        store_key = (CLIENT_ID, STORE_ID)
        
        with occupancy_tracker_lock:
            if store_key not in store_occupancy:
                store_occupancy[store_key] = 0
            
            store_occupancy[store_key] += change
            store_occupancy[store_key] = max(0, store_occupancy[store_key])
            current_occupancy = store_occupancy[store_key]
            
        self.logger.debug(f"Store occupancy changed by {change}. Current: {current_occupancy}")
        return current_occupancy

    def get_current_occupancy(self):
        """Get current store occupancy"""
        store_key = (CLIENT_ID, STORE_ID)
        with occupancy_tracker_lock:
            return store_occupancy.get(store_key, 0)

    def log_crossing_event(self, event_type: str, track_id: int = None):
        """Log entrance/exit events"""
        with occupancy_log_lock:
            with open(OCCUPANCY_LOG_FILE, 'a', newline='') as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(), 
                    event_type, 
                    track_id,
                    self.camera_id
                ])

    # ── GStreamer / VideoCapture factory ─────────────────────────────────────
    @staticmethod
    def _open_capture(video_source: str, use_gstreamer: bool = True):
        """
        Open a VideoCapture, preferring hardware-accelerated GStreamer on Jetson.

        For RTSP streams on Jetson: nvv4l2decoder offloads H.264/H.265 decode
        to the dedicated NVDEC engine — frees ~1 CPU core per camera and drops
        decode latency by 40–80 ms vs. software FFmpeg decode.

        Falls back to the standard OpenCV backend silently if GStreamer is not
        available (e.g. dev machines) or if the pipeline fails to open.
        """
        import platform
        _src = str(video_source)
        _is_rtsp = _src.startswith("rtsp://") or _src.startswith("rtsps://")
        _is_jetson = (platform.machine() in ("aarch64",) and platform.system() == "Linux")

        if use_gstreamer and _is_rtsp and _is_jetson:
            # Hardware-accelerated H.264 decode via Jetson NVDEC.
            # nvv4l2decoder → nvvidconv → appsink
            gst_pipeline = (
                f"rtspsrc location={_src} latency=100 ! "
                "rtph264depay ! h264parse ! "
                "nvv4l2decoder enable-max-performance=1 ! "
                "nvvidconv ! "
                "video/x-raw,format=BGRx ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink drop=1 max-buffers=1 sync=false"
            )
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap
            # GStreamer pipeline failed — fall through to default backend

        return cv2.VideoCapture(video_source)

    def run(self):
        cap = self._open_capture(self.video_source)
        if not cap.isOpened():
            cap.release()  # C2-fix: always release to avoid FD leak on failure
            self.logger.error(f"Failed to open video source: {self.video_source}")
            self.barrier.abort()
            return
        
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            cap.release()
            return

        self.logger.info("Processing started.")
        self.fps_start_time = time.time()

        # Detect if source is a live stream (RTSP/HTTP) vs. a local file
        _src = str(self.video_source)
        _is_live_stream = _src.startswith("rtsp://") or _src.startswith("http") or _src.isdigit()

        # Reconnection state
        _consecutive_failures = 0
        _MAX_RECONNECT_WAIT = 60  # cap backoff at 60 seconds

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # For file sources, just stop at EOF
                if not _is_live_stream:
                    self.logger.warning("End of video file.")
                    break

                # ── Live stream dropped — reconnect with exponential backoff ──
                _consecutive_failures += 1
                _backoff = min(2 ** _consecutive_failures, _MAX_RECONNECT_WAIT)
                self.logger.warning(
                    f"Stream dropped (attempt {_consecutive_failures}). "
                    f"Reconnecting in {_backoff}s..."
                )

                cap.release()

                # Wait with periodic checks so we can respond to stop_event
                _wait_end = time.time() + _backoff
                while time.time() < _wait_end:
                    if self.stop_event.is_set():
                        self.logger.info("Stop requested during reconnect wait.")
                        return
                    time.sleep(0.5)

                # Attempt to reopen the stream (re-use hardware pipeline)
                cap = self._open_capture(self.video_source)
                if cap.isOpened():
                    self.logger.info(
                        f"Reconnected to {self.video_source} "
                        f"after {_consecutive_failures} attempt(s)."
                    )
                    _consecutive_failures = 0

                    # Reset YOLO tracker state so ByteTrack doesn't carry
                    # stale track IDs from before the disconnect
                    try:
                        if hasattr(self.yolo, 'predictor') and self.yolo.predictor is not None:
                            self.yolo.predictor.trackers = []
                    except Exception:
                        pass  # safe to ignore — tracker resets on its own
                else:
                    cap.release()  # C2-reconnect-fix: release failed cap to avoid FD leak
                    self.logger.error(f"Reconnect failed. Will retry in {min(2 ** (_consecutive_failures + 1), _MAX_RECONNECT_WAIT)}s...")
                continue

            # Successful read — reset failure counter
            _consecutive_failures = 0

            # ── Create a clean copy of the raw frame for Re-ID / YOLO input ──
            # All visualizations (zones, bounding boxes, text) are drawn on
            # `draw_frame` so they never pollute the pixels used for embedding
            # extraction, attribute classification, or thumbnail capture.
            draw_frame = frame.copy()


            # Draw entrance line (on draw_frame only)
            if self.entrance_detector:
                p1, p2 = self.entrance_line
                cv2.line(draw_frame, tuple(p1), tuple(p2), (0, 0, 255), 3)
                
                mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                normal_scaled = self.entrance_detector.normal_vector * 50
                arrow_end = (int(mid_point[0] + normal_scaled[0]), int(mid_point[1] + normal_scaled[1]))
                cv2.arrowedLine(draw_frame, mid_point, arrow_end, (0, 255, 255), 2, tipLength=0.3)
                
                cv2.putText(draw_frame, f"ENTRANCE ({self.entrance_detector.inside_direction})", 
                           (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
                if self.is_entrance_camera:
                    occupancy_text = f"Occupancy: {self.get_current_occupancy()}"
                    cv2.putText(draw_frame, occupancy_text, (10, draw_frame.shape[0] - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ── SETUP MODE: Lite execution ──
            if self.setup_mode:
                # Draw setup mode watermark
                text = "🛠 SETUP MODE (AI OFF)"
                cv2.putText(draw_frame, text, (draw_frame.shape[1] // 2 - 150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3, cv2.LINE_AA)
                
                # Update global output frames for run.py display & MJPEG stream
                with self.lock:
                    self.output_frames[self.camera_id] = draw_frame
                    # Important: Provide raw frame for the zone editor in admin panel
                    self.latest_raw_frame = frame
                
                # Calculate dummy FPS
                self.frame_count += 1
                if self.frame_count >= 30:
                    now = time.time()
                    self.fps = self.frame_count / (now - self.fps_start_time)
                    self.frame_count = 0
                    self.fps_start_time = now
                
                # Keep CPU low
                time.sleep(0.01)
                continue

            import supervision as sv

            # ── Inference throttle: run YOLOX every N frames ──────────────────
            # On skipped frames ByteTrack continues via Kalman prediction —
            # track positions stay valid and all downstream logic still runs.
            _detect_every = getattr(config, 'DETECT_EVERY_N_FRAMES', 1)
            _run_detection = (self.frame_count % max(1, _detect_every) == 0)

            if _run_detection:
                # YOLOX ONNX detection on the CLEAN frame (no drawings)
                detections_array = self.yolo.detect(frame)

                # M5-fix: guard against unexpected model output shape before slicing
                _arr_ok = (
                    len(detections_array) > 0
                    and hasattr(detections_array, 'ndim')
                    and detections_array.ndim == 2
                    and detections_array.shape[1] >= 6
                )
                if _arr_ok:
                    xyxy = detections_array[:, :4]
                    confidence = detections_array[:, 4]
                    class_id = detections_array[:, 5].astype(int)
                    detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
                else:
                    detections_array = []
                    detections = sv.Detections.empty()
            else:
                # Skipped frame: pass empty detections so ByteTrack predicts forward
                detections_array = []
                detections = sv.Detections.empty()

            # Update independent supervision tracker
            tracked_detections = self.tracker.update_with_detections(detections)
            
            # --- DEBUG INFO EXPOSURE ---
            self.frame_detection_count = len(detections_array) if len(detections_array) > 0 else 0
            self.active_track_count = len(tracked_detections) if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None else 0
            
            frame_data_to_log = []
            current_occupancy = self.get_current_occupancy()
            
            if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None:
                boxes = tracked_detections.xyxy.astype(int)
                track_ids = tracked_detections.tracker_id.astype(int)

                # Perf: reload homography once per frame (was inside the per-track loop — N×slower)
                if time.time() - self.last_homography_check > 5.0:
                    from core.homography_manager import load_camera_homography
                    self.homography_matrix = load_camera_homography(self.camera_id)
                    self.last_homography_check = time.time()

                for i, track_id in enumerate(track_ids):
                    bbox = tuple(boxes[i])
                    x1, y1, x2, y2 = bbox
                    
                    # ── Check for significant overlap with other tracks ────────────────
                    # We no longer strictly gate on 1-pixel overlaps, as people walking together
                    # would never get their embeddings updated. OSNet-AIN's attention mechanism handles minor occlusions.
                    skip_thumb = False        
                    is_new_track = track_id not in self.track_attributes
                    if is_new_track:
                        current_position = (int((x1 + x2) / 2), int(y2))
                        now = time.time()
                        linked_global_id = None
                        linked_has_entered = False  # inherited from old track on recovery

                        # Spatio-Temporal Recovery: If an old track just died right here, assume it's the same person
                        for old_tid, old_attrs in self.track_attributes.items():
                            if old_tid == track_id: continue
                            time_since_seen = now - old_attrs.get("last_seen", now)

                            # Dropped between 0.2 s and 45 s ago.
                            # 45 s gives ByteTrack (30 s buffer) time to create a
                            # new local_id and spatio-temporal time to catch it.
                            # 500 px covers a person walking across a typical store
                            # aisle during the gap (was 250 px — too tight).
                            if 0.2 < time_since_seen < 45.0:
                                old_pos = old_attrs.get("last_position")
                                if old_pos and old_attrs.get("global_id") is not None:
                                    dist = np.linalg.norm(np.array(current_position) - np.array(old_pos))
                                    if dist < 500:
                                        linked_global_id = old_attrs["global_id"]
                                        # Carry over entry state so the new track doesn't re-count
                                        linked_has_entered = old_attrs.get("has_entered", False)
                                        self.logger.debug(f"Spatio-Temporal Link: Recovered track {old_tid} -> {track_id} (GID: {linked_global_id}, has_entered={linked_has_entered})")
                                        if self.reid_manager:
                                            self.reid_manager.link_local_to_global(self.camera_id, int(track_id), linked_global_id)
                                        break

                        self.track_attributes[track_id] = {
                            "last_position": None,
                            "crossing_status": "none",
                            "gender": None,
                            "age_category": None,
                            "detection_count": 0,
                            "last_seen": now,
                            # Inherit entry state from spatio-temporal recovery so a track
                            # that briefly disappears and reappears doesn't count twice.
                            "has_entered": linked_has_entered,
                            "first_zone_after_entry": None,
                            "entrance_timestamp": None,
                            "smoothed_position": None,
                            "global_id": linked_global_id,
                            "last_embedding_update": 0,
                            "last_crossing_ts": 0.0,   # cooldown: prevents oscillation double-counts
                        }
                    
                    attrs = self.track_attributes[track_id]
                    attrs["detection_count"] += 1
                    attrs["last_seen"] = time.time()

                    # ── Attribute Classification (before Re-ID so attrs feed into matching)
                    # Run once per new track, then refresh every 60 frames
                    _run_attrs = (attrs.get("gender") is None or
                                  attrs["detection_count"] % 120 == 0)  # perf: every 4s at 30fps
                    if _run_attrs and (self._attr_onnx_session or self.attribute_model):
                        try:
                            from core.utils_body import predict_decoder, extract_age_from_predictions, categorize_age

                            _hf, _wf = frame.shape[:2]
                            _cx1, _cy1 = max(0, x1), max(0, y1)
                            _cx2, _cy2 = min(_wf, x2), min(_hf, y2)

                            if _cx2 > _cx1 + 20 and _cy2 > _cy1 + 40:
                                if self._attr_onnx_session and self._attr_onnx_inp_name and self._attr_transforms:
                                    # Single ONNX session run — both output labels and raw_probs from one inference
                                    _crop_bgr = frame[_cy1:_cy2, _cx1:_cx2]
                                    _crop_rgb = cv2.cvtColor(_crop_bgr, cv2.COLOR_BGR2RGB)
                                    from PIL import Image as _Image
                                    _crop_pil = _Image.fromarray(_crop_rgb)
                                    _tensor_np = self._attr_transforms(_crop_pil).unsqueeze(0).numpy()
                                    _raw_np = self._attr_onnx_session.run(
                                        [self._attr_onnx_out_name], {self._attr_onnx_inp_name: _tensor_np}
                                    )[0]
                                    import torch as _torch
                                    _raw_out = _torch.from_numpy(_raw_np)
                                    _dec = predict_decoder('market')
                                    output = _dec.decode(_raw_out)
                                    raw_probs = _dec.decode_raw(_raw_out)
                                    attrs['raw_attr_probs'] = raw_probs
                                    attrs['attr_label_list'] = _dec.label_list
                                else:
                                    # PyTorch fallback
                                    import torch as _torch
                                    from PIL import Image as _Image
                                    # H3-fix: use the cached transforms pipeline instead of
                                    # rebuilding T.Compose (+ import overhead) on every frame.
                                    if self._attr_transforms is None:
                                        from torchvision import transforms as _T
                                        self._attr_transforms = _T.Compose([
                                            _T.Resize((288, 144)),
                                            _T.ToTensor(),
                                            _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
                                    _transforms = self._attr_transforms
                                    _crop_bgr = frame[_cy1:_cy2, _cx1:_cx2]
                                    _crop_rgb = cv2.cvtColor(_crop_bgr, cv2.COLOR_BGR2RGB)
                                    _crop_pil = _Image.fromarray(_crop_rgb)
                                    _tensor = _transforms(_crop_pil).unsqueeze(0).to(DEVICE)
                                    with _torch.no_grad():
                                        _raw_out = self.attribute_model.forward(_tensor)
                                    _dec = predict_decoder('market')
                                    output = _dec.decode(_raw_out)
                                    raw_probs = _dec.decode_raw(_raw_out)
                                    attrs['raw_attr_probs'] = raw_probs
                                    attrs['attr_label_list'] = _dec.label_list

                                # Process age
                                _age_val = extract_age_from_predictions(output)
                                output['age'] = _age_val
                                output['age_category'] = categorize_age(_age_val)
                            else:
                                if self._attr_onnx_session is not None:
                                    from core.utils_body import classify_body_onnx
                                    output = classify_body_onnx(frame, x1, y1, x2, y2, self._attr_onnx_session)
                                else:
                                    # Fallback if no model is loaded
                                    output = {"gender": "Unknown", "age": "adult", "age_category": "adult"}
                            attrs['gender']       = output.get("gender", "Unknown")
                            attrs['age_category'] = output.get("age_category", "adult")
                            attrs['attributes']   = output
                        except Exception as e:
                            import time as _t
                            _now = _t.monotonic()
                            _last = getattr(self, '_attr_err_last', 0)
                            _cnt  = getattr(self, '_attr_err_cnt',  0) + 1
                            self._attr_err_cnt  = _cnt
                            self._attr_err_last = _now if _now - _last >= 60 else _last
                            if _now - _last >= 60:          # log at most once per minute
                                self.logger.error(
                                    "Attr classify error: %s  (suppressing for 60 s; "
                                    "%d failures since last log)", e, _cnt)
                                self._attr_err_cnt = 0
                            attrs.setdefault('gender', 'Unknown')
                            attrs.setdefault('age_category', 'adult')
                            attrs.setdefault('attributes', {})
                        finally:
                            # M1-fix: empty_cache() on the per-frame hot path
                            # forces a CUDA synchronisation point every frame and
                            # limits attribute throughput to <5 FPS on Jetson.
                            # PyTorch/TRT reclaims memory automatically when tensors
                            # go out of scope; no explicit cache flush is needed here.
                            pass

                    gender       = attrs.get("gender") or "Unknown"
                    age_category = attrs.get("age_category") or "adult"
                    full_attrs   = attrs.get("attributes")        # decoded labels dict
                    raw_attr_probs   = attrs.get("raw_attr_probs")   # float32 ndarray
                    attr_label_list  = attrs.get("attr_label_list")  # list[str]

                    # Get frame dimensions for use in multiple places
                    h, w = frame.shape[:2]
                    attrs["frame_w"] = w
                    attrs["frame_h"] = h
                    # Store current bbox so run.py _make_retail_data can expose it to admin VLM UI
                    attrs["x1"] = x1
                    attrs["y1"] = y1
                    attrs["x2"] = x2
                    attrs["y2"] = y2
                    
                    # ── Cross-camera Re-ID (attributes are now available for fusion)
                    if self.reid_manager and getattr(config, 'REID_ENABLED', True):
                        margin_pct = getattr(config, 'REID_EDGE_MARGIN_PCT', 0.05)
                        margin_x = int(w * margin_pct)
                        margin_y = int(h * margin_pct)
                        is_on_edge = (x1 < margin_x or y1 < margin_y or x2 > w - margin_x or y2 > h - margin_y)

                        # Always compute sharpness — passed to manager as anchor quality signal
                        sharpness = 0.0
                        crop_for_sharp = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if crop_for_sharp.size > 0:
                            gray = cv2.cvtColor(crop_for_sharp, cv2.COLOR_BGR2GRAY)
                            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

                        base_sharpness = getattr(config, 'MIN_LAPLACIAN_SHARPNESS', 10.0)
                        # A frame is usable if it's not on an edge and not completely blurred.
                        # The manager handles its own confirmation window — no hard minting gate here.
                        is_acceptable = not is_on_edge and not skip_thumb and sharpness >= (base_sharpness * 0.5)

                        if (is_new_track or attrs["global_id"] is None):
                            if is_acceptable:
                                # All global_ids seen on this camera in the last 90 s get the
                                # context bonus inside register_or_match (+0.08 to emb_sim).
                                # Using _recently_active_gids rather than track_attributes means
                                # people whose track just dropped are still included.
                                _now_ra = time.time()
                                recent_gids = {
                                    gid for gid, ts in self._recently_active_gids.items()
                                    if _now_ra - ts <= 90.0
                                }

                                attrs["global_id"] = self.reid_manager.register_or_match(
                                    frame=frame,
                                    bbox=(x1, y1, x2, y2),
                                    camera_id=self.camera_id,
                                    local_track_id=int(track_id),
                                    attributes=full_attrs,
                                    raw_attr_probs=raw_attr_probs,
                                    attr_label_list=attr_label_list,
                                    skip_thumbnail=False,
                                    recent_gids=recent_gids,
                                    current_occupancy=current_occupancy,
                                    crop_sharpness=sharpness,
                                )
                            else:
                                if attrs["detection_count"] % 15 == 0:
                                    self.logger.debug(
                                        f"Track {track_id} skipped Re-ID: "
                                        f"on_edge={is_on_edge} sharpness={sharpness:.1f}"
                                    )
                        else:
                            # ── Periodic embedding update for existing tracks ──────────────────
                            detection_count = attrs["detection_count"]
                            last_update = attrs.get("last_embedding_update", 0)

                            # Fast early-track refresh: every 5 frames for first 10 detections.
                            # This quickly builds a rich, multi-angle anchor bank.
                            # After that, fall back to the normal update interval.
                            if detection_count <= 10:
                                update_interval = 5
                            else:
                                update_interval = getattr(config, 'REID_UPDATE_INTERVAL_FRAMES', 30)

                            if (detection_count - last_update) >= update_interval:
                                if is_acceptable:
                                    success = self.reid_manager.update_embedding(
                                        camera_id=self.camera_id,
                                        local_track_id=int(track_id),
                                        frame=frame,
                                        bbox=(x1, y1, x2, y2),
                                        attributes=full_attrs,
                                        raw_attr_probs=raw_attr_probs,
                                        attr_label_list=attr_label_list,
                                        skip_thumbnail=False,
                                        crop_sharpness=sharpness,
                                    )
                                    if success:
                                        attrs["last_embedding_update"] = detection_count
                            elif _run_attrs and full_attrs:
                                # Lightweight attribute refresh (no GPU needed)
                                self.reid_manager.update_attributes(
                                    camera_id=self.camera_id,
                                    local_track_id=int(track_id),
                                    attributes=full_attrs,
                                )
                    if self.reid_manager is None:
                        # Graceful fallback: prefix with camera_id to prevent database collisions
                        # IMPORTANT: Respect Spatio-Temporal Recovery ID if it exists!
                        global_id = attrs.get("global_id") or f"{self.camera_id}_{track_id}"
                        attrs["global_id"] = global_id  # Save it so we don't recompute
                    else:
                        global_id = attrs.get("global_id") or int(track_id)

                    # If ReID just resolved a global_id that was already counted as entered
                    # (person was re-detected after track loss), inherit the entry state so
                    # they aren't re-counted when they next cross the entrance line.
                    if global_id and global_id in self._counted_entry_global_ids and not attrs.get("has_entered"):
                        attrs["has_entered"] = True

                    center_x = int((x1 + x2) / 2)
                    bottom_y = int(y2)
                    current_position = (center_x, bottom_y)

                    # ENHANCED ENTRANCE/EXIT DETECTION with zone tracking
                    if self.entrance_detector and attrs["last_position"] is not None:
                        crossing_result = self.entrance_detector.detect_crossing(
                            prev_position=attrs["last_position"],
                            curr_position=current_position
                        )

                        if crossing_result:
                            _now_ts  = time.time()
                            _gid     = attrs.get("global_id")
                            # Cooldown: ignore crossings within 8 s of the last one for
                            # this track (prevents oscillation at the entrance line).
                            _cooldown = (_now_ts - attrs.get("last_crossing_ts", 0.0)) < 8.0

                            if crossing_result == 'entry':
                                # Deduplicate: skip if this global_id was already counted
                                _already = (_gid is not None and _gid in self._counted_entry_global_ids)
                                if not _already and not _cooldown:
                                    attrs["crossing_status"]  = "entered"
                                    attrs["has_entered"]       = True
                                    attrs["entrance_timestamp"] = datetime.now()
                                    attrs["last_crossing_ts"]  = _now_ts
                                    self.log_crossing_event("entry", track_id)
                                    current_occupancy = self.update_store_occupancy(1)
                                    if _gid is not None:
                                        self._counted_entry_global_ids.add(_gid)
                                    self.logger.debug(
                                        f"Track {track_id} (GID:{_gid}) ENTERED. Occupancy: {current_occupancy}"
                                    )
                                else:
                                    # Duplicate or still on cooldown — keep state consistent but don't count
                                    attrs["crossing_status"] = "entered"
                                    attrs["has_entered"]     = True
                                    self.logger.debug(
                                        f"[occupancy] Skipped duplicate entry track={track_id} GID={_gid} "
                                        f"({'dup gid' if _already else 'cooldown'})"
                                    )

                            elif crossing_result == 'exit':
                                if not _cooldown:
                                    attrs["crossing_status"]   = "exited"
                                    attrs["last_crossing_ts"]  = _now_ts
                                    self.log_crossing_event("exit", track_id)
                                    current_occupancy = self.update_store_occupancy(-1)
                                    # Allow this person to be counted again on future re-entry
                                    if _gid is not None:
                                        self._counted_entry_global_ids.discard(_gid)
                                    self.logger.debug(
                                        f"Track {track_id} (GID:{_gid}) EXITED. Occupancy: {current_occupancy}"
                                    )
                                    # Log first zone interaction if we captured it
                                    if attrs.get("first_zone_after_entry"):
                                        self.logger.debug(
                                            f"Track {track_id} first interaction was with zone: {attrs['first_zone_after_entry']}"
                                        )
                                else:
                                    attrs["crossing_status"] = "exited"
                    
                    # Update position for next frame
                    attrs["last_position"] = current_position

                    # EMA position smoothing — reduces zone flickering near boundaries
                    # alpha=0.35: responsive enough to track movement, smooth enough to avoid jitter
                    _alpha = getattr(config, 'TRACK_EMA_ALPHA', 0.35)
                    _raw = np.array(current_position, dtype=float)
                    if attrs["smoothed_position"] is None:
                        attrs["smoothed_position"] = _raw
                    else:
                        attrs["smoothed_position"] = _alpha * _raw + (1.0 - _alpha) * attrs["smoothed_position"]
                    smoothed_position = (int(attrs["smoothed_position"][0]), int(attrs["smoothed_position"][1]))

                    # ── Spatial position logging (every 5th detection for this track) ──
                    if attrs["detection_count"] % 5 == 0:
                        h_frame, w_frame = frame.shape[:2]
                        _cx = smoothed_position[0] / w_frame if w_frame > 0 else 0.0
                        _cy = smoothed_position[1] / h_frame if h_frame > 0 else 0.0
                        _gen = (attrs.get("gender") or "unknown").lower()
                        _age = (attrs.get("age_category") or "unknown").lower()
                        spatial_logger.log_position(
                            camera_id=self.camera_id,
                            track_id=int(track_id),
                            cx=_cx,
                            cy=_cy,
                            gender=_gen,
                            age_group=_age,
                        )

                    # Determine zone based on EMA-smoothed position (prevents flickering at zone edges)
                    zone = find_zone(smoothed_position, self.zones)
                    
                    # ENHANCED: Track first zone interaction
                    if attrs.get("first_zone_after_entry") is None and zone and zone != "Unknown":
                        attrs["first_zone_after_entry"] = zone
                        time_since_entry = 0.0
                        if attrs.get("entrance_timestamp"):
                            time_since_entry = (datetime.now() - attrs["entrance_timestamp"]).total_seconds()
                        
                        self.logger.debug(f"FIRST ZONE INTERACTION: Track {track_id} -> {zone} (after {time_since_entry:.1f}s)")
                        
                        # Store in global tracker for analytics
                        self.first_zone_tracker[track_id] = {
                            'zone': zone,
                            'timestamp': datetime.now(),
                            'time_to_first_interaction': time_since_entry,
                            'gender': attrs.get('gender', 'Unknown'),
                            'age_category': attrs.get('age_category', 'adult')  # Use age_category from attributes
                        }
                        
                        if not attrs.get("has_entered"):
                            attrs["has_entered"] = True

                    # Draw entrance debug (on draw_frame)
                    if DRAW_ENTRANCE_DEBUG and self.entrance_detector:
                        self._draw_entrance_debug(draw_frame, current_position, track_id)

                    # Compute display-ready values
                    age_display = age_category if age_category else "adult"
                    crossing_indicator = ""
                    if attrs["crossing_status"] == "entered":
                        crossing_indicator = " ->"
                    elif attrs["crossing_status"] == "exited":
                        crossing_indicator = " <-"

                    # ── VLM: save crop + request/get analysis ─────────────────
                    # Compute crop quality independently of ReID so this works
                    # even when reid_manager is None / ReID is disabled.
                    vlm_result = None
                    if self._vlm_analyst.is_enabled():
                        # Lightweight quality gate: not on edge, crop big enough
                        margin_pct = getattr(config, 'REID_EDGE_MARGIN_PCT', 0.05)
                        _mx = int(w * margin_pct)
                        _my = int(h * margin_pct)
                        _on_edge = (x1 < _mx or y1 < _my or
                                    x2 > w - _mx or y2 > h - _my)
                        _crop_area = max(0, x2 - x1) * max(0, y2 - y1)
                        _vlm_ok = not _on_edge and _crop_area >= getattr(
                            config, 'VLM_MIN_CROP_AREA', 3000)
                        if _vlm_ok:
                            # Add 25% padding around the bounding box so the crop
                            # shows context rather than a tight slice of the person.
                            _pad_x = max(10, int((x2 - x1) * 0.25))
                            _pad_y = max(10, int((y2 - y1) * 0.15))
                            _cx1 = max(0, x1 - _pad_x)
                            _cy1 = max(0, y1 - _pad_y)
                            _cx2 = min(w, x2 + _pad_x)
                            _cy2 = min(h, y2 + _pad_y)
                            crop_bgr = frame[_cy1:_cy2, _cx1:_cx2]
                            if crop_bgr.size > 0:
                                # Use the SAME key formula as _make_retail_data in run.py:
                                #   str(global_id) when a ReID global ID has been assigned,
                                #   "{camera_id}_{track_id}" otherwise.
                                # This ensures /api/crop/{track_id} finds the crop because
                                # the SSE stream sends the same key as the crop store key.
                                _crop_key = (
                                    str(attrs.get("global_id"))
                                    if attrs.get("global_id") is not None
                                    else f"{self.camera_id}_{track_id}"
                                )
                                self._vlm_analyst.save_crop(
                                    _crop_key, crop_bgr, self.camera_id,
                                    crop_bounds=(_cx1, _cy1, _cx2, _cy2),
                                )
                        # Fire analysis job only if explicitly enabled (saves GPU/API resources)
                        if self.auto_vlm_enabled:
                            self._vlm_session.request(str(global_id), mode="describe")
                        
                        vlm_result = self._vlm_session.get(str(global_id))

                    # ── HUD bounding box (bbox_renderer — includes VLM chip) ──
                    reid_enabled = self.reid_manager and getattr(config, 'REID_ENABLED', True)
                    self._bbox_renderer.draw_hud_box(
                        draw_frame,
                        bbox=(x1, y1, x2, y2),
                        global_id=global_id,
                        local_track_id=int(track_id),
                        camera_id=self.camera_id,
                        gender=gender or "?",
                        age_category=age_display,
                        zone=zone or "",
                        crossing_indicator=crossing_indicator,
                        vlm_result=vlm_result,
                        frame_counter=self.frame_counter,
                        reid_enabled=bool(reid_enabled),
                    )

                    # ── Live pose skeleton on video feed (--live-pose) ────────
                    # Only drawn for the person currently selected in the admin
                    # panel.  Keypoints are back-projected from the CURRENT
                    # detection bbox so the skeleton never drifts from the person.
                    # ── Live pose skeleton on video feed (--live-pose) ────────
                    # Only drawn for the person currently selected in the admin panel.
                    # Uses MediaPipe rendering on the ROI so it looks identical to
                    # the crop panel.  No age check — the person leaving frame removes
                    # their track from detections, so stale data is never reached.
                    if getattr(config, "LIVE_POSE_ENABLED", False) and self._vlm_analyst.is_enabled():
                        _active = self._vlm_analyst.get_active_target()
                        if _active:
                            _pkey = (
                                str(attrs.get("global_id"))
                                if attrs.get("global_id") is not None
                                else f"{self.camera_id}_{track_id}"
                            )
                            if _pkey == str(_active):
                                try:
                                    with self._vlm_analyst._track_crops_lock:
                                        _pe = self._vlm_analyst._track_crops.get(_pkey)
                                        _pd = _pe.get("pose_data") if _pe else None
                                    if _pd:
                                        # Padded bounds from CURRENT bbox — skeleton tracks person
                                        _ppx = max(10, int((x2 - x1) * 0.25))
                                        _ppy = max(10, int((y2 - y1) * 0.15))
                                        _draw_pose_overlay(
                                            draw_frame, _pd,
                                            max(0, x1 - _ppx), max(0, y1 - _ppy),
                                            min(w, x2 + _ppx), min(h, y2 + _ppy),
                                        )
                                except Exception:
                                    pass

                    # Highlight active VLM search targets instantly in the stream
                    if self._vlm_analyst.is_enabled():
                        if str(global_id) == self._vlm_analyst.get_active_target():
                            self._bbox_renderer.draw_selection_hud(
                                draw_frame,
                                bbox=(x1, y1, x2, y2),
                                frame_counter=self.frame_counter
                            )

                    # zone already determined above (line ~378) — do not re-call find_zone()

                    # Floor Plan Transformation (Homography)
                    world_x, world_y = None, None
                    if hasattr(self, 'homography_matrix') and self.homography_matrix is not None:
                        pts = np.array([[[float(current_position[0]), float(current_position[1])]]], dtype=np.float32)
                        transformed = cv2.perspectiveTransform(pts, self.homography_matrix)
                        world_x = round(float(transformed[0][0][0]), 2)
                        world_y = round(float(transformed[0][0][1]), 2)
                    
                    attrs["world_x"] = world_x
                    attrs["world_y"] = world_y

                    # ENHANCED: Log entry with age data and Re-ID global_id
                    log_entry = {
                        "client_id": CLIENT_ID,
                        "store_id": STORE_ID,
                        "camera_id": self.camera_id,
                        "timestamp": datetime.now().isoformat(),
                        "track_id": int(track_id),
                        "global_id": global_id,   # cross-camera identity
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "world_x": world_x,
                        "world_y": world_y,
                        "zone": zone,
                        "gender": gender or "Unknown",
                        "age_category": age_category or "adult",
                        "store_occupancy": current_occupancy,
                        "has_entered": attrs.get("has_entered", False),
                        "first_zone_after_entry": attrs.get("first_zone_after_entry"),
                        "crossing_status": attrs.get("crossing_status", "none")
                    }
                    frame_data_to_log.append(log_entry)

                    # Bug fix: reset crossing_status after logging so it appears exactly once
                    # in the CSV rather than on every frame for the lifetime of the track.
                    if attrs["crossing_status"] in ("entered", "exited"):
                        attrs["crossing_status"] = "none"
            
            # CRITICAL: Log occupancy snapshots for entrance cameras even with no detections
            elif self.is_entrance_camera and self.frame_count % 30 == 0:
                log_entry = {
                    "client_id": CLIENT_ID,
                    "store_id": STORE_ID,
                    "camera_id": self.camera_id, 
                    "timestamp": datetime.now().isoformat(), 
                    "track_id": None, 
                    "x1": None, 
                    "y1": None, 
                    "x2": None, 
                    "y2": None, 
                    "world_x": None,
                    "world_y": None,
                    "zone": None,
                    "gender": None,
                    "age_category": None,  # NEW: Include age category
                    "store_occupancy": current_occupancy,
                    "has_entered": False,
                    "first_zone_after_entry": None,
                    "crossing_status": "none"
                }
                frame_data_to_log.append(log_entry)
            
            if frame_data_to_log: 
                self.data_handler.write_data(frame_data_to_log)

            # Snapshot uses the ANNOTATED frame (shows what the user sees)
            if self.request_snapshot:
                self._take_and_upload_snapshot(draw_frame)
                self.request_snapshot = False

            # ── VLM auto-scan tick (once per frame, picks most-stale track) ─
            if self._vlm_analyst.is_enabled() and tracked_detections is not None:
                _active_gids = [
                    str(self.track_attributes[tid].get("global_id", tid))
                    for tid in (tracked_detections.tracker_id.astype(int)
                                if tracked_detections.tracker_id is not None else [])
                    if tid in self.track_attributes
                ]
                self._vlm_session.tick(_active_gids)

            # Update per-camera activity map — records when each global_id was last
            # seen so register_or_match can give them the context bonus even after
            # their local track drops.
            _ra_now = time.time()
            for _ra_attrs in self.track_attributes.values():
                _ra_gid = _ra_attrs.get("global_id")
                if _ra_gid is not None:
                    self._recently_active_gids[_ra_gid] = _ra_now
            # Prune entries older than 120 s to bound memory
            if self.frame_count % 300 == 0:
                self._recently_active_gids = {
                    g: t for g, t in self._recently_active_gids.items()
                    if _ra_now - t < 120.0
                }

            # M6-fix: clean up old tracking data more aggressively to prevent
            # unbounded dict growth on busy retail scenes (many unique people/day)
            if self.frame_count % 30 == 0 or len(self.track_attributes) > 500:
                self._cleanup_old_tracks()

            # ── Privacy Mode / Face Blurring ─────────────────────────────────
            if time.time() - self._last_config_check > 5.0:
                try:
                    import json
                    _dev_json = os.path.join(BASE_DIR, "device.json")
                    with open(_dev_json, "r") as f:
                        cfg = json.load(f)
                        self._privacy_mode = cfg.get("ui_settings", {}).get("privacy_mode", "disabled")
                        # Handle old boolean fallback
                        if self._privacy_mode is True: self._privacy_mode = "heuristic"
                        if self._privacy_mode is False: self._privacy_mode = "disabled"
                except Exception:
                    self._privacy_mode = "disabled"
                self._last_config_check = time.time()

            if self._privacy_mode != "disabled" and tracked_detections is not None and len(tracked_detections) > 0:
                for bbox in tracked_detections.xyxy.astype(int):
                    x1, y1, x2, y2 = bbox
                    # Ensure within bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(draw_frame.shape[1], x2), min(draw_frame.shape[0], y2)
                    
                    if self._privacy_mode == "heuristic":
                        head_h = int((y2 - y1) * 0.25)
                        head_y2 = y1 + head_h
                        if head_y2 > y1 and x2 > x1:
                            roi = draw_frame[y1:head_y2, x1:x2]
                            draw_frame[y1:head_y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)
                            
                    elif self._privacy_mode == "haar" and self._face_cascade is not None:
                        head_h = int((y2 - y1) * 0.4)
                        head_y2 = y1 + head_h
                        if head_y2 > y1 and x2 > x1:
                            roi = draw_frame[y1:head_y2, x1:x2]
                            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            faces = self._face_cascade.detectMultiScale(gray_roi, 1.1, 4)
                            for (fx, fy, fw, fh) in faces:
                                face_roi = roi[fy:fy+fh, fx:fx+fw]
                                roi[fy:fy+fh, fx:fx+fw] = cv2.GaussianBlur(face_roi, (51, 51), 0)

            # Push annotated draw_frame to display output
            with self.lock:
                self.output_frames[self.camera_id] = draw_frame.copy()
                self.latest_raw_frame = frame.copy()

            # DVR: save 1 JPEG/sec to the ring buffer (non-blocking rate-limiting inside)
            self._dvr_buffer.push(draw_frame)

            self.frame_count += 1
            self.frame_counter += 1   # drives bbox_renderer scan-line animation
            elapsed_time = time.time() - self.fps_start_time
            if elapsed_time >= 1.0:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.fps_start_time = time.time()

        cap.release()
        self.logger.info("Processing finished.")

    def _cleanup_old_tracks(self):
        """Remove tracking data for people who haven't been seen recently"""
        current_time = time.time()
        tracks_to_remove = []

        for track_id, attrs in self.track_attributes.items():
            if current_time - attrs.get('last_seen', current_time) > 60:  # was 30 — keep stale entries longer so spatio-temporal can find them
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            # Notify Re-ID manager so embedding is saved to recently-lost buffer
            if self.reid_manager:
                self.reid_manager.note_track_lost(self.camera_id, int(track_id))
            # Clean up VLM session cooldown + renderer scan-line animation state
            global_id = self.track_attributes[track_id].get("global_id")
            if global_id is not None:
                self._vlm_session.clear_track(str(global_id))
                self._bbox_renderer.clear_track_state(global_id)
            del self.track_attributes[track_id]

        if tracks_to_remove:
            self.logger.debug(f"Cleaned up {len(tracks_to_remove)} old tracks")

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────────────────────────────────

    def draw_person_box(
        self,
        frame: np.ndarray,
        bbox: tuple,
        global_id: int,
        local_track_id: int,
        gender: str = "?",
        age_category: str = "adult",
        crossing_indicator: str = "",
        reid_enabled: bool = True,
    ):
        """
        Premium HUD-style person bounding box renderer.

        Visual language
        ───────────────
        • Bounding Box      : ultra-thin crisp lines with reinforced thick corners (HUD style).
        • Floating Banner   : sleek semi-transparent dark banner anchored to top-left.
        • Identity Accent   : vibrant neon vertical stripe alongside the banner mapping to the person's unique global colour.
        • Info strip        : semi-transparent bar at bottom of bbox detailing demographics.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        color = generate_color(global_id)
        box_w = x2 - x1
        corner_len = max(10, min(20, int(box_w * 0.18)))
        thickness = 2

        # ── 1. Bounding Box Corners ──────────────────────────────────────────
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

        # ── 2. Top-Left HUD Banner (Dark background + Neon Accent line) ───────
        prefix = "G:" if reid_enabled else "ID:"
        header_text = f"{prefix} {global_id}"
        
        font_scale = 0.65
        font = cv2.FONT_HERSHEY_DUPLEX
        (text_w, text_h), baseline = cv2.getTextSize(header_text, font, font_scale, 1)

        banner_h = text_h + 12
        banner_w = text_w + 20
        
        banner_y1 = max(0, y1 - banner_h - 6)
        banner_y2 = banner_y1 + banner_h
        banner_x1 = x1
        banner_x2 = x1 + banner_w
        
        if banner_y2 > banner_y1 and banner_x2 > banner_x1 and banner_x2 <= w:
            # Solid dark background for maximum legibility against bright scenes
            roi = frame[banner_y1:banner_y2, banner_x1:banner_x2]
            dark = np.zeros_like(roi)
            frame[banner_y1:banner_y2, banner_x1:banner_x2] = cv2.addWeighted(roi, 0.15, dark, 0.85, 0)
            
            # Thick colorful accent marking the physical identity
            cv2.line(frame, (banner_x1 + 2, banner_y1), (banner_x1 + 2, banner_y2), color, 4)
            
            # Bright, crisp white identity text (thickness=2)
            text_x = banner_x1 + 10
            text_y = banner_y2 - 6
            cv2.putText(frame, header_text, (text_x, text_y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        # ── 3. Bottom Info Strip (Gender | Age) ──────────────────────────────
        strip_h   = 32
        strip_y1  = max(0, y2 - strip_h)
        strip_y2  = min(h, y2)
        strip_x2  = min(w, x2)

        if strip_y2 > strip_y1 and strip_x2 > x1:
            # Solid dark bottom strip for legibility
            roi = frame[strip_y1:strip_y2, x1:strip_x2]
            dark = np.zeros_like(roi)
            frame[strip_y1:strip_y2, x1:strip_x2] = cv2.addWeighted(roi, 0.20, dark, 0.80, 0)

            gender_icon = "M" if gender.lower() in ("male", "m") else ("F" if gender.lower() in ("female", "f") else "?")
            info_text = f"[{gender_icon}] {gender} | {age_category}{crossing_indicator}"
            
            cv2.putText(
                frame, info_text,
                (x1 + 6, strip_y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA,
            )

        # ── 4. Local-track debug note (tiny, subtle grey) ────────────────────
        debug_y = min(h - 5, y2 + 14)
        cv2.putText(
            frame, f"loc_trk:{local_track_id}",
            (x1, debug_y),
            cv2.FONT_HERSHEY_PLAIN, 0.85, (160, 160, 160), 1, cv2.LINE_AA,
        )

    def draw_label(self, frame, bbox, label_text, color):
        """Legacy plain-text label (kept for non-ReID paths)."""
        x1, y1, _, _ = bbox
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w + 10, y1), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _draw_entrance_debug(self, frame, person_position, track_id):
        """Draw debugging information for entrance detection"""
        if not self.entrance_detector:
            return
            
        # Determine side and distance
        side = self.entrance_detector.get_side_of_line(person_position)
        # Fix: use line crossing math directly if no distance helper exists
        # distance = self.entrance_detector.get_distance_to_line(person_position)
        
        # Color code based on side
        color = (0, 255, 0) if side == 'inside' else (255, 0, 0)
        cv2.circle(frame, person_position, 8, color, -1)

    def _take_and_upload_snapshot(self, frame):
        """Saves a JPEG to disk immediately, then submits the GCS upload to a
        background executor so the camera thread is never stalled by network I/O.
        C1-fix: was blocking the camera thread with no timeout.
        """
        try:
            timestamp = int(time.time())
            local_filename = f"snapshot_{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(local_filename, frame)  # fast — local disk only
            self._snapshot_executor.submit(
                self._upload_snapshot_worker, local_filename
            )
        except Exception as e:
            self.logger.error(f"Failed to save snapshot for upload: {e}")

    def _upload_snapshot_worker(self, local_filename: str):
        """Background worker: upload a saved snapshot to GCS, then delete it.
        Runs in the SnapUpload thread pool — never on the camera thread.
        """
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket_name = os.environ.get("GCS_BUCKET", "nort-support-files")
            bucket = client.bucket(bucket_name)
            destination = (
                f"snapshots/client_id={CLIENT_ID}/store_id={STORE_ID}"
                f"/{self.camera_id}/{os.path.basename(local_filename)}"
            )
            blob = bucket.blob(destination)
            self.logger.debug(f"Uploading snapshot: {local_filename} -> gs://{bucket_name}/{destination}")
            blob.upload_from_filename(local_filename, timeout=30)
            self.logger.debug("Snapshot uploaded successfully.")
        except Exception as e:
            self.logger.error(f"Snapshot GCS upload failed: {e}")
        finally:
            try:
                if os.path.exists(local_filename):
                    os.remove(local_filename)
            except OSError:
                pass