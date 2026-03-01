"""
Face Frame Detection Node for ComfyUI

Face detection and eye state analysis for video frame selection.
Uses MediaPipe (supports both legacy Solutions API and new Tasks API) for robust face detection.

Credits:
- MediaPipe face detection by Google (Apache License 2.0)
- Eye Aspect Ratio (EAR) algorithm for blink detection

Copyright (c) 2025 DaxNodes
Licensed under MIT License
"""

import torch
import numpy as np
import json
import os
import urllib.request
from ...utils.debug_utils import debug_print

class FaceFrameDetector:
    """Find optimal reference frame with face and open eyes for video continuation"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "detection_method": ([
                    "mediapipe",
                    "mediapipe_detailed", 
                    "bbox_fallback"
                ], {"default": "mediapipe"}),
                "min_face_size": ("INT", {
                    "default": 50,
                    "min": 20,
                    "max": 200,
                    "step": 10
                }),
                "eye_open_threshold": ("FLOAT", {
                    "default": 0.24,
                    "min": 0.15,
                    "max": 0.4,
                    "step": 0.01
                }),
                "face_confidence": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.3,
                    "max": 1.0,
                    "step": 0.05
                }),
            },
            "optional": {
                "scan_limit": ("INT", {
                    "default": 20,
                    "min": 5,
                    "max": 50,
                    "step": 1
                }),
                "require_frontal": ("BOOLEAN", {"default": True}),
                "prefer_center": ("BOOLEAN", {"default": True}),
                "debug_output": ("BOOLEAN", {"default": True, "tooltip": "Show debug bounding boxes and detection info"}),
                "enable_face_detection": ("BOOLEAN", {"default": True, "tooltip": "When false, skips detection and outputs last frame"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "IMAGE")
    RETURN_NAMES = ("best_face_frame", "frames_from_end", "detection_metadata", "debug_overlay")
    FUNCTION = "detect_best_face_frame"
    CATEGORY = "Video"
    
    def __init__(self):
        self.has_mediapipe = False
        self.use_new_api = False
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        
        try:
            import mediapipe as mp
            self.mp = mp
            
            # Check for legacy API (Solutions)
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_face_mesh = mp.solutions.face_mesh
                self.mp_drawing = mp.solutions.drawing_utils
                self.has_mediapipe = True
                self.use_new_api = False
                print("[FaceFrameDetector] Using MediaPipe Legacy Solutions API")
            # Check for new API (Tasks)
            else:
                try:
                    from mediapipe.tasks import python
                    from mediapipe.tasks.python import vision
                    self.mp_python = python
                    self.mp_vision = vision
                    self.has_mediapipe = True
                    self.use_new_api = True
                    print("[FaceFrameDetector] Using MediaPipe Tasks API (New)")
                    self._ensure_models_exist()
                except ImportError:
                     print("[FaceFrameDetector] MediaPipe Tasks API not found.")
                     
        except ImportError:
            self.has_mediapipe = False
            print("[FaceFrameDetector] MediaPipe not found. Install with: pip install mediapipe")
        
        try:
            import cv2
            self.cv2 = cv2
            self.has_cv2 = True
        except ImportError:
            self.has_cv2 = False
            print("[FaceFrameDetector] OpenCV not found. Some features limited.")

    def _ensure_models_exist(self):
        """Download detection models for the new API if missing"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
        models = {
            "detector.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            "landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        }
        
        for filename, url in models.items():
            path = os.path.join(self.models_dir, filename)
            if not os.path.exists(path):
                print(f"[FaceFrameDetector] Downloading model {filename}...")
                try:
                    urllib.request.urlretrieve(url, path)
                except Exception as e:
                    print(f"[FaceFrameDetector] Failed to download {filename}: {e}")

    def detect_best_face_frame(self, images, detection_method="mediapipe", 
                              min_face_size=50, eye_open_threshold=0.22, 
                              face_confidence=0.7, scan_limit=20, 
                              require_frontal=True, prefer_center=True,
                              debug_output=False, enable_face_detection=True):
        
        if not enable_face_detection:
            print("Face detection disabled, returning last frame")
            last_frame = images[-1]
            return (last_frame.unsqueeze(0) if last_frame.dim() == 3 else last_frame, 0, json.dumps({"status": "skipped"}), last_frame)
        
        if images is None or images.shape[0] == 0:
            empty_frame = torch.zeros(1, 64, 64, 3)
            return (empty_frame, 0, json.dumps({"error": "empty_input"}), empty_frame)
        
        total_frames = images.shape[0]
        scan_frames = min(scan_limit, total_frames)
        
        best_frame = None
        best_frame_idx = -1
        best_score = 0.0
        detection_results = []
        debug_frames = []
        
        # Analyze last frame first
        last_frame = images[-1]
        last_frame_result = self._detect_face(last_frame, detection_method, min_face_size, eye_open_threshold, face_confidence)
        
        if require_frontal and not last_frame_result["is_frontal"]:
            last_frame_result["score"] *= 0.5
        if prefer_center:
            last_frame_result["score"] *= self._calculate_center_bonus(last_frame_result, last_frame.shape)
        
        fallback_frame = None
        fallback_idx = 0
        fallback_score = 0
        
        for i in range(scan_frames):
            frame_idx = total_frames - 1 - i
            if frame_idx < 0: break
            
            # Skip very first few frames of video to avoid glitches if possible
            if frame_idx < 5 and total_frames > 10:
                continue
                
            frame = images[frame_idx]
            
            if i == 0:
                result = last_frame_result.copy()
            else:
                result = self._detect_face(frame, detection_method, min_face_size, eye_open_threshold, face_confidence)
                if require_frontal and not result["is_frontal"]:
                    result["score"] *= 0.5
                if prefer_center:
                    result["score"] *= self._calculate_center_bonus(result, frame.shape)
            
            detection_results.append({"frame_idx": frame_idx, "frames_from_end": i, **result})
            
            if debug_output:
                debug_frames.append(self._create_debug_overlay(frame, result))
            
            # Perfect frame criteria
            if result["has_face"] and result["eyes_open"] and result["score"] > 0.3:
                best_frame = frame
                best_frame_idx = i
                best_score = result["score"]
                print(f"Found perfect frame at index {frame_idx}")
                break
            
            # Fallback criteria (best EAR)
            elif result["has_face"] and result["score"] > 0.3:
                current_ear = result.get("eye_aspect_ratio", 0.0)
                if current_ear > fallback_score:
                    fallback_frame = frame
                    fallback_idx = i
                    fallback_score = current_ear
        
        if best_frame is None:
            if fallback_frame is not None:
                best_frame = fallback_frame
                best_frame_idx = fallback_idx
                print(f"Using fallback frame (best eyes) at index {total_frames - 1 - fallback_idx}")
            else:
                print("No good faces found, using last frame")
                best_frame = images[-1]
                best_frame_idx = 0

        metadata = {
            "total_frames_scanned": scan_frames,
            "selected_frames_from_end": best_frame_idx,
            "method": detection_method
        }
        
        if debug_output and debug_frames:
            debug_overlay = self._compile_debug_frames(debug_frames)
        else:
            debug_overlay = best_frame.unsqueeze(0) if best_frame.dim() == 3 else best_frame
            
        return (best_frame.unsqueeze(0) if best_frame.dim() == 3 else best_frame, best_frame_idx, json.dumps(metadata), debug_overlay)

    def _detect_face(self, frame, method, min_face_size, eye_threshold, confidence):
        if not self.has_mediapipe:
            return self._detect_bbox_fallback(frame, min_face_size, confidence)
            
        if self.use_new_api:
            return self._detect_with_tasks(frame, min_face_size, eye_threshold, confidence, detailed=(method=="mediapipe_detailed"))
        else:
            if method == "mediapipe_detailed":
                return self._detect_with_mediapipe_detailed(frame, min_face_size, eye_threshold, confidence)
            return self._detect_with_mediapipe(frame, min_face_size, eye_threshold, confidence)

    # --- New API (Tasks) Implementation ---
    def _detect_with_tasks(self, frame, min_face_size, eye_threshold, confidence, detailed=False):
        result = self._get_empty_result()
        result["method"] = "mediapipe_tasks"
        
        cv2_frame = self._tensor_to_cv2(frame)
        rgb_frame = self._bgr_to_rgb(cv2_frame)
        h, w = rgb_frame.shape[:2]
        
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 1. Face Detection
        model_path = os.path.join(self.models_dir, "detector.tflite")
        if not os.path.exists(model_path):
            print("Model detector.tflite missing")
            return self._detect_bbox_fallback(frame, min_face_size, confidence)

        base_options = self.mp_python.BaseOptions(model_asset_path=model_path)
        options = self.mp_vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=confidence)
        
        with self.mp_vision.FaceDetector.create_from_options(options) as detector:
            det_result = detector.detect(mp_image)
            
            if det_result.detections:
                # Filter valid faces
                valid_faces = []
                for det in det_result.detections:
                    bbox = det.bounding_box
                    if min(bbox.width, bbox.height) >= min_face_size:
                        valid_faces.append(det)
                
                if valid_faces:
                    result["has_face"] = True
                    result["num_faces"] = len(valid_faces)
                    
                    # Process primary face
                    primary = valid_faces[0]
                    bbox = primary.bounding_box
                    result["face_bbox"] = (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
                    result["score"] = primary.categories[0].score if primary.categories else 0.0
                    
                    # Frontal check
                    aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 0
                    result["is_frontal"] = 0.7 <= aspect_ratio <= 1.3
                    
                    # Check eyes (using Mesh if detailed or if we need accurate EAR)
                    # For basic mode, we might skip full mesh, but code asks for eyes_open.
                    # We'll use Landmarker for EAR if detailed OR if we want robust eye check
                    # To mimic old behavior, we try to get EAR for open/close check.
                    ear = self._analyze_eyes_with_tasks(rgb_frame)
                    result["eye_aspect_ratio"] = ear
                    result["eyes_open"] = ear > eye_threshold
                    
        return result

    def _analyze_eyes_with_tasks(self, rgb_frame):
        model_path = os.path.join(self.models_dir, "landmarker.task")
        if not os.path.exists(model_path):
            return 0.15 # Default closed-ish
            
        base_options = self.mp_python.BaseOptions(model_asset_path=model_path)
        options = self.mp_vision.FaceLandmarkerOptions(
            base_options=base_options, 
            num_faces=1,
            min_face_detection_confidence=0.5
        )
        
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            with self.mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
                land_result = landmarker.detect(mp_image)
                
                if land_result.face_landmarks:
                    landmarks = land_result.face_landmarks[0] # List of NormalizedLandmark
                    # Indices for Mesh (468 points model) - same as old
                    return self._calculate_ear_from_list(landmarks)
        except Exception as e:
            debug_print(f"Tasks EAR error: {e}")
            
        return 0.15

    def _calculate_ear_from_list(self, landmarks):
        # Helper to calculate EAR from list of objects with .x and .y
        left_indices = [362, 380, 374, 263, 386, 385]
        right_indices = [33, 159, 158, 133, 153, 145]
        
        def get_pt(idx):
            return np.array([landmarks[idx].x, landmarks[idx].y])
            
        def eye_ratio(indices):
            p = [get_pt(i) for i in indices]
            A = np.linalg.norm(p[1] - p[5])
            B = np.linalg.norm(p[2] - p[4])
            C = np.linalg.norm(p[0] - p[3])
            return (A + B) / (2.0 * C) if C > 0 else 0
            
        return (eye_ratio(left_indices) + eye_ratio(right_indices)) / 2

    # --- Old API (Solutions) Implementation ---
    def _detect_with_mediapipe(self, frame, min_face_size, eye_threshold, confidence):
        result = self._get_empty_result()
        result["method"] = "mediapipe_legacy"
        
        try:
            cv2_frame = self._tensor_to_cv2(frame)
            rgb_frame = self._bgr_to_rgb(cv2_frame)
            
            with self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=confidence) as face_detection:
                face_results = face_detection.process(rgb_frame)
                
                if face_results.detections:
                    valid_faces = []
                    h, w = rgb_frame.shape[:2]
                    
                    for detection in face_results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        if min(width, height) >= min_face_size:
                            valid_faces.append(detection)
                    
                    if valid_faces:
                        result["has_face"] = True
                        result["num_faces"] = len(valid_faces)
                        first = valid_faces[0]
                        
                        bbox = first.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        result["face_bbox"] = (x, y, width, height)
                        result["score"] = first.score[0]
                        result["is_frontal"] = 0.7 <= (width/height) <= 1.3
                        
                        ear = self._analyze_eyes_with_facemesh(rgb_frame)
                        result["eye_aspect_ratio"] = ear
                        result["eyes_open"] = ear > eye_threshold
                        
        except Exception as e:
            print(f"Legacy detection error: {e}")
            
        return result

    def _detect_with_mediapipe_detailed(self, frame, min_face_size, eye_threshold, confidence):
        # For legacy, detailed just adds landmarks to the result, effectively same for this logic
        return self._detect_with_mediapipe(frame, min_face_size, eye_threshold, confidence)

    def _analyze_eyes_with_facemesh(self, rgb_frame):
        try:
            with self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    return self._calculate_eye_aspect_ratio(results.multi_face_landmarks[0])
        except Exception:
            pass
        return 0.15

    def _calculate_eye_aspect_ratio(self, landmarks_obj):
        # Legacy: landmarks_obj has .landmark attribute which is a list
        return self._calculate_ear_from_list(landmarks_obj.landmark)

    # --- Common Helpers ---
    def _get_empty_result(self):
        return {
            "has_face": False, "eyes_open": False, "is_frontal": False,
            "score": 0.0, "face_bbox": None, "eye_aspect_ratio": 0.0,
            "num_faces": 0
        }

    def _detect_bbox_fallback(self, frame, min_face_size, confidence):
        result = self._get_empty_result()
        result["method"] = "bbox_fallback"
        # Simple center crop logic for fallback
        if isinstance(frame, torch.Tensor): h, w = frame.shape[-2:]
        else: h, w = frame.shape[:2]
        
        cw, ch = int(w*0.4), int(h*0.4)
        if min(cw, ch) >= min_face_size:
            result["has_face"] = True
            result["score"] = confidence
            result["eyes_open"] = True # Assume best
            result["face_bbox"] = ((w-cw)//2, (h-ch)//2, cw, ch)
        return result

    def _calculate_center_bonus(self, result, frame_shape):
        if not result["face_bbox"]: return 1.0
        x, y, w, h = result["face_bbox"]
        fh, fw = (frame_shape[-2:] if len(frame_shape) >= 2 else frame_shape[:2])
        cx, cy = x + w/2, y + h/2
        dist = np.sqrt(((cx - fw/2)/(fw/2))**2 + ((cy - fh/2)/(fh/2))**2)
        return max(0.5, 1.0 - dist * 0.3)

    def _create_debug_overlay(self, frame, result):
        img = self._tensor_to_cv2(frame)
        if result["face_bbox"] and self.has_cv2:
            x, y, w, h = result["face_bbox"]
            color = (0, 255, 0) if result["eyes_open"] else (0, 255, 255)
            self.cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            self.cv2.putText(img, f"EAR: {result.get('eye_aspect_ratio', 0):.2f}", (x, y-10), 
                           self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img

    def _compile_debug_frames(self, frames):
        tensors = []
        for f in frames:
            if isinstance(f, np.ndarray):
                f = self._bgr_to_rgb(f)
                tensors.append(torch.from_numpy(f).float() / 255.0)
            else: tensors.append(f)
        return torch.stack(tensors)

    def _tensor_to_cv2(self, tensor):
        if tensor.dim() == 4: tensor = tensor[0]
        img = tensor.cpu().numpy()
        if img.shape[0] == 3: img = np.transpose(img, (1, 2, 0))
        img = (img * 255).astype(np.uint8)
        if self.has_cv2 and len(img.shape) == 3: img = self.cv2.cvtColor(img, self.cv2.COLOR_RGB2BGR)
        return img

    def _bgr_to_rgb(self, bgr):
        if self.has_cv2: return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        return bgr

NODE_CLASS_MAPPINGS = {"FaceFrameDetector": FaceFrameDetector}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceFrameDetector": "Face Frame Detector"}