#!/usr/bin/env python3
"""
Gradio Dashboard for 3D Reconstruction with Distance Analysis and Safety Warnings
===================================================================================

Features:
- Upload image for 3D reconstruction
- Automatic object detection using YOLO
- 3D depth estimation using MiDaS
- Point cloud extraction and mesh generation
- Distance calculation between person and objects
- 2D vs 3D distance comparison analysis
- Safety warnings based on proximity thresholds
- Real-time visualization and results display
"""

import os
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torchvision.transforms as T
from torchvision import models

from ultralytics import YOLO
import open3d as o3d

import gradio as gr
from datetime import datetime

# =========================
# CONFIGURATION
# =========================

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
SAFETY_THRESHOLD_METERS = 1.0

# Reconstruction parameters (matching integrated_reconstruction_v2.py)
POISSON_DEPTH = 10
MESH_SMOOTHING_ITERATIONS = 15
OUTLIER_REMOVAL_NEIGHBORS = 50
DENSITY_THRESHOLD_PERCENTILE = 15

# COCO classes
COCO_CLASSES = {
    0: "person",
    56: "chair",
    57: "couch",
    59: "bed",
    60: "dining table",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    69: "oven",
    71: "sink",
    73: "book",
    75: "vase",
}
ALL_DETECT_CLASSES = list(COCO_CLASSES.keys())

# ShapeNet configuration
SHAPENET_DIR = os.path.join(os.path.dirname(__file__), "shapenet", "PartAnnotation")
SHAPENET_CATEGORIES = {
    "chair": "03001627",
    "table": "04379243",
    "dining table": "04379243",
    "vase": "03797390",
    "bottle": "03797390",
    "lamp": "03636649",
    "laptop": "03642806",
}
USE_SHAPENET_MATCHING = True
SHAPENET_MATCH_THRESHOLD = 0.25  # lower is stricter; try 0.2-0.3
SHAPENET_SAMPLE_COUNT = 25       # number of candidate models to sample per category

# Color mapping for objects
OBJECT_COLORS = {
    'person': (0, 255, 0),
    'chair': (255, 0, 0),
    'dining table': (0, 165, 255),
    'table': (0, 165, 255),
    'couch': (255, 128, 0),
    'bed': (128, 0, 255),
    'default': (255, 255, 0)
}

# Safety thresholds for different objects
SAFETY_THRESHOLDS = {
    'person': 0.5,
    'default': SAFETY_THRESHOLD_METERS
}

# =========================
# GLOBAL STATE & MODELS
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MiDaS model
print("Loading MiDaS depth estimation model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform_depth = midas_transforms.small_transform

# Load YOLO model
print("Loading YOLO object detection model...")
yolo_model = YOLO("yolov8n.pt")

print("✅ Models loaded successfully!")


# =========================
# UTILITY FUNCTIONS
# =========================

def load_image_rgb(path: str):
    """Load image and convert to RGB."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"Failed to read image from '{path}'")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, rgb


def depth_from_image(img_rgb, device, midas, transform_depth):
    """Estimate depth map from RGB image using MiDaS."""
    H, W = img_rgb.shape[:2]
    
    depth_input = transform_depth(img_rgb).to(device)
    with torch.no_grad():
        depth_pred = midas(depth_input)
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

    depth_map = depth_pred[0].cpu().numpy() if depth_pred.ndim == 3 else depth_pred.cpu().numpy()
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min) * 9.5 + 0.5
    return depth_map


def detect_all_objects_yolo(img_rgb, yolo_model):
    """Detect all objects using YOLO."""
    results = yolo_model.predict(img_rgb, classes=ALL_DETECT_CLASSES, conf=0.3, verbose=False)
    
    detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for bbox, cls_id, conf in zip(boxes, classes, confidences):
            cls_id = int(cls_id)
            detection = {
                'bbox': bbox,
                'class_id': cls_id,
                'class_name': COCO_CLASSES.get(cls_id, f"class_{cls_id}"),
                'confidence': float(conf)
            }
            detections.append(detection)
    
    return detections


def backproject_bbox_to_points(bbox, depth, K, n_samples=5000):
    """Back-project pixels to 3D."""
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth.shape[1]-1, x2), min(depth.shape[0]-1, y2)

    h_roi, w_roi = max(0, y2-y1), max(0, x2-x1)
    if h_roi == 0 or w_roi == 0:
        return np.empty((0, 3))

    n_samples = min(n_samples, h_roi * w_roi)
    if n_samples == 0:
        return np.empty((0, 3))

    u = np.random.randint(x1, x2, n_samples)
    v = np.random.randint(y1, y2, n_samples)

    Z = depth[v, u]
    valid = Z > 0.1
    if not np.any(valid):
        return np.empty((0, 3))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z, u, v = Z[valid], u[valid].astype(np.float32), v[valid].astype(np.float32)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return np.stack([X, Y, Z], axis=1)


def create_camera_matrix(H, W, fx=1000, fy=1000):
    """Create camera intrinsics matrix."""
    cx, cy = W / 2.0, H / 2.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return K


def extract_object_point_cloud(img_rgb, depth_map, K, object_bbox, n_samples=50000):
    """Extract point cloud from detected object."""
    points = backproject_bbox_to_points(object_bbox, depth_map, K, n_samples=n_samples)
    return points


def create_mesh_from_points(points, depth=POISSON_DEPTH, smooth_iterations=MESH_SMOOTHING_ITERATIONS):
    """Create 3D mesh from point cloud (matches integrated_reconstruction_v2.py)."""
    if len(points) < 4:
        return None, None
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Normalize
    center = pcd.get_center()
    pcd.translate(-center)
    scale = np.max(pcd.get_max_bound() - pcd.get_min_bound())
    if scale > 0:
        pcd.scale(1.0 / scale, center=[0, 0, 0])

    # Statistical outlier removal
    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=OUTLIER_REMOVAL_NEIGHBORS, std_ratio=1.5
    )
    
    if len(pcd_clean.points) < 4:
        return pcd_clean, None

    # Estimate normals
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
    )
    pcd_clean.orient_normals_consistent_tangent_plane(100)

    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_clean,
            depth=depth,
            width=0,
            scale=1.1,
            linear_fit=False
        )

        # Density filtering
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, DENSITY_THRESHOLD_PERCENTILE / 100.0)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Cleanup and refinement
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
        mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()
        
        print(f"    ✓ Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        return pcd_clean, mesh
    except Exception as e:
        print(f"Mesh creation failed: {e}")
        return pcd_clean, None


def save_depth_visualization(depth_map):
    """Save depth map visualization."""
    d_min, d_max = float(depth_map.min()), float(depth_map.max())
    depth_vis = (depth_map - d_min) / (d_max - d_min + 1e-8)
    depth_vis = (255.0 * (1.0 - depth_vis)).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)


def save_detections_overlay(img_bgr, detections):
    """Save detection visualization."""
    det_img = img_bgr.copy()
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        class_name = det['class_name']
        conf = det['confidence']
        color = OBJECT_COLORS.get(class_name, OBJECT_COLORS['default'])
        
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(det_img, (x1, y1-text_h-5), (x1+text_w, y1), color, -1)
        cv2.putText(det_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    return cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)


def min_distance_between_sets(A: np.ndarray, B: np.ndarray, sample_a: int = 500, sample_b: int = 500) -> float:
    """Calculate minimum distance between two point sets."""
    if len(A) == 0 or len(B) == 0:
        return float("inf")
    if len(A) > sample_a:
        A = A[np.random.choice(len(A), sample_a, replace=False)]
    if len(B) > sample_b:
        B = B[np.random.choice(len(B), sample_b, replace=False)]
    
    min_d = float("inf")
    chunk = 200
    for i in range(0, len(A), chunk):
        a_chunk = A[i : i + chunk]
        dists = np.linalg.norm(a_chunk[:, None, :] - B[None, :, :], axis=2)
        local_min = dists.min()
        if local_min < min_d:
            min_d = local_min
    return float(min_d)


# =========================
# SHAPENET MATCHING HELPERS
# =========================

def load_shapenet_model(category_id: str, model_idx: int = 0):
    """Load a ShapeNet model point cloud (.pts) for given category and index."""
    object_dir = os.path.join(SHAPENET_DIR, category_id, "points")
    try:
        object_files = list(Path(object_dir).glob("*.pts"))
    except Exception:
        object_files = []
    
    if not object_files:
        return None
    
    if model_idx >= len(object_files):
        model_idx = np.random.randint(0, len(object_files))
    
    object_file = object_files[model_idx]
    points = []
    try:
        with open(object_file, 'r') as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                if len(values) >= 3:
                    points.append(values[:3])
    except Exception:
        return None
    
    points = np.array(points)
    if points.size == 0:
        return None
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Normalize to unit scale centered at origin
    center = pcd.get_center()
    pcd.translate(-center)
    scale = np.max(pcd.get_max_bound() - pcd.get_min_bound())
    if scale > 0:
        pcd.scale(1.0 / scale, center=[0, 0, 0])
    
    return pcd


def find_best_shapenet_match(query_pcd: o3d.geometry.PointCloud, category_name: str, num_samples: int = SHAPENET_SAMPLE_COUNT):
    """Find best matching ShapeNet model by bidirectional point cloud distance."""
    if category_name not in SHAPENET_CATEGORIES:
        return None, -1.0
    
    category_id = SHAPENET_CATEGORIES[category_name]
    object_dir = os.path.join(SHAPENET_DIR, category_id, "points")
    try:
        object_files = list(Path(object_dir).glob("*.pts"))
    except Exception:
        object_files = []
    if not object_files:
        return None, -1.0
    
    num_samples = min(num_samples, len(object_files))
    sample_indices = np.random.choice(len(object_files), num_samples, replace=False)
    
    best_distance = float('inf')
    best_idx = -1
    
    print(f"  Searching {num_samples} {category_name} models...")
    for idx in sample_indices:
        shapenet_pcd = load_shapenet_model(category_id, idx)
        if shapenet_pcd is None:
            continue
        
        try:
            distances1 = np.asarray(query_pcd.compute_point_cloud_distance(shapenet_pcd))
            distances2 = np.asarray(shapenet_pcd.compute_point_cloud_distance(query_pcd))
            distance = float(distances1.mean() + distances2.mean())
        except Exception:
            continue
        
        if distance < best_distance:
            best_distance = distance
            best_idx = idx
    
    if best_idx == -1:
        return None, -1.0
    print(f"  ✓ Best match found (distance: {best_distance:.4f})")
    return best_idx, best_distance


def reconstruct_from_shapenet(query_pcd: o3d.geometry.PointCloud, category_name: str):
    """Reconstruct mesh using the best ShapeNet model for the given category."""
    best_idx, distance = find_best_shapenet_match(query_pcd, category_name)
    if best_idx is None or best_idx == -1:
        return None, None
    
    category_id = SHAPENET_CATEGORIES[category_name]
    shapenet_pcd = load_shapenet_model(category_id, best_idx)
    if shapenet_pcd is None:
        return None, None
    
    # Estimate normals for reconstruction
    shapenet_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
    )
    shapenet_pcd.orient_normals_consistent_tangent_plane(100)
    
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            shapenet_pcd,
            depth=10,
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        densities = np.asarray(densities)
        if len(densities) > 0:
            density_threshold = np.quantile(densities, 0.15)
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)
        
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
        mesh.compute_vertex_normals()
        
        return mesh, shapenet_pcd
    except Exception as e:
        print(f"  ShapeNet reconstruction failed: {e}")
        return None, shapenet_pcd


# =========================
# MAIN PROCESSING FUNCTION
# =========================

def process_image(image_input, progress=gr.Progress()):
    """
    Main processing function for the dashboard.
    Takes an image and returns reconstruction results with distance analysis.
    Shows detection and depth early, then processes reconstruction.
    """
    
    try:
        # Handle PIL Image input from Gradio
        if isinstance(image_input, Image.Image):
            temp_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
            image_input.save(temp_path)
            image_path = temp_path
        else:
            image_path = image_input

        print(f"\n{'='*60}")
        print(f"Processing image: {image_path}")
        print(f"{'='*60}")

        # Load image
        progress(0, desc="Loading image...")
        img_bgr, img_rgb = load_image_rgb(image_path)
        H, W = img_rgb.shape[:2]
        print(f"✓ Image loaded: {W}x{H}")

        # Create camera matrix
        K = create_camera_matrix(H, W)
        print(f"✓ Camera matrix created")

        # Estimate depth
        progress(0.1, desc="Estimating depth map...")
        print("🔍 Estimating depth map...")
        depth_map = depth_from_image(img_rgb, device, midas, transform_depth)
        depth_vis = save_depth_visualization(depth_map)
        print(f"✓ Depth estimated (range: {depth_map.min():.2f}-{depth_map.max():.2f})")

        # Detect objects
        progress(0.2, desc="Detecting objects...")
        print("🎯 Detecting objects with YOLO...")
        detections = detect_all_objects_yolo(img_rgb, yolo_model)
        print(f"✓ Found {len(detections)} objects")
        for i, det in enumerate(detections):
            print(f"  [{i}] {det['class_name']}: {det['confidence']:.2f}")

        # Save detection overlay
        det_overlay = save_detections_overlay(img_bgr, detections)
        
        # EARLY RETURN: Show detection and depth immediately
        early_summary = f"""
## 🔍 Detection Complete

**Image Size:** {W}x{H}
**Objects Detected:** {len(detections)}

{chr(10).join([f"- {det['class_name']} ({det['confidence']:.2f})" for det in detections])}

⏳ **Reconstructing 3D meshes... This will take a few minutes.**
"""
        
        # Yield early results while reconstruction continues
        yield (
            det_overlay,
            depth_vis, 
            "## ⏳ Reconstruction in progress...\n\nPlease wait while 3D meshes are being generated.",
            "## ⏳ 2D vs 3D comparison in progress...\n\nPreparing distance table and explanation.",
            early_summary,
            gr.update(choices=[], value=None, interactive=False),
            {},
            None,
            "Reconstruction in progress..."
        )

        # Extract point clouds for all objects
        objects_data = []
        person_points = None
        
        print("\n🌐 Extracting 3D point clouds...")
        progress(0.3, desc="Extracting 3D point clouds...")
        for idx, det in enumerate(detections):
            class_name = det['class_name']
            bbox = det['bbox']
            
            progress(0.3 + (0.5 * idx / len(detections)), desc=f"Reconstructing {class_name} ({idx+1}/{len(detections)})...")
            print(f"\n  [{idx+1}/{len(detections)}] Processing {class_name}...")
            
            # Extract points
            points = extract_object_point_cloud(img_rgb, depth_map, K, bbox, n_samples=50000)
            
            if len(points) < 100:
                print(f"    ⚠️  Skipped (too few points: {len(points)})")
                continue
            
            print(f"    ✓ Extracted {len(points)} points")
            
            # Store for distance calculation
            det['points'] = points
            
            if len(points) > 10000:
                mesh = None
                pcd = None
                
                # Use pure Poisson reconstruction (same as CLI)
                try:
                    print(f"    [POISSON] Reconstructing...")
                    pcd, mesh = create_mesh_from_points(points)
                except:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    mesh = None
                
                if mesh:
                    # Flip person mesh if upside down (match CLI behavior)
                    if class_name == 'person':
                        mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0)), center=(0, 0, 0))
                    
                    # Color meshes (match CLI)
                    if class_name == 'person':
                        mesh.paint_uniform_color([0.8, 0.6, 0.5])
                    elif 'chair' in class_name:
                        mesh.paint_uniform_color([0.6, 0.4, 0.2])
                    elif 'table' in class_name:
                        mesh.paint_uniform_color([0.7, 0.5, 0.3])
                    else:
                        mesh.paint_uniform_color([0.5, 0.5, 0.5])
                    
                    obj_info = {
                        'class_name': class_name,
                        'bbox': bbox,
                        'points': points,
                        'pcd': pcd,
                        'mesh': mesh,
                        'index': idx
                    }
                    objects_data.append(obj_info)
                    print(f"    ✓ Mesh: {len(mesh.vertices)} vertices")
                else:
                    obj_info = {
                        'class_name': class_name,
                        'bbox': bbox,
                        'points': points,
                        'pcd': pcd,
                        'mesh': None,
                        'index': idx
                    }
                    objects_data.append(obj_info)
            else:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                obj_info = {
                    'class_name': class_name,
                    'bbox': bbox,
                    'points': points,
                    'pcd': pcd,
                    'mesh': None,
                    'index': idx
                }
                objects_data.append(obj_info)

        # Calculate distances and warnings (match CLI method exactly)
        print("\n📏 Calculating distances...")
        distance_results = []
        comparison_rows = []
        warnings_text = []
        
        # Find person
        person_points = None
        for obj_info in objects_data:
            if obj_info['class_name'].lower() == 'person':
                person_points = obj_info['points']
                break
        
        if person_points is None:
            warnings_text.append("⚠️  WARNING: No person detected in image!")
        else:
            # Calculate person's 3D center
            person_center = np.mean(person_points, axis=0)
            person_x, person_y, person_z = person_center
            person_bbox = None
            for obj_info in objects_data:
                if obj_info['class_name'].lower() == 'person':
                    person_bbox = obj_info['bbox']
                    break

            image_diagonal = float(np.sqrt((W ** 2) + (H ** 2)))
            if person_bbox is not None:
                px1, py1, px2, py2 = person_bbox
                person_cx_2d = (px1 + px2) / 2.0
                person_cy_2d = (py1 + py2) / 2.0
            else:
                person_cx_2d, person_cy_2d = W / 2.0, H / 2.0
            
            for obj_info in objects_data:
                if obj_info['class_name'].lower() == 'person':
                    continue
                
                obj_class = obj_info['class_name']
                obj_points = obj_info['points']
                
                # Calculate object's 3D center
                obj_center = np.mean(obj_points, axis=0)
                obj_x, obj_y, obj_z = obj_center

                # 2D center distance from image-space bounding boxes
                ox1, oy1, ox2, oy2 = obj_info['bbox']
                obj_cx_2d = (ox1 + ox2) / 2.0
                obj_cy_2d = (oy1 + oy2) / 2.0
                dist_2d_px = float(np.sqrt((obj_cx_2d - person_cx_2d) ** 2 + (obj_cy_2d - person_cy_2d) ** 2))
                dist_2d_norm = (dist_2d_px / image_diagonal) if image_diagonal > 0 else 0.0
                depth_gap = float(abs(obj_z - person_z))
                
                # Calculate 3D Euclidean distance between centers
                avg_distance = np.sqrt(
                    (obj_x - person_x)**2 + 
                    (obj_y - person_y)**2 + 
                    (obj_z - person_z)**2
                )
                
                # Calculate minimum distance (closest points between objects)
                # Sample every 10th point for efficiency (match CLI)
                min_distance = float('inf')
                for p_point in person_points[::10]:
                    distances = np.linalg.norm(obj_points - p_point, axis=1)
                    min_dist = np.min(distances)
                    if min_dist < min_distance:
                        min_distance = min_dist
                
                # Get safety threshold for this object
                safety_threshold = SAFETY_THRESHOLDS.get(obj_class, SAFETY_THRESHOLDS['default'])
                
                is_warning = min_distance < safety_threshold
                status = "🚨 WARNING" if is_warning else "✓ SAFE"
                
                distance_results.append({
                    'object': obj_class,
                    'distance_m': min_distance,
                    'avg_distance_m': avg_distance,
                    'threshold_m': safety_threshold,
                    'status': status,
                    'is_warning': is_warning
                })
                
                print(f"  {status}: {obj_class} - Min={min_distance:.3f}m, Avg={avg_distance:.3f}m (threshold: {safety_threshold}m)")
                
                if is_warning:
                    warnings_text.append(
                        f"🚨 PROXIMITY ALERT: {obj_class} is only {min_distance:.3f}m away "
                        f"(safety threshold: {safety_threshold}m)"
                    )

                comparison_rows.append({
                    'object': obj_class,
                    'distance_2d_px': dist_2d_px,
                    'distance_2d_norm': dist_2d_norm,
                    'distance_3d_min_m': min_distance,
                    'distance_3d_avg_m': avg_distance,
                    'depth_gap_m': depth_gap,
                })

        # Create results table
        results_text = "## 📊 Distance Analysis Results\n\n"
        results_text += "| Object | Min Distance (m) | Avg Distance (m) | Threshold (m) | Status |\n"
        results_text += "|--------|------------------|------------------|---------------|--------|\n"
        
        for result in distance_results:
            status_emoji = "🚨" if result['is_warning'] else "✓"
            results_text += f"| {result['object']} | {result['distance_m']:.3f} | {result.get('avg_distance_m', 0):.3f} | {result['threshold_m']:.3f} | {status_emoji} {result['status']} |\n"

        # Create warnings section
        warnings_section = "\n## ⚠️ Safety Warnings\n\n"
        if warnings_text:
            warnings_section += "\n".join(warnings_text)
        else:
            warnings_section += "✓ No safety warnings - all objects are at safe distance!"

        final_results_text = results_text + warnings_section

        # Build 2D vs 3D tab content as table + explanation (no graph)
        if not comparison_rows:
            comparison_text = (
                "## 🔬 2D vs 3D Distance Comparison\n\n"
                "Not enough objects to compare (need a detected person and at least one other object)."
            )
        else:
            comparison_text = "## 🔬 2D vs 3D Distance Comparison\n\n"
            comparison_text += "| Object | 2D Distance (px) | 2D (% image diagonal) | 3D Min Distance (m) | 3D Avg Distance (m) | Depth Gap (m) |\n"
            comparison_text += "|--------|------------------:|----------------------:|--------------------:|--------------------:|--------------:|\n"

            for row in comparison_rows:
                comparison_text += (
                    f"| {row['object']} | {row['distance_2d_px']:.1f} | {row['distance_2d_norm']*100:.1f}% | "
                    f"{row['distance_3d_min_m']:.3f} | {row['distance_3d_avg_m']:.3f} | {row['depth_gap_m']:.3f} |\n"
                )

            by_2d = [r['object'] for r in sorted(comparison_rows, key=lambda r: r['distance_2d_px'])]
            by_3d = [r['object'] for r in sorted(comparison_rows, key=lambda r: r['distance_3d_min_m'])]
            rank_mismatch_count = sum(1 for i in range(min(len(by_2d), len(by_3d))) if by_2d[i] != by_3d[i])
            avg_depth_gap = float(np.mean([r['depth_gap_m'] for r in comparison_rows])) if comparison_rows else 0.0

            misleading_cases = [
                r for r in comparison_rows
                if (r['distance_2d_norm'] < 0.15 and r['distance_3d_min_m'] > SAFETY_THRESHOLD_METERS)
                or (r['distance_2d_norm'] > 0.35 and r['distance_3d_min_m'] < SAFETY_THRESHOLD_METERS)
            ]

            comparison_text += "\n### Why 3D Distance Is Better\n"
            comparison_text += "- **2D distance is pixel-based** and does not include depth, so it cannot represent true physical separation.\n"
            comparison_text += "- **3D distance uses reconstructed geometry** and gives metric distances in meters, which are physically meaningful.\n"
            comparison_text += f"- **2D vs 3D nearest-object ranking mismatch:** {rank_mismatch_count} position(s).\n"
            comparison_text += f"- **Average depth separation across compared objects:** {avg_depth_gap:.3f} m.\n"
            if misleading_cases:
                comparison_text += f"- **Misleading 2D cases detected:** {len(misleading_cases)} (objects may look near/far in image but differ in real 3D space).\n"
            else:
                comparison_text += "- **In this image**, 2D and 3D orderings are closer, but 3D remains the accurate metric for safety decisions.\n"

        # Build mesh outputs for viewer
        mesh_state = {}
        mesh_choices: List[str] = []
        default_mesh_label = None
        default_mesh_path = None

        # Write meshes to workspace directory for proper Gradio file serving
        # Create a meshes output directory in the workspace
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        meshes_dir = os.path.join(workspace_dir, "gradio_meshes")
        os.makedirs(meshes_dir, exist_ok=True)
        
        for obj_info in objects_data:
            mesh = obj_info.get('mesh')
            if mesh is None:
                continue
            
            # Validate mesh has actual geometry
            if not mesh.has_vertices() or len(mesh.vertices) < 3:
                print(f"⚠️  Mesh for {obj_info['class_name']} has no vertices, skipping")
                continue
            
            if not mesh.has_triangles() or len(mesh.triangles) < 1:
                print(f"⚠️  Mesh for {obj_info['class_name']} has no triangles, skipping")
                continue

            try:
                # Remove NaN vertices that cause Babylon.js errors
                vertices = np.asarray(mesh.vertices)
                valid_mask = ~np.isnan(vertices).any(axis=1)
                if not valid_mask.all():
                    print(f"⚠️  Removing {(~valid_mask).sum()} NaN vertices from {obj_info['class_name']}")
                    mesh = mesh.select_by_index(np.where(valid_mask)[0])
                
                # Ensure mesh still has geometry after NaN removal
                if len(mesh.vertices) < 3 or len(mesh.triangles) < 1:
                    print(f"⚠️  Mesh became invalid after NaN removal")
                    continue
                
                # Ensure mesh has proper normals (colors already set during reconstruction)
                if not mesh.has_vertex_normals():
                    mesh.compute_vertex_normals()
                
                # Create mesh filename with timestamp to avoid caching issues
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mesh_filename = f"{obj_info['class_name'].replace(' ', '_')}_{obj_info['index']+1}_{timestamp}.obj"
                mesh_path = os.path.join(meshes_dir, mesh_filename)
                
                # Write OBJ file in ASCII format (default) with vertex normals and colors
                o3d.io.write_triangle_mesh(mesh_path, mesh, write_vertex_normals=True, write_vertex_colors=True, write_ascii=True)
                
                # Verify file was created and check for NaN values
                if not os.path.exists(mesh_path):
                    print(f"⚠️  Mesh file not created at {mesh_path}")
                    continue
                
                file_size = os.path.getsize(mesh_path)
                
                # Quick check for NaN in OBJ file
                try:
                    with open(mesh_path, 'r') as f:
                        content = f.read()
                        if '-nan' in content.lower() or 'nan(' in content.lower():
                            print(f"⚠️  WARNING: NaN detected in OBJ file!")
                            # Try to read and re-export to clean mesh
                            mesh_read = o3d.io.read_triangle_mesh(mesh_path)
                            o3d.io.write_triangle_mesh(mesh_path, mesh_read, write_vertex_normals=True, write_vertex_colors=True, write_ascii=True)
                            print(f"   Re-exported mesh to remove NaN values")
                except Exception as check_err:
                    print(f"   (Could not verify NaN in file: {check_err})")
                
                print(f"✓ Exported mesh to {mesh_path}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles, file size: {file_size} bytes")

                label = f"{obj_info['class_name']} #{obj_info['index']+1}"
                mesh_state[label] = mesh_path
                mesh_choices.append(label)

                if default_mesh_label is None:
                    default_mesh_label = label
                    default_mesh_path = mesh_path
            except Exception as mesh_err:
                print(f"⚠️  Failed to export mesh for {obj_info['class_name']}: {mesh_err}")
                import traceback
                traceback.print_exc()

        # Create summary
        progress(0.9, desc="Finalizing results...")
        summary = f"""
## 📈 Processing Summary

**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Image Size:** {W}x{H}

**Detections:** {len(detections)} objects found
**Successfully Reconstructed:** {len(objects_data)} objects
**Distance Calculations:** {len(distance_results)} comparisons

**Safety Status:** {'🚨 WARNINGS PRESENT' if any(r['is_warning'] for r in distance_results) else '✅ ALL SAFE'}
"""

        print(f"\n{'='*60}")
        print("✅ Processing Complete!")
        print(f"{'='*60}\n")

        progress(1.0, desc="Complete!")
        
        # FINAL RETURN with all results
        yield (
            det_overlay,               # detection overlay
            depth_vis,                 # depth visualization
            final_results_text,        # distance and warning analysis
            comparison_text,           # 2D vs 3D table + explanation
            summary,                   # summary statistics
            gr.update(
                choices=mesh_choices,
                value=default_mesh_label,
                interactive=bool(mesh_choices)
            ),                        # dropdown update
            mesh_state,                # mapping label -> mesh path
            default_mesh_path,         # initial 3D model path
            (f"✅ Successfully reconstructed {len(mesh_choices)} objects!" if mesh_choices else "No reconstructed meshes available.")
        )

    except Exception as e:
        error_msg = f"❌ Error processing image: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        yield (
            None,
            None,
            f"## Error\n\n{error_msg}",
            f"## Error\n\n{error_msg}",
            f"## Error\n\n{error_msg}",
            gr.update(choices=[], value=None, interactive=False),
            {},
            None,
            error_msg
        )


def update_mesh_view(selected_label, mesh_state):
    """Return mesh path for selected label."""
    print(f"[DEBUG] update_mesh_view called: selected_label={selected_label}, mesh_state keys={list(mesh_state.keys()) if mesh_state else 'None'}")
    if not mesh_state:
        print("[DEBUG] mesh_state is empty")
        return None
    if not selected_label:
        print("[DEBUG] selected_label is empty")
        return None
    
    mesh_path = mesh_state.get(selected_label)
    if not mesh_path:
        print(f"[DEBUG] No mesh path found for {selected_label}")
        return None
    
    # Verify file exists and has content
    if not os.path.exists(mesh_path):
        print(f"[DEBUG] Mesh file not found: {mesh_path}")
        return None
    
    file_size = os.path.getsize(mesh_path)
    if file_size == 0:
        print(f"[DEBUG] Mesh file is empty: {mesh_path}")
        return None
    
    # Convert to file:// URL for proper cross-platform support
    # Model3D component handles file:// URLs correctly
    file_url = "file:///" + os.path.normpath(mesh_path).replace("\\", "/")
    print(f"[DEBUG] Returning mesh URL: {file_url} (size: {file_size} bytes)")
    return mesh_path  # Return original path - Gradio will handle it


# =========================
# GRADIO INTERFACE
# =========================

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="3D Reconstruction Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎯 3D Object Reconstruction & Safety Analysis Dashboard
        
        Upload an image to automatically:
        - Detect objects using YOLO
        - Estimate depth maps
        - Reconstruct 3D point clouds and meshes
        - Calculate distances between person and objects
        - Generate safety warnings for proximity alerts
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Input")
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    sources=["upload", "webcam"],
                )
                process_btn = gr.Button(
                    "🚀 Process Image",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")
                with gr.Tabs():
                    with gr.TabItem("Detection"):
                        detection_output = gr.Image(
                            label="Object Detection Overlay",
                            type="pil"
                        )
                    
                    with gr.TabItem("Depth Map"):
                        depth_output = gr.Image(
                            label="Depth Estimation",
                            type="pil"
                        )

                    with gr.TabItem("2D vs 3D Distance"):
                        comparison_output = gr.Markdown(
                            label="2D vs 3D Difference"
                        )
                    
                    with gr.TabItem("3D Meshes"):
                        gr.Markdown("### 🧊 Reconstructed Objects")
                        mesh_selector = gr.Dropdown(
                            label="Select Object",
                            choices=[],
                            value=None,
                            interactive=False,
                        )
                        mesh_viewer = gr.Model3D(
                            label="3D Mesh Viewer",
                            clear_color=[1, 1, 1, 0]
                        )
                        mesh_status = gr.Markdown(visible=True)
                        mesh_state = gr.State({})
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📏 Distance & Safety Analysis")
                analysis_output = gr.Markdown(label="Analysis Results")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Summary")
                summary_output = gr.Markdown(label="Processing Summary")
        
        # Set up button click
        process_btn.click(
            fn=process_image,
            inputs=[image_input],
            outputs=[
                detection_output,
                depth_output,
                analysis_output,
                comparison_output,
                summary_output,
                mesh_selector,
                mesh_state,
                mesh_viewer,
                mesh_status
            ],
            show_progress=True
        )

        mesh_selector.change(
            fn=update_mesh_view,
            inputs=[mesh_selector, mesh_state],
            outputs=mesh_viewer,
        )
        
        gr.Markdown("""
        ---
        
        ### 📝 How to Use:
        1. Upload an image containing people and objects
        2. Click "Process Image" to start the analysis
        3. View detection results, depth maps, and distance calculations
        4. Check safety warnings for proximity alerts
        
        ### 🔧 Configuration:
        - **Safety Threshold:** 1.0m for general objects, 0.5m for people
        - **Detection Confidence:** 30% (YOLO)
        - **Depth Model:** MiDaS (Intel)
        - **Reconstruction:** Poisson surface reconstruction
        """)
    
    return demo


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting 3D Reconstruction Dashboard")
    print("="*60)
    
    # Create meshes directory if it doesn't exist
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    meshes_dir = os.path.join(workspace_dir, "gradio_meshes")
    os.makedirs(meshes_dir, exist_ok=True)
    print(f"📁 Mesh output directory: {meshes_dir}")
    
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        allowed_paths=[meshes_dir]  # Allow Gradio to serve mesh files
    )
