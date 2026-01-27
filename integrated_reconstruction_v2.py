#!/usr/bin/env python3
"""
Integrated 2D to 3D Reconstruction Pipeline with Multi-Image Support
--------------------------------------------------------------------

Features:
- Upload single image or process directory
- Automatic depth map generation
- Individual object detection and reconstruction
- ShapeNet model matching for furniture
- Interactive 3D viewer
- Separate output directories per image
"""

import os
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image
import datetime
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision import models

from ultralytics import YOLO
import open3d as o3d


# =========================
# CONFIGURATION
# =========================

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
SAFETY_THRESHOLD_METERS = 1.0

# ShapeNet configuration
SHAPENET_DIR = r"C:\Users\user\Documents\GitHub\Open3D\shapenet\PartAnnotation"
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

# Reconstruction parameters
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


# =========================
# UTILITY FUNCTIONS
# =========================

def find_available_images(directory=None):
    """Scan directory for image files."""
    if directory is None:
        directory = '.'
    
    if not os.path.isdir(directory):
        return []
    
    available_images = []
    for ext in IMAGE_EXTENSIONS:
        for f in os.listdir(directory):
            if f.lower().endswith(ext.lower()):
                full_path = os.path.join(directory, f)
                if os.path.isfile(full_path):
                    available_images.append(full_path)
    
    available_images = sorted(list(set(available_images)))
    return available_images


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


def extract_object_point_cloud(img_rgb, depth_map, K, object_bbox, n_samples=50000):
    """Extract point cloud from detected object."""
    points = backproject_bbox_to_points(object_bbox, depth_map, K, n_samples=n_samples)
    return points


def save_depth_visualization(depth_map, output_path="depth_map.png"):
    """Save depth map visualization."""
    d_min, d_max = float(depth_map.min()), float(depth_map.max())
    depth_vis = (depth_map - d_min) / (d_max - d_min + 1e-8)
    depth_vis = (255.0 * (1.0 - depth_vis)).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    cv2.imwrite(output_path, depth_colored)
    return depth_vis


def save_detections_overlay(img_bgr, detections, output_path="detections.jpg"):
    """Save detection visualization."""
    det_img = img_bgr.copy()
    colors = {
        'person': (0, 255, 0),
        'chair': (255, 0, 0),
        'dining table': (0, 165, 255),
        'default': (255, 255, 0)
    }
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        class_name = det['class_name']
        conf = det['confidence']
        color = colors.get(class_name, colors['default'])
        
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(det_img, (x1, y1-text_h-5), (x1+text_w, y1), color, -1)
        cv2.putText(det_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    cv2.imwrite(output_path, det_img)
    return det_img


def save_mesh_outputs(mesh, pcd, object_name, idx, output_prefix=""):
    """Save mesh and point cloud."""
    if output_prefix:
        prefix = f"{output_prefix}_object_{idx}_{object_name.replace(' ', '_')}"
    else:
        prefix = f"object_{idx}_{object_name.replace(' ', '_')}"
    
    o3d.io.write_triangle_mesh(f"{prefix}_mesh.ply", mesh)
    o3d.io.write_triangle_mesh(f"{prefix}_mesh.obj", mesh)
    o3d.io.write_point_cloud(f"{prefix}_pointcloud.ply", pcd)
    
    return prefix


def save_points_as_pts(points, object_name, idx, output_prefix=""):
    """Save point cloud as .pts file."""
    if output_prefix:
        pts_filename = f"{output_prefix}_object_{idx}_{object_name.replace(' ', '_')}.pts"
    else:
        pts_filename = f"object_{idx}_{object_name.replace(' ', '_')}.pts"
    
    np.savetxt(pts_filename, points, fmt='%.6f', delimiter=' ')
    return pts_filename


def create_mesh_from_points(points, depth=POISSON_DEPTH, smooth_iterations=MESH_SMOOTHING_ITERATIONS):
    """Create 3D mesh from point cloud."""
    print("\n🎨 Creating point cloud...")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    center = pcd.get_center()
    pcd.translate(-center)
    scale = np.max(pcd.get_max_bound() - pcd.get_min_bound())
    if scale > 0:
        pcd.scale(1.0 / scale, center=[0, 0, 0])

    print(f"✓ Point cloud created with {len(points)} points")
    print("\n⚡ Enhancing point cloud quality...")

    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=OUTLIER_REMOVAL_NEIGHBORS, std_ratio=1.5
    )
    print(f"  Removed {len(pcd.points) - len(pcd_clean.points)} outliers")

    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
    )
    pcd_clean.orient_normals_consistent_tangent_plane(100)
    print(f"✓ Enhanced to {len(pcd_clean.points)} points with normals")

    print("\n🔨 Reconstructing high-quality 3D mesh...")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_clean,
        depth=depth,
        width=0,
        scale=1.1,
        linear_fit=False
    )

    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, DENSITY_THRESHOLD_PERCENTILE / 100.0)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print(f"  Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    print("\n✨ Refining mesh...")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
    mesh = mesh.subdivide_midpoint(number_of_iterations=1)
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()

    print(f"✓ Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    return mesh, pcd_clean


def load_shapenet_model(category_id, model_idx=0):
    """Load ShapeNet model."""
    object_dir = os.path.join(SHAPENET_DIR, category_id, "points")
    object_files = list(Path(object_dir).glob("*.pts"))
    
    if not object_files:
        return None
    
    if model_idx >= len(object_files):
        model_idx = np.random.randint(0, len(object_files))
    
    object_file = object_files[model_idx]
    points = []
    with open(object_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if len(values) >= 3:
                points.append(values[:3])
    
    points = np.array(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    center = pcd.get_center()
    pcd.translate(-center)
    scale = np.max(pcd.get_max_bound() - pcd.get_min_bound())
    if scale > 0:
        pcd.scale(1.0 / scale, center=[0, 0, 0])
    
    return pcd


def find_best_shapenet_match(query_pcd, category_name, num_samples=25):
    """Find best matching ShapeNet model."""
    if category_name not in SHAPENET_CATEGORIES:
        return None, -1
    
    category_id = SHAPENET_CATEGORIES[category_name]
    object_dir = os.path.join(SHAPENET_DIR, category_id, "points")
    object_files = list(Path(object_dir).glob("*.pts"))
    
    if not object_files:
        return None, -1
    
    num_samples = min(num_samples, len(object_files))
    sample_indices = np.random.choice(len(object_files), num_samples, replace=False)
    
    best_distance = float('inf')
    best_idx = 0
    
    print(f"  Searching {num_samples} {category_name} models...")

    for idx in sample_indices:
        shapenet_pcd = load_shapenet_model(category_id, idx)
        if shapenet_pcd is None:
            continue
        
        distances1 = np.asarray(query_pcd.compute_point_cloud_distance(shapenet_pcd))
        distances2 = np.asarray(shapenet_pcd.compute_point_cloud_distance(query_pcd))
        distance = distances1.mean() + distances2.mean()
        
        if distance < best_distance:
            best_distance = distance
            best_idx = idx
    
    print(f"  ✓ Best match found (distance: {best_distance:.4f})")
    return best_idx, best_distance


def reconstruct_from_shapenet(query_pcd, category_name):
    """Reconstruct using ShapeNet model."""
    best_idx, distance = find_best_shapenet_match(query_pcd, category_name)
    
    if best_idx == -1:
        return None, None
    
    category_id = SHAPENET_CATEGORIES[category_name]
    shapenet_pcd = load_shapenet_model(category_id, best_idx)
    
    if shapenet_pcd is None:
        return None, None
    
    print(f"  Creating mesh from ShapeNet model...")
    shapenet_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
    )
    shapenet_pcd.orient_normals_consistent_tangent_plane(100)
    
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


def visualize_mesh(mesh, window_name="3D Object"):
    """Visualize mesh."""
    print("\n🎮 Launching 3D viewer...")
    print("   Left click + drag: Rotate | Scroll: Zoom | Q/ESC: Close")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=1000)
    vis.add_geometry(mesh)
    
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.mesh_show_back_face = True
    opt.light_on = True
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)
    
    vis.run()
    vis.destroy_window()
    print("✅ Viewer closed")


# =========================
# MAIN PIPELINE
# =========================

def process_single_image(IMAGE_PATH, device, yolo_model, midas, transform_depth):
    """Process a single image."""
    try:
        # Setup output directory
        image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
        output_dir = f"outputs_{image_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ Output directory: {output_dir}")
        
        output_prefix = os.path.join(output_dir, image_name)
        
        # Load image
        img_bgr, img_rgb = load_image_rgb(IMAGE_PATH)
        H, W = img_rgb.shape[:2]
        print(f"[INFO] Image: {W}x{H}")
        
        K = np.array([[1000.0, 0.0, W/2.0], [0.0, 1000.0, H/2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        
        # STEP 1: Depth
        print("\n" + "=" * 70)
        print("STEP 1: DEPTH ESTIMATION")
        print("=" * 70)
        depth_map = depth_from_image(img_rgb, device, midas, transform_depth)
        save_depth_visualization(depth_map, f"{output_prefix}_depth.png")
        print(f"[OK] Depth: {output_prefix}_depth.png")
        
        # STEP 2: Detection
        print("\n" + "=" * 70)
        print("STEP 2: OBJECT DETECTION")
        print("=" * 70)
        detections = detect_all_objects_yolo(img_rgb, yolo_model)
        
        if not detections:
            print("[ERROR] No objects detected")
            return
        
        print(f"[OK] Detected {len(detections)} objects:")
        for det in detections:
            print(f"     {det['class_name']} ({det['confidence']:.2f})")
        
        save_detections_overlay(img_bgr, detections, f"{output_prefix}_detections.jpg")
        
        # STEP 3: Reconstruction
        print("\n" + "=" * 70)
        print("STEP 3: 3D RECONSTRUCTION")
        print("=" * 70)
        
        reconstructed = []
        
        for i, det in enumerate(detections):
            print(f"\n[{i+1}/{len(detections)}] {det['class_name']}...")
            
            points = extract_object_point_cloud(img_rgb, depth_map, K, det['bbox'], n_samples=50000)
            
            if len(points) < 100:
                print(f"  [SKIP] Not enough points ({len(points)})")
                continue
            
            print(f"  Points: {len(points)}")
            pts_file = save_points_as_pts(points, det['class_name'], i+1, output_prefix)
            
            # Store for distance calculation
            det['points'] = points
            det['bbox_for_distance'] = det['bbox']
            
            if len(points) > 10000:
                mesh = None
                pcd = None
                
                # Use pure Poisson for all objects (same as person reconstruction)
                # Skip ShapeNet - Poisson gives superior accuracy for real-world objects
                try:
                    print(f"  [POISSON] Reconstructing...")
                    mesh, pcd = create_mesh_from_points(points)
                except:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    mesh = None
                
                if mesh:
                    # Flip person mesh if upside down
                    if det['class_name'] == 'person':
                        mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0)), center=(0, 0, 0))
                    
                    # Color
                    if det['class_name'] == 'person':
                        mesh.paint_uniform_color([0.8, 0.6, 0.5])
                    elif 'chair' in det['class_name']:
                        mesh.paint_uniform_color([0.6, 0.4, 0.2])
                    elif 'table' in det['class_name']:
                        mesh.paint_uniform_color([0.7, 0.5, 0.3])
                    else:
                        mesh.paint_uniform_color([0.5, 0.5, 0.5])
                    
                    prefix = save_mesh_outputs(mesh, pcd, det['class_name'], i+1, output_prefix)
                    reconstructed.append({
                        'name': det['class_name'],
                        'mesh': mesh,
                        'pcd': pcd,
                        'prefix': prefix,
                        'vertices': len(mesh.vertices),
                        'triangles': len(mesh.triangles),
                        'pts_file': pts_file,
                        'has_mesh': True,
                        'detection': det
                    })
                    print(f"  ✓ Mesh: {len(mesh.vertices)} vertices")
                else:
                    reconstructed.append({
                        'name': det['class_name'],
                        'mesh': None,
                        'pcd': pcd,
                        'has_mesh': False,
                        'pts_file': pts_file,
                        'detection': det
                    })
            else:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                reconstructed.append({
                    'name': det['class_name'],
                    'mesh': None,
                    'pcd': pcd,
                    'has_mesh': False,
                    'pts_file': pts_file,
                    'detection': det
                })
        
        # Summary
        print("\n" + "=" * 70)
        print(f"✅ COMPLETE: {len(reconstructed)} objects")
        print("=" * 70)
        print(f"\n📁 Outputs in: {output_dir}/")
        
        # STEP 4: Safety Distance Analysis
        print("\n" + "=" * 70)
        print("STEP 4: SAFETY DISTANCE ANALYSIS (3D Euclidean Distance)")
        print("=" * 70)
        print("\nMETHOD: Using 3D Euclidean distance formula:")
        print("   distance = √[(x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²]")
        print("=" * 70)
        
        # Find person
        person_obj = None
        for obj in reconstructed:
            if obj['name'] == 'person' and 'detection' in obj:
                person_obj = obj
                break
        
        if person_obj:
            person_points = person_obj['detection']['points']
            
            # Calculate person's 3D center and closest point
            person_center = np.mean(person_points, axis=0)  # [x, y, z]
            person_x, person_y, person_z = person_center
            
            # Find closest point to origin
            distances_from_origin = np.linalg.norm(person_points, axis=1)
            closest_idx = np.argmin(distances_from_origin)
            person_closest = person_points[closest_idx]
            
            print(f"\n👤 PERSON:")
            print(f"   Center Point (x, y, z): ({person_x:.2f}, {person_y:.2f}, {person_z:.2f})m")
            print(f"   Closest Point to Camera: ({person_closest[0]:.2f}, {person_closest[1]:.2f}, {person_closest[2]:.2f})m")
            print(f"   Number of points: {len(person_points)}")
            
            warnings = []
            safe_objects = []
            
            for obj in reconstructed:
                if obj['name'] != 'person' and 'detection' in obj:
                    obj_points = obj['detection']['points']
                    
                    # Calculate object's 3D center
                    obj_center = np.mean(obj_points, axis=0)
                    obj_x, obj_y, obj_z = obj_center
                    
                    # Calculate 3D Euclidean distance between centers
                    avg_distance = np.sqrt(
                        (obj_x - person_x)**2 + 
                        (obj_y - person_y)**2 + 
                        (obj_z - person_z)**2
                    )
                    
                    # Calculate minimum distance (closest points between objects)
                    # For each person point, find distance to all object points
                    min_distance = float('inf')
                    for p_point in person_points[::10]:  # Sample every 10th point for efficiency
                        distances = np.linalg.norm(obj_points - p_point, axis=1)
                        min_dist = np.min(distances)
                        if min_dist < min_distance:
                            min_distance = min_dist
                    
                    obj_info = {
                        'name': obj['name'],
                        'avg_distance': avg_distance,
                        'min_distance': min_distance,
                        'obj_center': obj_center,
                        'obj_x': obj_x,
                        'obj_y': obj_y,
                        'obj_z': obj_z,
                        'person_center': person_center
                    }
                    
                    if min_distance < SAFETY_THRESHOLD_METERS:
                        warnings.append(obj_info)
                    else:
                        safe_objects.append(obj_info)
            
            # Display warnings
            if warnings:
                print(f"\n⚠️  WARNING: {len(warnings)} object(s) within {SAFETY_THRESHOLD_METERS}m safety zone!")
                print("=" * 70)
                for idx, w in enumerate(sorted(warnings, key=lambda x: x['min_distance']), 1):
                    print(f"\n   🚨 OBJECT #{idx}: {w['name'].upper()}")
                    print(f"      Object Center (x, y, z): ({w['obj_x']:.2f}, {w['obj_y']:.2f}, {w['obj_z']:.2f})m")
                    print(f"      Person Center (x, y, z): ({w['person_center'][0]:.2f}, {w['person_center'][1]:.2f}, {w['person_center'][2]:.2f})m")
                    print(f"      ")
                    print(f"      Distance Calculation:")
                    dx = w['obj_x'] - w['person_center'][0]
                    dy = w['obj_y'] - w['person_center'][1]
                    dz = w['obj_z'] - w['person_center'][2]
                    print(f"         Δx = {w['obj_x']:.2f} - {w['person_center'][0]:.2f} = {dx:.2f}m")
                    print(f"         Δy = {w['obj_y']:.2f} - {w['person_center'][1]:.2f} = {dy:.2f}m")
                    print(f"         Δz = {w['obj_z']:.2f} - {w['person_center'][2]:.2f} = {dz:.2f}m")
                    print(f"      ")
                    print(f"      ➜ Center-to-Center Distance: √({dx:.2f}² + {dy:.2f}² + {dz:.2f}²) = {w['avg_distance']:.2f}m")
                    print(f"      ➜ Minimum Distance (closest points): {w['min_distance']:.2f}m ⚠️ UNSAFE!")
            else:
                print("\n✅ SAFE: All objects are at safe distance")
            
            # Display safe objects
            if safe_objects:
                print(f"\n" + "=" * 70)
                print(f"✅ {len(safe_objects)} SAFE object(s) (distance ≥ {SAFETY_THRESHOLD_METERS}m):")
                print("=" * 70)
                for s in sorted(safe_objects, key=lambda x: x['min_distance']):
                    print(f"   • {s['name']}: Min={s['min_distance']:.2f}m, Avg={s['avg_distance']:.2f}m")
                    print(f"     Position (x,y,z): ({s['obj_x']:.2f}, {s['obj_y']:.2f}, {s['obj_z']:.2f})m")
        else:
            print("\n[INFO] No person detected - skipping safety analysis")
        
        # Interactive viewer
        if reconstructed:
            while True:
                print("\n🎮 View options:")
                for i, obj in enumerate(reconstructed):
                    print(f"   {i+1}. {obj['name']}" + (" (mesh)" if obj['has_mesh'] else " (cloud)"))
                print(f"   {len(reconstructed)+1}. All together")
                print(f"   0. Exit")
                
                try:
                    choice = int(input("\nChoice: "))
                    if choice == 0 or choice == '0':
                        break
                    elif 1 <= choice <= len(reconstructed):
                        obj = reconstructed[choice-1]
                        if obj['has_mesh']:
                            visualize_mesh(obj['mesh'], f"{obj['name']} - Mesh")
                        else:
                            pcd = o3d.io.read_point_cloud(obj['pts_file'])
                            visualize_mesh(pcd, f"{obj['name']} - Point Cloud")
                    elif choice == len(reconstructed) + 1:
                        combined = o3d.geometry.TriangleMesh()
                        for obj in reconstructed:
                            if obj['has_mesh']:
                                combined += obj['mesh']
                        if len(combined.vertices) > 0:
                            visualize_mesh(combined, "All Objects")
                except:
                    break
    
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")


def main():
    print("=" * 70)
    print("🚀 2D to 3D Reconstruction Pipeline")
    print("=" * 70)
    
    # Select images
    print("\n📷 IMAGE SOURCE:")
    print("   1. Upload single image")
    print("   2. Process directory")
    print("   3. Use current directory")
    
    print("\nChoice (1-3): ", end="")
    try:
        option = int(input().strip())
    except:
        option = 3
    
    images = []
    
    if option == 1:
        print("\nImage path: ", end="")
        path = input().strip().strip('"\'')
        if os.path.exists(path) and any(path.lower().endswith(e.lower()) for e in IMAGE_EXTENSIONS):
            images = [path]
        else:
            print("[ERROR] Invalid file")
            return
    elif option == 2:
        print("\nDirectory: ", end="")
        directory = input().strip().strip('"\'')
        images = find_available_images(directory)
        if not images:
            print("[ERROR] No images found")
            return
        print(f"✓ Found {len(images)} image(s)")
    else:
        images = find_available_images('.')
        if not images:
            print("[ERROR] No images in current directory")
            return
        print(f"✓ Found {len(images)} image(s)")
    
    # Setup models
    print("\n[INFO] Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO("yolov8n.pt")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas.to(device)
    midas.eval()
    midas_t = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_t.dpt_transform
    
    # Process
    for idx, img_path in enumerate(images):
        print(f"\n{'='*70}")
        print(f"IMAGE {idx+1}/{len(images)}: {os.path.basename(img_path)}")
        print(f"{'='*70}")
        process_single_image(img_path, device, yolo, midas, transform)
    
    print(f"\n{'='*70}")
    print(f"✅ ALL DONE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
