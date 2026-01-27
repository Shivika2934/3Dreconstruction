"""
Gradio Dashboard Configuration File
====================================

Customize the behavior of the 3D reconstruction dashboard by editing this file.
Then import it in gradio_dashboard.py
"""

# =========================
# SAFETY THRESHOLDS (meters)
# =========================

# Minimum safe distance for different object types
# If actual distance falls below these values, a warning is triggered
SAFETY_THRESHOLDS = {
    'person': 0.5,           # 50cm minimum for people proximity
    'chair': 0.7,            # 70cm for chairs
    'table': 1.0,            # 1m for tables
    'couch': 0.8,            # 80cm for couches
    'bed': 1.0,              # 1m for beds
    'default': 1.0           # 1m for anything else
}

# =========================
# RECONSTRUCTION PARAMETERS
# =========================

# Poisson surface reconstruction depth
# Higher values = better quality but slower processing
# Range: 6-12 (default: 8)
POISSON_DEPTH = 8

# Number of iterations for mesh smoothing
# Higher = smoother but slower
MESH_SMOOTHING_ITERATIONS = 10

# Statistical outlier removal
# Number of neighbors to consider when removing noise
OUTLIER_REMOVAL_NEIGHBORS = 50

# Density threshold for filtering low-density mesh regions
# Range: 5-25 (percentage of points to consider as outliers)
DENSITY_THRESHOLD_PERCENTILE = 15

# =========================
# OBJECT DETECTION (YOLO)
# =========================

# YOLO confidence threshold (0.0-1.0)
# Lower = more detections but more false positives
YOLO_CONFIDENCE = 0.3

# =========================
# POINT CLOUD SAMPLING
# =========================

# Number of points to sample from each detected object
# Higher = more detail but slower processing
# Range: 1000-50000
N_SAMPLES_RECONSTRUCTION = 10000

# Number of points to sample for distance calculation
# Lower = faster but less accurate distance measurements
# Range: 100-2000
N_SAMPLES_DISTANCE = 500

# =========================
# CAMERA INTRINSICS
# =========================

# Default focal length (in pixels)
# Used for back-projection from 2D to 3D
DEFAULT_FX_FY = 1000.0

# =========================
# VISUALIZATION OPTIONS
# =========================

# Show/hide different visualizations
SHOW_DEPTH_MAP = True
SHOW_DETECTION_OVERLAY = True
SHOW_DISTANCE_TABLE = True
SHOW_SAFETY_WARNINGS = True

# Point cloud visualization settings
POINT_CLOUD_POINT_SIZE = 2
POINT_CLOUD_BACKGROUND = (255, 255, 255)

# =========================
# PERFORMANCE TUNING
# =========================

# Maximum points to keep per object (for performance)
# Set to None for unlimited
MAX_POINTS_PER_OBJECT = 50000

# Use GPU acceleration
USE_GPU = True

# =========================
# GRADIO INTERFACE SETTINGS
# =========================

# Server configuration
SERVER_NAME = "0.0.0.0"
SERVER_PORT = 7860
SHARE = False  # Share link (works with ngrok)
SHOW_ERROR = True

# =========================
# OBJECT COLORS (BGR format)
# =========================

OBJECT_COLORS = {
    'person': (0, 255, 0),           # Green
    'chair': (255, 0, 0),            # Blue
    'dining table': (0, 165, 255),   # Orange
    'table': (0, 165, 255),          # Orange
    'couch': (255, 128, 0),          # Light Blue
    'bed': (128, 0, 255),            # Magenta
    'tv': (255, 255, 0),             # Cyan
    'laptop': (200, 200, 200),       # Gray
    'default': (255, 255, 0)         # Yellow
}

# =========================
# PROCESSING SETTINGS
# =========================

# Save intermediate outputs
SAVE_OUTPUTS = False
OUTPUT_DIR = "gradio_outputs"

# Enable verbose logging
VERBOSE = True

# =========================
# ADVANCED OPTIONS
# =========================

# Use MiDaS depth model variant
# Options: 'small' (fast), 'base' (balanced), 'large' (accurate)
MIDAS_MODEL = 'small'

# Image preprocessing
AUTO_BRIGHTNESS = True
AUTO_CONTRAST = True

# Mesh simplification (for faster visualization)
# Set to None to disable, value between 0.0-1.0
MESH_SIMPLIFICATION = None  # 0.5 = 50% simplification

# =========================
# FUNCTION TEMPLATES
# =========================

def get_safety_threshold(object_class: str) -> float:
    """
    Get safety threshold for a specific object class.
    
    Args:
        object_class: Name of the object class
        
    Returns:
        Safety threshold in meters
    """
    return SAFETY_THRESHOLDS.get(object_class, SAFETY_THRESHOLDS['default'])


def get_object_color(object_class: str) -> tuple:
    """
    Get display color for a specific object class.
    
    Args:
        object_class: Name of the object class
        
    Returns:
        Color as (B, G, R) tuple
    """
    return OBJECT_COLORS.get(object_class, OBJECT_COLORS['default'])


# =========================
# PRESET CONFIGURATIONS
# =========================

PRESETS = {
    'quality': {
        'description': 'High quality reconstruction (slower)',
        'POISSON_DEPTH': 10,
        'OUTLIER_REMOVAL_NEIGHBORS': 100,
        'N_SAMPLES_RECONSTRUCTION': 50000,
        'MESH_SMOOTHING_ITERATIONS': 15,
    },
    'balanced': {
        'description': 'Balanced quality and speed (recommended)',
        'POISSON_DEPTH': 8,
        'OUTLIER_REMOVAL_NEIGHBORS': 50,
        'N_SAMPLES_RECONSTRUCTION': 10000,
        'MESH_SMOOTHING_ITERATIONS': 10,
    },
    'fast': {
        'description': 'Fast processing (lower quality)',
        'POISSON_DEPTH': 6,
        'OUTLIER_REMOVAL_NEIGHBORS': 30,
        'N_SAMPLES_RECONSTRUCTION': 5000,
        'MESH_SMOOTHING_ITERATIONS': 5,
    },
    'realtime': {
        'description': 'Real-time processing (minimal quality)',
        'POISSON_DEPTH': 5,
        'OUTLIER_REMOVAL_NEIGHBORS': 20,
        'N_SAMPLES_RECONSTRUCTION': 2000,
        'MESH_SMOOTHING_ITERATIONS': 2,
    }
}


def apply_preset(preset_name: str) -> dict:
    """
    Apply a preset configuration.
    
    Args:
        preset_name: Name of the preset ('quality', 'balanced', 'fast', 'realtime')
        
    Returns:
        Dictionary with preset settings
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    return PRESETS[preset_name]


# =========================
# USAGE EXAMPLES
# =========================

if __name__ == "__main__":
    """
    Example usage of configuration functions
    """
    
    # Get safety threshold for a chair
    chair_threshold = get_safety_threshold('chair')
    print(f"Safety threshold for chair: {chair_threshold}m")
    
    # Get color for a person
    person_color = get_object_color('person')
    print(f"Person color (BGR): {person_color}")
    
    # Apply a preset
    fast_preset = apply_preset('fast')
    print(f"Fast preset: {fast_preset}")
    
    # View all thresholds
    print("\nAll safety thresholds:")
    for obj_class, threshold in SAFETY_THRESHOLDS.items():
        print(f"  {obj_class}: {threshold}m")
