import os
import numpy as np

LABELS_DIR = os.path.join(os.path.dirname(__file__), "../labeled_data")

def ensure_directory_exists(directory):
    """Ensures the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_label(image_name):
    """Loads saved lane points and road type if available."""
    label_path = os.path.join(LABELS_DIR, image_name.replace(".jpg", ".txt"))

    if os.path.exists(label_path):
        try:
            data = np.loadtxt(label_path, delimiter=',')
            if data.ndim == 0 or data.size == 0:
                return [], 0  # Empty label file

            points = data[:-1] if data.size > 1 else []  # Extract lane points
            road_type = int(data[-1]) if data.size > 1 else 0  # Extract road type

            if len(points) % 3 == 0:
                return np.array(points).reshape(-1, 3).tolist(), road_type
            else:
                print(f"Warning: {label_path} has incorrect format. Resetting labels.")
                return [], 0

        except Exception as e:
            print(f"Error loading {image_name}: {e}")
            return [], 0
    return [], 0

def save_label(image_name, points, road_type):
    """Saves labeled lane points and road type."""
    ensure_directory_exists(LABELS_DIR)
    label_path = os.path.join(LABELS_DIR, image_name.replace(".jpg", ".txt"))

    points_flattened = np.array(points).flatten()
    label_data = np.hstack((points_flattened, road_type))
    np.savetxt(label_path, label_data, fmt='%f', delimiter=',')
    print(f"Saved labels to {label_path}")
