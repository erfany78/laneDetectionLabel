import cv2
import os
import numpy as np
from src.data_loader import load_label, save_label

# Constants
NUM_POINTS = 10
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "../dataset_xy")
ROAD_TYPES = ["Regular Road", "Crossroad"]

# Global variables
points = []
road_type = 0
current_image = None
image_path = None
image_files = []
current_idx = 0

def draw_ui(image):
    """Draws road type text and labeled points on the image."""
    display_image = image.copy()
    cv2.putText(display_image, f"Road Type: {ROAD_TYPES[road_type]}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    height, width, _ = display_image.shape
    for x, y, _ in points:
        if x != 0 and y != 0:
            cv2.circle(display_image, (int(x * width / 2 + width / 2), int(y * height / 2 + height / 2)),
                       5, (0, 0, 255), -1)
    return display_image

def click_event(event, x, y, flags, param):
    """Handles mouse clicks for adding/removing lane points."""
    global points, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        height, width, _ = current_image.shape
        x_norm = (x - width / 2) / (width / 2)
        y_norm = (y - height / 2) / (height / 2)

        if len(points) < NUM_POINTS:
            points.append([x_norm, y_norm, 1])
            print(f"Point {len(points)} added: ({x_norm:.2f}, {y_norm:.2f})")

        cv2.imshow("Labeling", draw_ui(current_image))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            print(f"Removed last point, {len(points)} remaining.")
        cv2.imshow("Labeling", draw_ui(current_image))

def switch_road_type():
    """Toggles the road type between Regular and Crossroad."""
    global road_type
    road_type = (road_type + 1) % 2
    print(f"Switched road type to: {ROAD_TYPES[road_type]}")
    cv2.imshow("Labeling", draw_ui(current_image))

def label_image(image_path):
    """Loads and labels an image."""
    global current_image, points, road_type, current_idx
    image = cv2.imread(image_path)
    current_image = image.copy()
    points, road_type = load_label(os.path.basename(image_path))

    cv2.imshow("Labeling", draw_ui(current_image))
    cv2.setMouseCallback("Labeling", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("d"):
            save_label(os.path.basename(image_path), points, road_type)
            return False
        elif key == ord("a"):
            save_label(os.path.basename(image_path), points, road_type)
            return True
        elif key == ord("w"):
            switch_road_type()
        elif key == ord("q"):
            save_label(os.path.basename(image_path), points, road_type)
            break

def main():
    """Main function to label images."""
    global image_path, current_idx, image_files
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]

    while True:
        image_path = os.path.join(IMAGE_DIR, image_files[current_idx])
        if not label_image(image_path):
            current_idx = (current_idx + 1) % len(image_files)
        else:
            current_idx = (current_idx - 1) % len(image_files)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
