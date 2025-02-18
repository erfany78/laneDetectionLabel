import cv2
import os
import numpy as np
from src.data_loader import load_label, save_label

# Constants
NUM_POINTS = 9
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "../dataset_xy")
ROAD_TYPES = ["Regular Road", "Crossroad"]
OFFSET = 66  # Offset of 66 pixels for both sides

# Global variables
points = []
road_type = 0
current_image = None
image_path = None
image_files = []
current_idx = 0

def draw_ui(image):
    """Draws road type text, labeled points, and three horizontal lines on the image."""
    height, width, _ = image.shape

    # Create a black background larger than the original image
    black_background = np.zeros((height + 2 * OFFSET, width + 2 * OFFSET, 3), dtype=np.uint8)

    # Place the original image at the center of the black background
    black_background[OFFSET:OFFSET+height, OFFSET:OFFSET+width] = image

    # Add road type text on the black background
    cv2.putText(black_background, f"Type: {ROAD_TYPES[road_type]}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Draw three horizontal lines to help position the points
    # Line 1: at 25% height of the image
    line_y1 = int(height * 0.45) + OFFSET
    cv2.line(black_background, (0, line_y1), (width + OFFSET, line_y1), (0, 255, 255), 2)  # Yellow

    # Line 2: at 50% height of the image (middle line)
    line_y2 = int(height * 0.65) + OFFSET
    cv2.line(black_background, (0, line_y2), (width + OFFSET, line_y2), (255, 255, 0), 2)  # Cyan

    # Line 3: at 75% height of the image
    line_y3 = int(height * 0.90 ) + OFFSET
    cv2.line(black_background, (0, line_y3), (width + OFFSET, line_y3), (0, 255, 0), 2)  # Green

    # Loop through the points and check if they are inside the visible area of the image
    for index, (x, y, _) in enumerate(points):  # Add index to track position
        if x != 0 and y != 0:
            # Apply colors based on index
            if index < 3:
                color = (255, 0, 0)  # Blue for first 3 points
            elif index < 6:
                color = (0, 255, 0)  # Green for next 3 points
            else:
                color = (0, 0, 255)  # Red for last 3 points

            # Map normalized points (-1 to 1) to pixel values for image coordinates
            x_offset = int((x + 1) * (width / 2))  # Convert to image coordinates and apply offset
            y_offset = int((y + 1) * (height / 2))

            # Check if the point is within the visible bounds of the original image
            if OFFSET <= x_offset < width + OFFSET and OFFSET <= y_offset < height + OFFSET:
                # If visible, fill the point
                cv2.circle(black_background, (x_offset, y_offset), 5, color, -1)
            else:
                # If invisible, outline the point
                cv2.circle(black_background, (x_offset, y_offset), 5, color, 2)

    return black_background

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
