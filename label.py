import cv2
import os
import numpy as np

# Constants
NUM_POINTS = 6
LABELS_DIR = "labeled_data"
IMAGE_DIR = "dataset_xy"
ROAD_TYPES = ["Regular Road", "Crossroad"]

# Ensure output directory exists
if not os.path.exists(LABELS_DIR):
    os.makedirs(LABELS_DIR)

# Global variables
points = []
road_type = 0  # 0 = Regular Road, 1 = Crossroad
current_image = None
image_path = None
image_files = []
current_idx = 0

def load_label(image_name):
    """Loads saved points and road type if available."""
    global points, road_type
    label_path = os.path.join(LABELS_DIR, image_name.replace(".jpg", ".txt"))

    if os.path.exists(label_path):
        try:
            data = np.loadtxt(label_path, delimiter=',')

            # Ensure data is not empty or incorrectly formatted
            if data.ndim == 0 or data.size == 0:
                print(f"Empty label file found for {image_name}. Resetting labels.")
                points = []
                road_type = 0
                return

            # Extract points and road type
            points_data = data[:-1] if data.size > 1 else []  # Ensure points exist
            road_type = int(data[-1]) if data.size > 1 else 0  # Extract road type

            if len(points_data) % 3 == 0:
                points = np.array(points_data).reshape(-1, 3).tolist()
            else:
                print(f"Warning: Label file {label_path} has incorrect format. Resetting labels.")
                points = []

            print(f"Loaded labels for {image_name}: {len(points)} points, Road Type: {ROAD_TYPES[road_type]}")

        except Exception as e:
            print(f"Error loading labels for {image_name}: {e}")
            points = []
            road_type = 0
    else:
        points = []
        road_type = 0

def draw_ui(image):
    """Draw road type text and labeled points on the image."""
    global points, road_type

    display_image = image.copy()
    cv2.putText(display_image, f"Road Type: {ROAD_TYPES[road_type]}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw labeled lane points
    height, width, _ = display_image.shape
    for point in points:
        x, y, _ = point
        if x != 0 and y != 0:  # Ignore padding points
            cv2.circle(display_image,
                       (int(x * width / 2 + width / 2), int(y * height / 2 + height / 2)),
                       5, (0, 0, 255), -1)

    return display_image

def save_label():
    """Saves the points and road type to a text file."""
    global points, road_type, image_path
    filename = os.path.basename(image_path).replace(".jpg", ".txt")
    label_path = os.path.join(LABELS_DIR, filename)

    # Save as flattened points and road type at the end
    points_flattened = np.array(points).flatten()
    label_data = np.hstack((points_flattened, road_type))

    np.savetxt(label_path, label_data, fmt='%f', delimiter=',')
    print(f"Labels saved to {label_path}")

def click_event(event, x, y, flags, param):
    """Handles mouse clicks for adding and removing points."""
    global points, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        # Normalize and store points
        height, width, _ = current_image.shape
        x_norm = (x - width / 2) / (width / 2)
        y_norm = (y - height / 2) / (height / 2)

        if len(points) < NUM_POINTS:
            points.append([x_norm, y_norm, 1])  # Last element '1' means visible
            print(f"Point {len(points)}: ({x_norm:.2f}, {y_norm:.2f})")

        cv2.imshow("Labeling", draw_ui(current_image))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 0:
            points.pop()
            print(f"Removed last point, {len(points)} points remaining.")

        cv2.imshow("Labeling", draw_ui(current_image))

def switch_road_type():
    """Switches between road types (Regular Road <=> Crossroad)."""
    global road_type
    road_type = (road_type + 1) % 2
    print(f"Road type switched to: {ROAD_TYPES[road_type]}")
    cv2.imshow("Labeling", draw_ui(current_image))

def label_image(image_path):
    """Loads and labels the given image."""
    global current_image, points, road_type, current_idx

    # Load image and existing labels
    image = cv2.imread(image_path)
    current_image = image.copy()
    load_label(os.path.basename(image_path))

    cv2.imshow("Labeling", draw_ui(current_image))
    cv2.setMouseCallback("Labeling", click_event)

    print("\nInstructions:")
    print("- Left-click to add a point.")
    print("- Right-click to remove the last point.")
    print("- Press 'W' to switch road type.")
    print("- Press 'D' for next image, 'A' for previous image.")
    print("- Press 'Q' to save and exit.")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("d"):  # Move to next image
            print("Moving to next image...")
            save_label()
            return False

        elif key == ord("a"):  # Move to previous image
            print("Going back to previous image...")
            save_label()
            return True

        elif key == ord("w"):  # Switch road type
            switch_road_type()

        elif key == ord("q"):  # Save and exit
            break

    save_label()
    return False

def main():
    """Main function to label all images."""
    global image_path, current_idx, image_files

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
    print(f"Found {len(image_files)} images to label.")

    while True:
        image_path = os.path.join(IMAGE_DIR, image_files[current_idx])
        print(f"\nLabeling {image_files[current_idx]}...")

        if not label_image(image_path):  # Move to next image
            current_idx = (current_idx + 1) % len(image_files)
        else:  # Go back to the previous image
            current_idx = (current_idx - 1) % len(image_files)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
