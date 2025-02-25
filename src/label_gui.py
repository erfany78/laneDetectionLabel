import cv2
import os
import numpy as np
from src.data_loader import load_label, save_label

# Constants
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "../dataset")
ROAD_TYPES = ["Regular Road", "RIGHT OUT OF ROAD", "LEFT OUT OF ROAD", "Crossroad", "T-Junction"]
MAX_POINTS = 5

# Global variables
points = []  # Will store all points added by the user
road_type = 0
current_image = None
image_path = None
image_files = []
current_idx = 0


def bezier_curve(p0, p1, p2, t):
    """Calculate the quadratic Bezier curve point for parameter t."""
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    return (x, y)


def draw_ui(image):
    """Draws road type text, labeled points, and arcs between points on the image."""
    height, width, _ = image.shape

    # Create a black background the same size as the original image
    black_background = np.zeros((height, width, 3), dtype=np.uint8)

    # Place the original image on the black background
    black_background[:height, :width] = image

    # Add road type text on the black background
    cv2.putText(black_background, f"Type: {ROAD_TYPES[road_type]}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Draw arcs between each pair of points (Bezier curve)
    for i in range(len(points) - 1):
        p0 = points[i][:2]  # Start point (x1, y1)
        p2 = points[i + 1][:2]  # End point (x2, y2)

        # Midpoint for the curve (control point)
        p1 = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2)  # Midpoint between the two points

        # Generate the points along the Bezier curve (arc)
        if i == 0:
            curve_points = []
            for t in np.linspace(0, 1, 5):  # Generate 5 points along the curve
                curve_points.append(bezier_curve(p0, p1, p2, t))

            # Convert curve points to image coordinates and draw them
            for i, point in enumerate(curve_points):
                x_offset = int((point[0] + 1) * (width / 2))
                y_offset = int((point[1] + 1) * (height / 2))

                if 0 <= x_offset < width and 0 <= y_offset < height:
                    # If visible, fill the point
                    if len(points) == MAX_POINTS - i and len(points) < MAX_POINTS:
                        cv2.line(black_background, (0, y_offset), (width, y_offset), (255, 255, 0))
                else:
                    # If invisible, outline the point
                    cv2.circle(black_background, (x_offset, y_offset), 5, (255, 255, 0), 2)

    # Loop through the points and check if they are inside the visible area of the image
    for index, (x, y, _) in enumerate(points):  # Add index to track position
        if x != 0 and y != 0:
            # Apply colors based on index
            color = (255, 0, 0) if index < 2 else (0, 255, 0)  # Blue for first point, Green for subsequent points

            # Map normalized points (-1 to 1) to pixel values for image coordinates
            x_offset = int((x + 1) * (width / 2))  # Convert to image coordinates
            y_offset = int((y + 1) * (height / 2))

            # Check if the point is within the visible bounds of the original image
            if 0 <= x_offset < width and 0 <= y_offset < height:
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
        # Check if the number of points is less than 5 before adding a new point
        if len(points) < MAX_POINTS:
            height, width, _ = current_image.shape
            x_norm = (x - width / 2) / (width / 2)
            y_norm = (y - height / 2) / (height / 2)

            # Add the clicked point to the list
            points.append([x_norm, y_norm, 1])
            print(f"Point {len(points)} added: ({x_norm:.2f}, {y_norm:.2f})")

            cv2.imshow("Labeling", draw_ui(current_image))
        else:
            print("Maximum number of points (5) reached.")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()  # Remove the last point
            print(f"Removed last point, {len(points)} remaining.")
        cv2.imshow("Labeling", draw_ui(current_image))


def switch_road_type(upper=True):
    """Toggles the road type between Regular and Crossroad."""
    global road_type
    if upper:
        road_type = (road_type + 1) % len(ROAD_TYPES)
    else:
        road_type = (road_type - 1) % len(ROAD_TYPES)

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
        elif key == ord("s"):
            switch_road_type(False)
        elif key == ord("q"):
            save_label(os.path.basename(image_path), points, road_type)
            break


def main():
    """Main function to label images."""
    global image_path, current_idx, image_files
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]

    while True:
        image_path = os.path.join(IMAGE_DIR, image_files[current_idx])
        if not label_image(image_path):
            current_idx = (current_idx + 1) % len(image_files)
        else:
            current_idx = (current_idx - 1) % len(image_files)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
