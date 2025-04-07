#!/usr/bin/env python3
"""
GUI application for labeling road images with points and road type.
The application allows users to place points on road images and classify
the road type for lane detection purposes.
"""

import os
import sys
import cv2
import numpy as np
from typing import List, Tuple, Optional, Any
from data_loader import load_label, save_label


class LabelGUI:
    """
    A GUI application for labeling road images with points and road types.
    
    This class provides a graphical interface to place points on road images
    and classify the road type. It supports navigation between images and
    saving of labels.
    """
    
    # Constants
    IMAGE_DIR = os.path.join(os.path.dirname(__file__), "../dataset")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output")
    ROAD_TYPES = ["Regular Road", "RIGHT OUT OF ROAD", "LEFT OUT OF ROAD", 
                  "OUT", "Crossroad", "T-Junction"]
    MAX_POINTS = 5  # Maximum points per lane
    MAX_LANES = 2   # Maximum number of lanes
    LANE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # Different color for each lane
    OFFSET_WIDTH = 320  # Width of black offset area on each side for invisible points
    INPUT_SIZE = (640, 640)  # Width, Height for display/working image
    OUTPUT_SIZE = (224, 224)  # Width, Height for output image

    def __init__(self) -> None:
        """Initialize the LabelGUI application with default values."""
        # Instance variables
        self.points: List[List[List[int]]] = [[] for _ in range(self.MAX_LANES)]  # List of lanes, each containing points
        self.current_lane = 0  # Currently active lane index
        self.road_type: int = 0
        self.current_image: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        self.image_files: List[str] = []
        self.current_idx: int = 0
        self.original_image_size: Tuple[int, int] = (0, 0)  # Store original image size before resizing
        
        # Ensure output directory exists
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    @staticmethod
    def calculate_angle_distance(reference_point: Tuple[int, int], point: Tuple[int, int]) -> Tuple[float, float]:
        """
        Calculate angle and distance from reference point to target point.
        
        Args:
            reference_point: (x, y) coordinates of reference point
            point: (x, y) coordinates of target point
            
        Returns:
            Tuple containing (angle in degrees, distance in pixels)
        """
        # Calculate distance using Euclidean distance
        dx = point[0] - reference_point[0]
        dy = point[1] - reference_point[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate angle (in degrees, 0 is right, increases counterclockwise)
        # arctan2 returns angle in radians, convert to degrees
        angle = np.degrees(np.arctan2(-dy, dx))  # -dy because y increases downward in image coordinates
        
        # Normalize angle to [0, 360)
        angle = (angle + 360) % 360
        
        return angle, distance

    @staticmethod
    def bezier_curve(p0: Tuple[int, int], p1: Tuple[float, float], 
                     p2: Tuple[int, int], t: float) -> Tuple[int, int]:
        """
        Calculate the quadratic Bezier curve point for parameter t.
        
        Args:
            p0: Start point (x, y)
            p1: Control point (x, y)
            p2: End point (x, y)
            t: Parameter value between 0 and 1
            
        Returns:
            Tuple containing (x, y) coordinates of the point on the curve
        """
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        return (int(x), int(y))

    def draw_ui(self, image: np.ndarray, current_idx: int, total_images: int) -> np.ndarray:
        """
        Draw UI elements on the image including road type text, points, and arcs.
        
        Args:
            image: The original image to draw on
            current_idx: Current image index
            total_images: Total number of images
            
        Returns:
            Image with UI elements drawn on it
        """
        if image is None:
            print("Error: Cannot draw UI on None image")
            return np.zeros((600, 800, 3), dtype=np.uint8)
            
        height, width, _ = image.shape
        
        # Create a black background the same size as the original image plus offset areas
        black_background = np.zeros((height, width + 2 * self.OFFSET_WIDTH, 3), dtype=np.uint8)

        # Place the original image on the black background (centered)
        black_background[:height, self.OFFSET_WIDTH:self.OFFSET_WIDTH + width] = image

        # Add header with black background (full width, 50px height)
        cv2.rectangle(black_background, (0, 0), (width + 2 * self.OFFSET_WIDTH, 50), (0, 0, 0), -1)
        
        # Add road type text
        cv2.putText(black_background, f"Type: {self.ROAD_TYPES[self.road_type]}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Add image counter text
        cv2.putText(black_background, f"Image {current_idx + 1}/{total_images}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Add resolution information
        cv2.putText(black_background, f"Input: {self.INPUT_SIZE[0]}x{self.INPUT_SIZE[1]}, Output: {self.OUTPUT_SIZE[0]}x{self.OUTPUT_SIZE[1]}", 
                   (width + 2 * self.OFFSET_WIDTH - 450, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Draw lane selection info
        lane_text = f"Lane: {self.current_lane+1}/{self.MAX_LANES}"
        cv2.putText(black_background, lane_text, (width + 2 * self.OFFSET_WIDTH - 450, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.LANE_COLORS[self.current_lane], 1, cv2.LINE_AA)

        # Draw status in right corner - red if fewer than max points, green if max reached
        current_lane_points = len(self.points[self.current_lane])
        points_color = (0, 255, 0) if current_lane_points >= self.MAX_POINTS else (0, 0, 255)
        cv2.rectangle(black_background, (width + 2 * self.OFFSET_WIDTH - 150, 20), 
                      (width + 2 * self.OFFSET_WIDTH - 10, 50), points_color, 1)
        cv2.putText(black_background, f"Points: {current_lane_points}/{self.MAX_POINTS}", 
                   (width + 2 * self.OFFSET_WIDTH - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
       
        # Draw bezier curves between points for all lanes
        self._draw_curves(black_background, width + 2 * self.OFFSET_WIDTH, height)
        
        # Draw points for all lanes
        self._draw_points(black_background, width + 2 * self.OFFSET_WIDTH, height)

        return black_background
    
    def _draw_curves(self, image: np.ndarray, width: int, height: int) -> None:
        """
        Draw bezier curves between points.
        
        Args:
            image: Image to draw on
            width: Image width
            height: Image height
        """
        # Draw arcs for each lane
        for lane_idx, lane_points in enumerate(self.points):
            # Skip empty lanes
            if len(lane_points) < 2:  # Need at least 2 points to draw a curve
                continue
                
            # Get lane color
            lane_color = self.LANE_COLORS[lane_idx % len(self.LANE_COLORS)]
            
            # Draw arcs between each pair of points in the lane
            for i in range(len(lane_points) - 1):
                try:
                    # Ensure we're using integer coordinates and add offset
                    p0 = (int(lane_points[i][0]) + self.OFFSET_WIDTH, int(lane_points[i][1]))  # Start point
                    p2 = (int(lane_points[i + 1][0]) + self.OFFSET_WIDTH, int(lane_points[i + 1][1]))  # End point

                    # Midpoint for the curve (control point)
                    p1 = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2)

                    # Draw the curve for the current lane
                    # Active lane gets full curve, other lanes get dashed curve
                    if lane_idx == self.current_lane:
                        # For the active lane, draw full curve
                        curve_points = []
                        # for t in np.linspace(0, 1, 20):  # Generate 20 points for smooth curve
                        #     curve_points.append(self.bezier_curve(p0, p1, p2, t))
                        
                        # # Draw the curve
                        # for j in range(len(curve_points) - 1):
                        #     cv2.line(image, curve_points[j], curve_points[j + 1], lane_color, 2)

                        # Only draw the first curve with extra indicators
                        if i == 0:
                            curve_points = []
                            for t in np.linspace(0, 1, 5):  # Generate 5 points along the curve
                                curve_points.append(self.bezier_curve(p0, p1, p2, t))

                            # Draw the generated curve points
                            for idx, point in enumerate(curve_points):
                                x_offset, y_offset = point
                                
                                if 0 <= x_offset < width and 0 <= y_offset < height:
                                    # Draw horizontal guideline if appropriate
                                    if len(lane_points) == self.MAX_POINTS - idx and len(lane_points) < self.MAX_POINTS and self.road_type < 3:
                                        cv2.line(image, (0, y_offset), (width, y_offset), (255, 255, 0))
                                else:
                                    # If point is outside visible area, just outline it
                                    cv2.circle(image, (int(x_offset), int(y_offset)), 5, (255, 255, 0), 2)
                    else:
                        # For inactive lanes, draw dashed curve
                        curve_points = []
                        for t in np.linspace(0, 1, 10):  # Generate 10 points for dashed curve
                            curve_points.append(self.bezier_curve(p0, p1, p2, t))
                        
                        # Draw dashed curve (every other segment)
                        for j in range(0, len(curve_points) - 1, 2):
                            cv2.line(image, curve_points[j], curve_points[j + 1], lane_color, 1)
                except Exception as e:
                    print(f"Error drawing arc for lane {lane_idx}: {e}")
    
    def _draw_points(self, image: np.ndarray, width: int, height: int) -> None:
        """
        Draw all points on the image.
        
        Args:
            image: Image to draw on
            width: Image width (including offset areas)
            height: Image height
        """
        original_width = width - 2 * self.OFFSET_WIDTH
        
        # Loop through all lanes
        for lane_idx, lane_points in enumerate(self.points):
            # Get lane color
            lane_color = self.LANE_COLORS[lane_idx % len(self.LANE_COLORS)]
            
            # Loop through the points in this lane
            for index, point in enumerate(lane_points):
                try:
                    # Make sure we have proper integer coordinates with offset
                    x = int(point[0]) + self.OFFSET_WIDTH
                    y = int(point[1])
                    
                    if x != self.OFFSET_WIDTH or y != 0:  # Skip (0,0) points
                        # Check if the point is within the original image area
                        in_original_area = self.OFFSET_WIDTH <= x < self.OFFSET_WIDTH + original_width and 0 <= y < height
                        
                        # Determine point style
                        fill = -1 if lane_idx == self.current_lane else 1  # Filled circle for active lane
                        
                        if in_original_area:
                            # If visible in original image, draw the point
                            cv2.circle(image, (x, y), 5, lane_color, fill)
                            # Add point number for current lane
                            if lane_idx == self.current_lane:
                                cv2.putText(image, f"{index+1}", (x-3, y+3), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                        else:
                            # If in offset area, draw outlined point with connection line
                            cv2.circle(image, (x, y), 5, lane_color, 2)
                            
                            # Draw dashed connection line to edge of original image
                            if x < self.OFFSET_WIDTH:  # Point is in left offset
                                edge_x = self.OFFSET_WIDTH
                                for i in range(0, self.OFFSET_WIDTH - x, 10):
                                    cv2.line(image, (x + i, y), (x + i + 5, y), lane_color, 1)
                            elif x >= self.OFFSET_WIDTH + original_width:  # Point is in right offset
                                edge_x = self.OFFSET_WIDTH + original_width - 1
                                for i in range(0, x - edge_x, 10):
                                    cv2.line(image, (edge_x + i, y), (edge_x + i + 5, y), lane_color, 1)
                        
                        # Display angle and distance for non-reference points on active lane
                        if lane_idx == self.current_lane and index > 0 and len(point) >= 5:
                            angle = point[3]
                            distance = point[4]
                            text = f"{angle:.1f}, {distance:.0f}px"
                            cv2.putText(image, text, (x + 10, y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                except Exception as e:
                    print(f"Error drawing point {index} in lane {lane_idx}: {e}")

    def click_event(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """
        Handle mouse click events for adding/removing points.
        
        Args:
            event: The type of mouse event
            x: X coordinate of the event
            y: Y coordinate of the event
            flags: Additional flags
            param: Additional parameters
        """
        if self.current_image is None:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the number of points is less than max before adding a new point
            current_lane_points = self.points[self.current_lane]
            if len(current_lane_points) < self.MAX_POINTS:
                # Adjust x coordinate to account for offset
                adjusted_x = x - self.OFFSET_WIDTH
                
                # Set visibility flag (1 if in original image area, 0 if in offset area)
                original_width = self.current_image.shape[1]
                visibility = 1 if 0 <= adjusted_x < original_width and 0 <= y < self.current_image.shape[0] else 0
                
                # Store actual pixel coordinates with visibility flag
                new_point = [adjusted_x, int(y), visibility]
                
                # Calculate angle and distance if this isn't the first point
                if len(current_lane_points) > 0:
                    reference_point = (current_lane_points[0][0], current_lane_points[0][1])
                    angle, distance = self.calculate_angle_distance(reference_point, (adjusted_x, y))
                    new_point.extend([angle, distance])
                    print(f"Lane {self.current_lane+1}, Point {len(current_lane_points)+1} added: ({adjusted_x}, {y}) - Angle: {angle:.2f}°, Distance: {distance:.2f}px")
                else:
                    # First point is the reference, angle and distance are 0
                    new_point.extend([0.0, 0.0])
                    print(f"Lane {self.current_lane+1}, Reference point added: ({adjusted_x}, {y})")
                
                self.points[self.current_lane].append(new_point)
                cv2.imshow("Labeling", self.draw_ui(self.current_image, 
                                                   self.current_idx, 
                                                   len(self.image_files)))
            else:
                print(f"Maximum number of points ({self.MAX_POINTS}) reached for lane {self.current_lane+1}.")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points[self.current_lane]:
                self.points[self.current_lane].pop()  # Remove the last point from current lane
                print(f"Removed last point from lane {self.current_lane+1}, {len(self.points[self.current_lane])} remaining.")
            cv2.imshow("Labeling", self.draw_ui(self.current_image, 
                                               self.current_idx, 
                                               len(self.image_files)))

    def switch_lane(self, increase: bool = True) -> None:
        """
        Switch between lanes.
        
        Args:
            increase: If True, increment lane, otherwise decrement
        """
        if increase:
            self.current_lane = (self.current_lane + 1) % self.MAX_LANES
        else:
            self.current_lane = (self.current_lane - 1) % self.MAX_LANES

        print(f"Switched to lane {self.current_lane+1}")
        
        if self.current_image is not None:
            cv2.imshow("Labeling", self.draw_ui(self.current_image, 
                                               self.current_idx, 
                                               len(self.image_files)))
    
    def switch_road_type(self, increase: bool = True) -> None:
        """
        Toggle the road type.
        
        Args:
            increase: If True, increment road type, otherwise decrement
        """
        if increase:
            self.road_type = (self.road_type + 1) % len(self.ROAD_TYPES)
        else:
            self.road_type = (self.road_type - 1) % len(self.ROAD_TYPES)

        print(f"Switched road type to: {self.ROAD_TYPES[self.road_type]}")
        
        if self.current_image is not None:
            cv2.imshow("Labeling", self.draw_ui(self.current_image, 
                                               self.current_idx, 
                                               len(self.image_files)))

    def label_image(self, image_path: str) -> bool:
        """
        Load and label an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if user wants to go to previous image, False for next image
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return False
            
        # Store original dimensions before resizing
        self.original_image_size = (image.shape[1], image.shape[0])
        
        # Resize the image to INPUT_SIZE for display
        resized_image = self._resize_image_preserve_aspect(image, self.INPUT_SIZE)
        self.current_image = resized_image.copy()
        
        # Load existing label if available
        loaded_points, loaded_road_type = load_label(os.path.basename(image_path), self.OUTPUT_DIR)
        
        # Set road type from loaded data
        if loaded_road_type is not None:
            self.road_type = loaded_road_type
        else:
            self.road_type = 0  # Default
        
        # Ensure points are in the correct format and scale from OUTPUT_SIZE to INPUT_SIZE
        self.points = [[] for _ in range(self.MAX_LANES)]
        for lane_idx, lane_points in enumerate(loaded_points):
            for point in lane_points:
                try:
                    if len(point) >= 3:
                        # Scale points from OUTPUT_SIZE to INPUT_SIZE
                        x = int(float(point[0]) * (self.INPUT_SIZE[0] / self.OUTPUT_SIZE[0]))
                        y = int(float(point[1]) * (self.INPUT_SIZE[1] / self.OUTPUT_SIZE[1]))
                        
                        # Keep other properties like visibility
                        scaled_point = [x, y, int(float(point[2]))]
                        if len(point) > 3:
                            scaled_point.extend(point[3:])
                            
                        self.points[lane_idx].append(scaled_point)
                except Exception as e:
                    print(f"Error converting point {point}: {e}")
        
        # Display the image
        cv2.namedWindow("Labeling")
        cv2.imshow("Labeling", self.draw_ui(self.current_image, 
                                           self.current_idx, 
                                           len(self.image_files)))
        cv2.setMouseCallback("Labeling", self.click_event)

        # Wait for keyboard input
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("d"):  # Next image
                self._save_current_label(image)
                return False
                
            elif key == ord("a"):  # Previous image
                self._save_current_label(image)
                return True
                
            elif key == ord("w"):  # Increment road type
                self.switch_road_type(True)
                
            elif key == ord("s"):  # Decrement road type
                self.switch_road_type(False)

            elif key == ord("e")  or key == ord(" "):  # Next lane
                self.switch_lane(True)
                
            elif key == ord("q"):  # Previous lane / Quit
                if key & cv2.EVENT_FLAG_CTRLKEY:  # Ctrl+Q to quit
                    self._save_current_label(image)
                    return None  # Signal to exit the program
                else:
                    self.switch_lane(False)  # Just Q for previous lane

    def _save_current_label(self, image: np.ndarray) -> None:
        """
        Save the current image label.
        
        Args:
            image: The image being labeled
        """
        if self.image_path:
            # Resize original image to OUTPUT_SIZE for saving
            output_image = self._resize_image_preserve_aspect(image, self.OUTPUT_SIZE)
            
            # Scale points from INPUT_SIZE to OUTPUT_SIZE
            scaled_points = [[] for _ in range(self.MAX_LANES)]
            for lane_idx, lane_points in enumerate(self.points):
                for point in lane_points:
                    x = int(point[0] * (self.OUTPUT_SIZE[0] / self.INPUT_SIZE[0]))
                    y = int(point[1] * (self.OUTPUT_SIZE[1] / self.INPUT_SIZE[1]))
                    scaled_point = [x, y]
                    
                    # Keep the visibility flag and other properties
                    if len(point) > 2:
                        scaled_point.extend(point[2:])
                    
                    scaled_points[lane_idx].append(scaled_point)
                
            save_label(os.path.basename(self.image_path), 
                      scaled_points, 
                      self.road_type, 
                      self.OUTPUT_DIR, 
                      output_image)
    
    def _resize_image_preserve_aspect(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size while preserving aspect ratio.
        The image is letterboxed or pillarboxed as needed.
        
        Args:
            image: Image to resize
            target_size: Target size as (width, height)
            
        Returns:
            Resized image
        """
        target_width, target_height = target_size
        h, w = image.shape[:2]
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize the image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a black canvas of target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate offsets for centering
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2
        
        # Place the resized image on the canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas

    def run(self) -> None:
        """Run the labeling application."""
        # Get list of image files
        try:
            self.image_files = [f for f in os.listdir(self.IMAGE_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
            if not self.image_files:
                print(f"No images found in {self.IMAGE_DIR}")
                return
        except Exception as e:
            print(f"Error loading image directory {self.IMAGE_DIR}: {e}")
            return

        # Main loop
        while True:
            # Set current image path
            self.image_path = os.path.join(self.IMAGE_DIR, self.image_files[self.current_idx])
            
            # Label current image
            result = self.label_image(self.image_path)
            
            # Handle navigation result
            if result is None:  # Quit
                break
            elif result:  # Go to previous image
                self.current_idx = (self.current_idx - 1) % len(self.image_files)
            else:  # Go to next image
                self.current_idx = (self.current_idx + 1) % len(self.image_files)

        cv2.destroyAllWindows()


def main() -> None:
    """Main entry point for the application."""
    app = LabelGUI()
    app.run()


if __name__ == "__main__":
    main()