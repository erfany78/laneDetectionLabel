import os
import numpy as np
import cv2
import json
import datetime

def load_label(image_name, output_dir):
    """Loads saved lane points and road type from the same folder where the image is located."""
    # Path to the label file with the same base name as the image
    label_path = os.path.join(output_dir, image_name.replace(".png", ".json").replace(".jpg", ".json"))
    
    if os.path.exists(label_path):
        try:
            # Load the label data from JSON file
            with open(label_path, 'r') as f:
                data = json.load(f)
            
            # Extract road type from JSON structure
            road_type = data.get('road_type', 0)
            
            # Check if this is multi-lane format
            if 'lanes' in data:
                # Multi-lane format
                lanes = data.get('lanes', [])
                processed_lanes = []
                
                # Process each lane's points
                for lane in lanes:
                    lane_points = []
                    for point in lane:
                        if len(point) >= 3:
                            new_point = [int(point[0]), int(point[1]), int(point[2])]
                            # Add angle and distance if available
                            if len(point) >= 5:
                                new_point.extend([float(point[3]), float(point[4])])
                            lane_points.append(new_point)
                    processed_lanes.append(lane_points)
                
                return processed_lanes, road_type
            else:
                # Legacy single-lane format - convert to multi-lane
                points = data.get('points', [])
                processed_points = []
                
                for point in points:
                    if len(point) >= 3:
                        new_point = [int(point[0]), int(point[1]), int(point[2])]
                        # Add angle and distance if available
                        if len(point) >= 5:
                            new_point.extend([float(point[3]), float(point[4])])
                        processed_points.append(new_point)
                
                # Return as a list with a single lane
                return [processed_points], road_type
            
        except Exception as e:
            print(f"Error loading {image_name}: {e}")
            return [[] for _ in range(5)], 0  # Return empty lanes for MAX_LANES=5
    else:
        # Try to load from old format for backwards compatibility
        old_label_path = os.path.join(output_dir, image_name.replace(".png", ".txt").replace(".jpg", ".txt"))
        if os.path.exists(old_label_path):
            try:
                print(f"Loading from legacy format: {old_label_path}")
                # Load the label data (points and road type)
                data = np.loadtxt(old_label_path, delimiter=',')
                
                # Extract points and road type
                if data.ndim == 0 or data.size == 0:
                    return [[] for _ in range(5)], 0  # Empty label file

                road_type = int(data[-1]) if data.size > 1 else 0  # Extract road type
                
                # Process the points as a single lane
                if data.size > 1:
                    points_data = data[:-1]  # All data except the last element (road type)
                    # Ensure points are in groups of 3
                    if len(points_data) % 3 == 0:
                        points = np.array(points_data).reshape(-1, 3).tolist()
                        # Make sure all point coordinates are integers
                        for i in range(len(points)):
                            points[i] = [int(float(points[i][0])), int(float(points[i][1])), int(float(points[i][2]))]
                        
                        # Save in new format for next time
                        save_label(image_name, [points], road_type, output_dir, None, convert_only=True)
                        return [points] + [[] for _ in range(4)], road_type  # One lane with points, rest empty
            except Exception as e:
                print(f"Error loading legacy format {image_name}: {e}")
                
        print(f"Label file {label_path} not found.")
        return [[] for _ in range(5)], 0  # Return empty lanes for MAX_LANES=5


def save_label(image_name, lanes, road_type, output_dir, current_image, convert_only=False):
    """
    Saves the image and corresponding label (lanes of points and road type) in JSON format.
    
    Args:
        image_name: Name of the image file
        lanes: List of lanes, where each lane is a list of points
        road_type: Type of road classification
        output_dir: Directory to save the output
        current_image: Image data to save (optional)
        convert_only: If True, only convert format without overwriting image
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if we have any points in any lanes
    total_points = sum(len(lane) for lane in lanes)
    if total_points == 0 and not convert_only:
        print(f"Warning: No points to save for {image_name}")
        return
    
    # Save the image with the same name if provided
    if current_image is not None and not convert_only:
        image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(image_path, current_image)

    # Save the label data as a JSON file with the same name (image_name) in the folder
    label_path = os.path.join(output_dir, image_name.replace(".png", ".json").replace(".jpg", ".json"))
    
    # Process all lanes
    lanes_to_save = []
    for lane in lanes:
        # Process points in this lane
        lane_points = []
        for point in lane:
            if len(point) >= 5:  # Has angle and distance data
                lane_points.append([
                    int(float(point[0])), 
                    int(float(point[1])), 
                    int(float(point[2])),
                    float(point[3]),  # angle
                    float(point[4])   # distance
                ])
            else:  # Basic x, y, visibility data
                lane_points.append([
                    int(float(point[0])), 
                    int(float(point[1])), 
                    int(float(point[2]))
                ])
        lanes_to_save.append(lane_points)
    
    # Get current date and time
    current_datetime = datetime.datetime.now().isoformat()
    
    # Create JSON structure
    label_data = {
        "image_name": image_name,
        "creation_date": current_datetime,
        "lanes": lanes_to_save,
        "road_type": int(road_type)
    }
    
    # Save the JSON data
    with open(label_path, 'w') as f:
        json.dump(label_data, f, indent=2)
    
    if not convert_only:
        print(f"Saved{'image and ' if current_image is not None else ' '}label to {label_path}")

