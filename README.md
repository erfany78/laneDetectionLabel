# Lane Detection Labeling Tool

A graphical application for labeling road images with points and road types for lane detection purposes.

## Overview

This tool allows users to:
- Place points marking lane boundaries on road images
- Classify road types (Regular, Out of Road, Crossroad, etc.)
- Navigate through a dataset of images
- Save labels for machine learning training

## Installation

### Requirements
- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

### Setup
1. Place your images in the `dataset` directory
2. Labels will be saved to the `output` directory

## Usage

Run the application with:
```
python src/label_guiV2.py
```

## Interface Guide

```
+---------------------------------------------------------------------+
|  Type: Regular Road            Lane: 1/2            Points: 3/5     |
|  Image 1/20                    640x640, Output: 224x224             |
+---------------------------------------------------------------------+
|                                                                     |
|      [BLACK OFFSET AREA]     [IMAGE DISPLAY AREA]   [BLACK OFFSET]  |
|                                                                     |
|                                ●1                                   |
|                                 \                                   |
|                                  \                                  |
|                                   ●2                                |
|                                    \                                |
|                                     ●3    [Lane points with         |
|                                           angle, distance info]     |
|                                                                     |
|                                                                     |
+---------------------------------------------------------------------+
```

### Keyboard Controls

| Key       | Action                |
|-----------|------------------------|
| `a`       | Previous image         |
| `d`       | Next image             |
| `w`       | Next road type         |
| `s`       | Previous road type     |
| `e` / `Space` | Next lane          |
| `q`       | Previous lane          |
| `Ctrl+q`  | Quit application       |

### Mouse Controls

| Action           | Result                   |
|------------------|-----------------------------|
| Left-click       | Add point to current lane   |
| Right-click      | Remove last point from lane |

## Road Types

The tool supports the following road types:
1. Regular Road
2. RIGHT OUT OF ROAD
3. LEFT OUT OF ROAD
4. OUT
5. Crossroad
6. T-Junction

## Labeling Tips

### Points System
- You can add up to 5 points per lane
- Maximum 2 lanes can be labeled per image
- Points outside the image area are supported (shown in black offset areas)
- Points display angles and distances relative to the first point

### Visual Indicators
- Current lane points are shown with filled circles
- Inactive lane points are shown with outlined circles
- Point numbers are displayed for the active lane
- Current lane is color-coded (blue for lane 1, green for lane 2)

### Workflow Recommendation
1. Select the appropriate road type using `w`/`s`
2. Place points for the first lane (left-click)
3. Switch to the next lane with `e` or Space
4. Place points for the second lane
5. Move to the next image with `d` when finished

## Output Format

Labels are saved to the `output` directory with the following information:
- Points coordinates (scaled to 224x224)
- Point visibility flags
- Angle and distance information
- Road type classification

## Example

**Regular Road with Two Lanes:**
```
Lane 1: [Reference point] → [Point 2: 15°, 50px] → [Point 3: 30°, 100px]
Lane 2: [Reference point] → [Point 2: 165°, 50px] → [Point 3: 150°, 100px]
```
