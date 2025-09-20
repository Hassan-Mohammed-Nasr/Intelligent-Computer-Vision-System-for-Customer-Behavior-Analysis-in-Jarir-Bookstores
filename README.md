# Intelligent-Computer-Vision-System-for-Customer-Behavior-Analysis-in-Jarir-Bookstores

This project implements an advanced computer vision system for analyzing customer behavior in retail environments, specifically designed for Jarir Bookstores. The system uses YOLO for object detection, DeepSORT for tracking, and custom post-processing to analyze customer movement patterns, dwell times, and generate heatmaps.

## **Features**

* Person Detection & Tracking: Uses YOLOv8 and DeepSORT for robust person detection and tracking
* ID Re-association: Custom PostTracker class prevents ID switching and handles re-identification
* Counting System: Tracks customers entering specific zones with configurable dwell time thresholds
* Behavior Analysis: Distinguishes between customers who "dwelled" vs. "passed by" different areas
* Heatmap Generation: Creates real-time and cumulative heatmaps showing customer movement patterns
* CSV Export: Saves analysis results including dwell times and behavior classifications

## **Requirements**
### **Python Dependencies**
Install the required packages by running the installation cells in the notebook:

```python

!pip install ultralytics

!pip install git+https://github.com/KaiyangZhou/deep-person-reid.git

!pip install deep-sort-realtime
```



### **Hardware Requirements**

* CUDA-compatible GPU (recommended for real-time processing)

## **Setup Instructions**

### **1. Configure File Paths**
Before running the code, you must update the following file paths in the notebook:

#### **Video Input**

```python 
#In the "Read video and grab first frame" cell:
cap = cv2.VideoCapture('PATH TO VIDEO')  # Replace with your video file path

#In the main processing loop:
cap = cv2.VideoCapture('PATH TO VIDEO')  # Replace with your video file path
```

#### **Frame Output Path**

```python
# For saving the first frame:
cv2.imwrite('PATH TO SAVE THE IMAGE', frame)  # Replace with desired image save path
```
#### **Video Output Path**

```python
# For saving the processed video:
video_writer = cv2.VideoWriter('PATH TO SAVE THE NEW VIDEO',
                               cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (w, h))
```
#### **Heatmap Background (Optional)**

```python
# In the static heatmap section:
cap = cv2.VideoCapture('PATH TO VIDEO')  # Replace with your video file path for background
```

### **2. Define Tracking Areas**

The system requires you to define polygonal areas for tracking. Currently, three areas are defined with their main zones and entrance zones for CCTV S2G40A:

```python
# Example area definitions (update coordinates based on your video):
area_1_main_pts = np.array([[1019, 235], [658, 215], [563, 663], [1279, 606]], np.int32)
area_1_entrance_1_pts = np.array([[665, 163], [987, 184], [1014, 231], [661, 210]], np.int32)
# ... add more areas as needed
```
#### **How to get coordinates:**

1- Run the first frame extraction cell to save a frame image

2- Use an image editing tool to identify pixel coordinates for your areas of interest. (e.g. https://www.image-map.net/)

3- Update the coordinate arrays accordingly

## **How to Run**
#### **Step 1: Install Dependencies**

Run the installation cells at the beginning of the notebook to install required packages.

#### **Step 2: Extract First Frame (Optional)**
Run the frame extraction cell to get a reference image for defining your tracking areas:

```python
# This saves the first frame as an image for area definition
cap = cv2.VideoCapture('YOUR_VIDEO_PATH')
ret, frame = cap.read()
cap.release()
cv2.imwrite('YOUR_IMAGE_PATH', frame)
```
#### **Step 3: Define Tracking Areas**

Update the area coordinate arrays based on your store layout and the extracted frame.

#### **Step 4: Configure Parameters (optional but not recomended unless needed)**

Adjust tracking and counting parameters in the main loop:
```python
# YOLO detection confidence
detector = YoloDetector('yolov8l', confidence = 0.55, target_classes = [0])

# PostTracker parameters
merger = PostTracker(
    similarity_threshold=0.62,      # Appearance similarity threshold
    max_dist_from_last_seen=150,    # Max pixel distance for re-association
    time_threshold=500,             # Max frames to keep lost tracks
    motion_lambda=0.33,             # Weight for spatial vs appearance matching
    max_cost_threshold=0.6,         # Max cost for valid re-association
)

# Counting system parameters (dwell time in seconds)
counter1 = CountingSystem(..., dwell_time_seconds=5, ...)
```
#### **Step 5: Run Main Processing**

Execute the main processing loop cell. This will:


* Process the video frame by frame
* Detect and track people
* Update counting systems and heatmaps
* Generate an output video with overlays
* Save analysis results to CSV files

#### **Step 6: Generate Static Heatmap (Optional)**

Run the final cell to generate a static heatmap image overlaid on the store layout.

## **Output Files**

The system generates several output files:

* Processed Video: Video with bounding boxes, IDs, and zone overlays
* Dwell Time CSVs: ```dwell_times_[AREA_NAME].csv``` - Contains dwell times for each tracked person
* Analysis CSVs: ```analysis_[AREA_NAME].csv``` - Contains behavior analysis (dwelled vs. passed by)
* Heatmap Images:

    * ```final_heatmap_colored.jpg``` - Pure heatmap
    * ```final_heatmap_overlay.jpg``` - Heatmap overlaid on store layout



