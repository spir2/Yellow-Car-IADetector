# Yellow Car Detector

This Python project utilizes OpenCV and YOLO (You Only Look Once) for the detection of yellow cars in video streams or video files. It's designed to run on environments where OpenCV is supported, and it has been tested with YOLOv4 for object detection.
![visual](https://github.com/spir2/Yellow-Car-IADetector/assets/130176259/b40d1633-3ea0-4c29-81c7-04ce5f7e6395)
## Prerequisites

Before running this project, ensure you have the following installed:
- Python 3.11
- OpenCV library (`cv2`)
- NumPy

Additionally, you will need the following YOLOv4 files:
- `yolov4.cfg`: Configuration file
    https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg
- `yolov4.weights`: Pre-trained weights
    https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
- `coco.names`: File containing COCO dataset class labels
    https://github.com/pjreddie/darknet/blob/master/data/coco.names
  
## Installation

1. Clone this repository to your local machine.
2. Download the YOLOv4 configuration file, weights, and COCO names from the official YOLO website or a trusted source.
3. Place the `yolov4.cfg`, `yolov4.weights`, and `coco.names` files in the same directory as the script.

## Usage

To run the Yellow Car Detector, navigate to the directory containing the script and execute the following command:

```bash
python yellow_car_detector.py
```
Upon execution, the program will prompt you to select the video source:

* Enter 0 for live camera feed.
* Enter 1 to load a video file. You will need to specify the path to your video file.
* The detector will process the selected video source and display the results in a new window, highlighting detected yellow cars with bounding boxes and labels.
