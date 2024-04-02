import cv2
import numpy as np
import Run_a_PATH_in_VSCODE # if you have issue whith the folder path (my case :/)
from asscii_menu import Menu

class YellowCarDetector:
    def __init__(self, video_source=0, yolo_cfg='yolov4.cfg', yolo_weights='yolov4.weights', coco_names='coco.names'):
        # Load the YOLO object detector trained on COCO dataset (80 classes)
        self.net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
        # Load the class labels our YOLO model was trained on
        self.LABELS = open(coco_names).read().strip().split("\n")
        # Get the names of the output layers
        layer_names = self.net.getLayerNames()
        self.ln = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        # Initialize a list of colors to represent each possible class label
        self.COLORS = np.random.uniform(0, 255, size=(len(self.LABELS), 3))
        # Initialize the video stream
        self.cap = cv2.VideoCapture(video_source)

    def detect_yellow_cars(self, image):
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        # Perform a forward pass through the YOLO network
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5 and self.LABELS[classID] == "car":
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        car_count = 0
        if len(indexes) > 0:
            # avoid error when using LiveCam
            if isinstance(indexes, tuple):
                indexes = indexes[0]
            for i in indexes.flatten():
                car_count += 1
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                # Extract the ROI for car detection
                roi = image[y:y+h, x:x+w]
                
                # check Yellow color
                yellow_percentage = self.detect_yellow_color(roi, w, h)
                label = f"Car {car_count}: Yellow {yellow_percentage:.2f}%"
                
                color = [int(c) for c in self.COLORS[i]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Image", image)
    
    def detect_yellow_color(self, roi, w, h):
        lower_yellow_rgb = np.array([150, 150, 0])
        upper_yellow_rgb = np.array([255, 255, 120])
        try : 
            # ROI in HSV
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Yellow range HSV
            lower_yellow_hsv = np.array([20, 100, 100])
            upper_yellow_hsv = np.array([30, 255, 255])
            
            mask_hsv = cv2.inRange(roi_hsv, lower_yellow_hsv, upper_yellow_hsv)
            yellow_pixels_hsv = cv2.countNonZero(mask_hsv)
            total_pixels = roi_hsv.size / 3  
            yellow_percentage_hsv = (yellow_pixels_hsv / total_pixels) * 100
        except:
            yellow_percentage_hsv = 0 
        return yellow_percentage_hsv

    def process_video(self):
        desired_width = 800 
        while True:
            ret, image = self.cap.read()
            if not ret:
                break
            # keep the scale w/h
            scale = desired_width / image.shape[1]
            new_dim = (desired_width, int(image.shape[0] * scale))
            resized_image = cv2.resize(image, new_dim)
            # run the fontion to detect cars 
            self.detect_yellow_cars(resized_image)

            cv2.imshow("Image", resized_image)
            if cv2.waitKey(1) == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Menu()
    choice = int(input('Live Cam (0) -- Video Load (1)  :')) 
    if choice: video_path = '.\datasset\Video3.mp4'  # (path for a video test) or put your video in the folder 'datasset'
    else: video_path = 0
    detector = YellowCarDetector(video_source=video_path)
    detector.process_video()
