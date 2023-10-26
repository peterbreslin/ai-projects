import cv2
import torch
import numpy as np


class LiveDetection:
    """
    Uses the YOLO (v5) model to make inferences on live video.
    """

    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names #list of string labels corresponding to the numeric class labels in the model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def get_webcam_feed(self):
        """
        Creates a video capture object to capture frames from the laptop's built-in webcam.
        """
        return cv2.VideoCapture(0)  #zero is default for the built-in webcam.


    def load_model(self):
        """
        Loads the pre-trained Yolo5 model from pytorch hub.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model


    def score_frame(self, frame):
        """
        Performs the object detection.
         - The model identifies objects within the the single frame that is given as input (i.e. scoring).
         - Returns the labels of the detected objects in the frame and their corresponding bounding box coordinates.
        """
        self.model.to(self.device) #setting the model's device to the class-defined device (i.e. cpu or gpu)
        frame = [frame] #input frame must be in numpy/list/tuple format
        predictions = self.model(frame)
        labels, coords = predictions.xyxyn[0][:, -1].numpy(), predictions.xyxyn[0][:, :-1].numpy()
        return labels, coords


    def label_conversion(self, numeric_label):
        """
        Converts a given numeric class label to its corresponding string label.
        """
        return self.classes[int(numeric_label)]


    def plot_bounding_boxes(self, frame, labels, coords):
        """
        Plots the bounding boxes and labels onto the input scored frame.
        """
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        # Looping over all detected objects.
        for i in range(len(labels)):
            row = coords[i]
            confidence = row[4] #the confidence score in YOLO models is typically the fifth element in the coord array 
            if confidence >= 0.2: 
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (128, 0, 128)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                label = f"{self.label_conversion(labels[i])} ({confidence:.2f})"
                ypos = y1-10
                cv2.putText(frame, label, (x1, ypos), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr, 2)
        return frame


    def __call__(self):
        """
        This is called when the Class is executed: it runs the loop to read the video frame by frame.
        """
        webcam = self.get_webcam_feed()
        assert webcam.isOpened()
        while True:
            ret, frame = webcam.read()
            assert ret
            frame = cv2.flip(frame, 1)
            labels, coords = self.score_frame(frame)
            frame = self.plot_bounding_boxes(frame, labels, coords)
            if not ret:
                break
            cv2.imshow('Object Detection', frame)
            key = cv2.waitKey(1)  #display speed (1ms delay)

            if key == 27:  #press Esc key to exit the loop
                break

        webcam.release()
        cv2.destroyAllWindows()


x = LiveDetection()
x()