import cv2 as cv
import time
from collections import deque
import numpy as np
from scipy.signal import find_peaks
import os

def detect_cars(video_file):
    Conf_threshold = 0.4
    NMS_threshold = 0.4

    class_name = []
    with open('classes.txt', 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]

    net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    cap = cv.VideoCapture(video_file)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(video_file))
    out = cv.VideoWriter(output_file, cv.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    car_counts = deque()
    frame_counter = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)

        car_count = sum(1 for (classid, score, box) in zip(classes, scores, boxes)
                        if class_name[classid] in ['car', 'bus', 'truck', 'motorbike'])

        # Draw boxes on frame (but don't display it)
        for (classid, score, box) in zip(classes, scores, boxes):
            color = (0, 255, 0)
            label = f"{class_name[classid]}: {score:.2f}"
            cv.rectangle(frame, box, color, 2)
            cv.putText(frame, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out.write(frame)
        car_counts.append((time.time(), car_count))

        # Keep data for last 30 seconds
        while car_counts and car_counts[0][0] < time.time() - 30:
            car_counts.popleft()

    cap.release()
    out.release()

    # Analyze peaks
    car_count_values = [count for _, count in car_counts]
    peaks, _ = find_peaks(car_count_values)
    mean_peak_value = np.mean([car_count_values[i] for i in peaks]) if peaks.size > 0 else 0

    return mean_peak_value
