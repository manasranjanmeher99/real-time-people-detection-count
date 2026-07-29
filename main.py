#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch

# Force MPS to appear unavailable so nothing tries to route through it
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False


# In[ ]:


import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

VIDEO = "source/people-walking.mp4"
video_info = sv.VideoInfo.from_video_path(VIDEO)
print(video_info.resolution_wh)

colors = sv.ColorPalette.from_hex([
    "#3B82F6",
    "#10B981",
    "#F59E0B",
    "#EC4899",
])

polygons = [
    np.array([[21, 29], [1886, 32], [1889, 1041], [42, 1038], [33, 26]])
]

zones = [sv.PolygonZone(polygon=polygon) for polygon in polygons]
zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=zone,
        color=colors.by_idx(index),
        thickness=2,
        text_thickness=1,
        text_scale=1,
    )
    for index, zone in enumerate(zones)
]

box_annotator = sv.BoxAnnotator(color=colors.by_idx(0), thickness=2)
label_annotator = sv.LabelAnnotator(color=colors.by_idx(0), text_thickness=1, text_scale=0.5)

# ---- Two lines:   upper for IN (moving down), lower for OUT (moving up) ----
UPPER_Y = 400
LOWER_Y = 650

upper_line = sv.LineZone(
    start=sv.Point(21, UPPER_Y),
    end=sv.Point(1889, UPPER_Y),
)
lower_line = sv.LineZone(
    start=sv.Point(21, LOWER_Y),
    end=sv.Point(1889, LOWER_Y),
)

upper_line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.6, color=sv.Color.GREEN)
lower_line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.6, color=sv.Color.RED)

byte_tracker = sv.ByteTrack(frame_rate=video_info.fps)


def process_frame(frame: np.ndarray, i) -> np.ndarray:
    results = model(frame, imgsz=1280, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for tracker_id, class_id in zip(detections.tracker_id, detections.class_id)
    ]

    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    for zone, zone_annotator in zip(zones, zone_annotators):
        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

    upper_line.trigger(detections=detections)
    lower_line.trigger(detections=detections)

    frame = upper_line_annotator.annotate(frame=frame, line_counter=upper_line)
    frame = lower_line_annotator.annotate(frame=frame, line_counter=lower_line)

    # Overlay combined IN/OUT totals in a corner
    cv2.putText(
        frame,
        f"IN: {upper_line.in_count}   OUT: {lower_line.out_count}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )

    return frame

sv.process_video(source_path=VIDEO, target_path="output/result.mp4", callback=process_frame)

print(f"Total IN (crossed upper line downward): {upper_line.in_count}")
print(f"Total OUT (crossed lower line upward): {lower_line.out_count}")



# In[ ]:


import cv2
import numpy as np
import supervision as sv

VIDEO = "source/people-walking.mp4"
video_info = sv.VideoInfo.from_video_path(VIDEO)

heat_map_annotator = sv.HeatMapAnnotator(
    radius=25,
    opacity=0.5,
)

byte_tracker = sv.ByteTrack(frame_rate=video_info.fps)

last_frame = None

for frame in sv.get_video_frames_generator(VIDEO):
    results = model(frame, imgsz=1280, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)

    frame = heat_map_annotator.annotate(scene=frame.copy(), detections=detections)
    last_frame = frame

cv2.imwrite("output/heatmap.png", last_frame)

