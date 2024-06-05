import numpy as np
from ultralytics import YOLO
import supervision as sv
from typing import List, Dict
import pickle
import os
from src.utils.utils import get_center_of_bbox, get_foot_position
import cv2


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def _detect_frames(self, frames: List, batch_size: int = 20) -> List:
        detections = list()
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i : i + batch_size], conf=0.3)
            detections += detections_batch
        return detections

    def get_object_tracks(
        self, frames, read_from_cache: bool = False
    ) -> Dict[str, List]:
        if read_from_cache:
            with open("output/tracks_cache.yaml", "rb") as tracks_cache:
                tracks = pickle.load(tracks_cache)
            return tracks

        detections = self._detect_frames(frames)

        tracks = {"players": list(), "referees": list(), "ball": list()}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks["players"].append(dict())
            tracks["referees"].append(dict())
            tracks["ball"].append(dict())

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    tracks["players"][frame_num][track_id]["position"] = (
                        get_foot_position(bbox)
                    )

                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    tracks["referees"][frame_num][track_id]["position"] = (
                        get_foot_position(bbox)
                    )

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
                    tracks["ball"][frame_num][1]["position"] = get_center_of_bbox(bbox)

        if not os.path.exists("output/tracks_cache.yaml"):
            with open("output/tracks_cache.yaml", "wb") as tracks_cache:
                pickle.dump(tracks, tracks_cache)

        return tracks
