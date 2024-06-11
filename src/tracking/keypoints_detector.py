from ultralytics import YOLO
from typing import List, Dict
import pickle
from src.utils.utils import get_center_of_bbox


class KeypointsDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # dict in a form of
        # tactical_model_keypoint_class: tv_camera_model_keypoint_class
        self.keypoints_map_dict = {
            0: 10,  # ok
            1: 2,  # ok
            2: 7,  # ok
            3: 3,  # ok
            4: 11,  # ok
            5: 1,  # ???
            6: 13,  # ok
            7: 5,  # ok
            8: 8,  # ok
            9: 4,  # ok
            10: 12,  # ok
            11: 6,  # ok
            12: 15,  # ok
            13: 14,  # ok
            14: 16,  # ok
            15: 17,  # ???
            16: 18,  # ok
            17: 26,  # ok
            18: 23,  # ok
            19: 27,  # ok
            20: 19,  # ok
            21: 25,  # ok
            22: 21,  # ok
            23: 29,  # ok
            24: 24,  # ok
            25: 28,  # ok
            26: 20,  # ok
            27: 30,  # ???
        }

    def _detect_keypoints(self, frames, batch_size: int = 20) -> List:
        detections = list()
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i : i + batch_size], conf=0.3)
            detections += detections_batch
        return detections

    def _get_keypoints_from_tactical_model(self, detection, frame_num, keypoints):
        keypoints_coords = detection.boxes.xyxy.tolist()
        keypoints_classes = detection.boxes.cls.tolist()

        keypoints["keypoints"][frame_num] = {
            self.keypoints_map_dict[int(class_id)]: get_center_of_bbox(coord)
            for class_id, coord in zip(keypoints_classes, keypoints_coords)
        }
        return keypoints

    def get_keypoints_positions(
        self, frames, read_from_cache: bool = False, cache_path: str = None
    ) -> Dict[str, List]:
        if read_from_cache:
            with open(cache_path, "rb") as keypoints_cache:
                keypoints = pickle.load(keypoints_cache)
            return keypoints

        detections = self._detect_keypoints(frames)

        keypoints = {"keypoints": list()}

        for frame_num, detection in enumerate(detections):
            keypoints["keypoints"].append(dict())
            if detection.keypoints is None:
                keypoints = self._get_keypoints_from_tactical_model(
                    detection, frame_num, keypoints
                )
            else:
                keypoints_coords = detection.keypoints.xy[0]
                for keypoint_idx, keypoint_coord in enumerate(keypoints_coords, 1):
                    if tuple(keypoint_coord.tolist()) == (0.0, 0.0):
                        continue
                    keypoints["keypoints"][frame_num][
                        keypoint_idx
                    ] = keypoint_coord.tolist()

        with open(cache_path, "wb") as keypoints_cache:
            pickle.dump(keypoints, keypoints_cache)

        return keypoints
