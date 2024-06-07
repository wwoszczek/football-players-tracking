from ultralytics import YOLO
from typing import List, Dict
import pickle


class KeypointsDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def _detect_keypoints(self, frames, batch_size: int = 20) -> List:
        detections = list()
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i : i + batch_size], conf=0.3)
            detections += detections_batch
        return detections

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
            keypoints_coords = detection.keypoints.xy[0]
            keypoints["keypoints"].append(dict())
            for keypoint_idx, keypoint_coord in enumerate(keypoints_coords, 1):
                if tuple(keypoint_coord.tolist()) == (0.0, 0.0):
                    continue
                keypoints["keypoints"][frame_num][
                    keypoint_idx
                ] = keypoint_coord.tolist()

        with open(cache_path, "wb") as keypoints_cache:
            pickle.dump(keypoints, keypoints_cache)

        return keypoints
