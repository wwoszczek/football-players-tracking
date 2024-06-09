from typing import List
import cv2
import numpy as np
import numpy.typing as npt
from src.field.football_field import FootballField
from src.utils.utils import get_center_of_bbox


class ViewTransformer:
    def __init__(self, keypoints, football_field: FootballField) -> None:
        """
        Initialize the ViewTransformer and calculate homography matrix for
        source and target points from keypoints detected and football field coordinates

        Args:
            keypoints: Keypoints detected used as source points for homography calculation.
            football_field (FootballField): football field object used for target points homography calculation.

        Raises:
            ValueError: If source and target do not have the same shape or if they are
                not 2D coordinates.
        """

        if len(list(keypoints["keypoints"][0].values())[0]) != 2:
            raise ValueError("Source keypoints must be 2D coordinates.")

        homography_matrices = list()
        for frame_num, keypoints_dict in enumerate(keypoints["keypoints"]):
            source = list()
            target = list()
            for keypoint_idx, keypoint_coords in keypoints_dict.items():
                source.append(keypoint_coords)
                target.append(football_field.config.vertices[keypoint_idx - 1])

            source = np.array(source).astype(np.float32)
            target = np.array(target).astype(np.float32)
            m, _ = cv2.findHomography(source, target)
            if m is None:
                raise ValueError("Homography matrix could not be calculated.")
            homography_matrices.append(m)

        self.homography_matrices = homography_matrices

    def _transform_points(
        self, tracks, object_type="players", team_id=1
    ) -> List[npt.NDArray[np.float32]]:
        """
        Transform source points to target points using homography matrices calulated
        :param tracks: (Dict) tracks dict
        :param object_type: (str) One of 'players', 'referees' or 'ball'
        :return:  Target points numpy array
        """
        if object_type in ("players", "referees", "ball"):
            tracks_list = tracks[object_type]
        else:
            raise ValueError(
                "Object type not valid, must be 'players', 'referees' or 'ball'."
            )

        if len(self.homography_matrices) != len(tracks_list):
            raise ValueError(
                "Number of frames don't match for homography matrices calculated and tracks given."
            )

        target_points_list = list()
        for frame_num, track_dict in enumerate(tracks_list):
            if not track_dict:
                target_points_list.append(
                    list()
                )  # ??? how to append empty point for later drawing
                continue

            bboxes = track_dict.values()
            points = list()
            for bbox in bboxes:
                if object_type == "ball":
                    points.append(get_center_of_bbox(bbox["bbox"]))
                elif object_type == "players":
                    if bbox["team"] == team_id:
                        points.append(bbox["position"])
                else:
                    points.append(bbox["position"])
            points = np.array(points)
            reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
            transformed_points = cv2.perspectiveTransform(
                reshaped_points, self.homography_matrices[frame_num]
            )
            target_points_list.append(
                transformed_points.reshape(-1, 2).astype(np.float32)
            )

        return target_points_list

    def transform_tracks(self, tracks):
        target_players_team_1 = self._transform_points(
            tracks, object_type="players", team_id=1
        )
        target_players_team_2 = self._transform_points(
            tracks, object_type="players", team_id=2
        )
        target_referees = self._transform_points(tracks, object_type="referees")
        target_balls = self._transform_points(tracks, object_type="ball")

        return (
            target_players_team_1,
            target_players_team_2,
            target_referees,
            target_balls,
        )
