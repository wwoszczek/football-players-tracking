import json

import numpy as np
import yaml
from typing import Tuple
from PIL import Image, ImageColor
import skimage


def get_keypoints_possitions_and_class_names(
    config_dict: dict,
) -> Tuple[dict, dict, dict]:
    # Get tactical map keypoints positions dict
    with open(
        config_dict["keypoints_pitch_map_position_path"], "r"
    ) as keypoints_pitch_map_position_stream:
        keypoints_map_dict = json.load(keypoints_pitch_map_position_stream)

    # Get football field keypoints numerical to alphabetical mapping
    with open(config_dict["keypoints_classes_path"], "r") as keypoints_classes_stream:
        keypoints_classes_dict = yaml.safe_load(keypoints_classes_stream)["names"]

    # Get players tracking numerical to alphabetical mapping
    with open(config_dict["players_classes_path"], "r") as players_classes_stream:
        players_classes_dict = yaml.safe_load(players_classes_stream)["names"]

    return keypoints_map_dict, keypoints_classes_dict, players_classes_dict


def get_team_colors(
    config_dict: dict,
    team1_name="team1",
    team1_p_color=None,
    team1_gk_color=None,
    team2_name="team2",
    team2_p_color=None,
    team2_gk_color=None,
) -> Tuple[dict, list]:
    if config_dict["test_video"]:
        colors_dict = {
            team1_name: [
                (41, 71, 138),
                (220, 98, 88),
            ],  # Chelsea colors (Players kit color, GK kit color)
            team2_name: [
                (144, 200, 255),
                (188, 199, 3),
            ],  # Man City colors (Players kit color, GK kit color)
        }
    else:
        team1_p_color_rgb = ImageColor.getcolor(team1_p_color, "RGB")
        team1_gk_color_rgb = ImageColor.getcolor(team1_gk_color, "RGB")
        team2_p_color_rgb = ImageColor.getcolor(team2_p_color, "RGB")
        team2_gk_color_rgb = ImageColor.getcolor(team2_gk_color, "RGB")

        colors_dict = {
            team1_name: [team1_p_color_rgb, team1_gk_color_rgb],
            team2_name: [team2_p_color_rgb, team2_gk_color_rgb],
        }

    colors_list = (
        colors_dict[team1_name] + colors_dict[team2_name]
    )  # Define color list to be used for detected player team prediction
    color_list_lab = [
        skimage.color.rgb2lab([i / 255 for i in c]) for c in colors_list
    ]  # Converting color_list to L*a*b* space

    return colors_dict, color_list_lab


def get_video_path(config_dict: dict) -> str:
    if config_dict["test_video"]:
        return config_dict["test_video_path"]
    else:
        pass  # TODO implement uploading video


def get_detection_information(detection_result: list) -> Tuple:
    bounding_boxes_xyxy = detection_result[0].boxes.xyxy.cpu().numpy()
    bounding_boxes_xywh = detection_result[0].boxes.xywh.cpu().numpy()
    labels = list(detection_result[0].boxes.label.cls.cpu().numpy())
    confs = list(detection_result[0].boxes.conf.cpu().numpy())

    return bounding_boxes_xyxy, bounding_boxes_xywh, labels, confs


def get_destination_coords(
    source_coords: np.ndarray, homography_matrix: np.ndarray
) -> np.ndarray:
    # Convert to homogeneous coordinates by adding a row of ones
    homogeneous_src_coords = np.hstack(
        [source_coords, np.ones((source_coords.shape[0], 1))]
    )

    # Apply homography transformation
    dest_coords_homog = np.dot(homography_matrix, homogeneous_src_coords.T)

    # Normalize back to 2D coordinates
    dest_coords_normalized = dest_coords_homog / dest_coords_homog[2, :]

    # Exclude the normalization row and transpose back to original shape
    dest_coords = dest_coords_normalized[:2, :].T

    return dest_coords
