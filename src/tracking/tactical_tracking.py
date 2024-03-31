import os
import pandas as pd
import numpy as np
import cv2
import skimage
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error
from src.tracking.utils import (
    get_keypoints_possitions_and_class_names,
    get_team_colors,
    get_video_path,
    get_detection_information,
    get_destination_coords,
)
import yaml
import time


def make_tactical_tracking():
    return 0


with open("tracking_config.yaml", "r") as tracking_config_stream:
    CONFIG_DICT = yaml.safe_load(tracking_config_stream)

KEYPOINTS_MAP_DICT, KEYPOINTS_CLASSES_DICT, PLAYERS_CLASSES_DICT = (
    get_keypoints_possitions_and_class_names(CONFIG_DICT)
)

TACTICAL_MAP_PATH = CONFIG_DICT["tactical_map_path"]

# (Upload video if wanted and) get video path
# TODO - implement uploading
VIDEO_PATH = get_video_path(CONFIG_DICT)

# for now it is given
# TODO - make it based on bounding boxes colors
COLORS_DICT, COLORS_LIST_LAB = get_team_colors(CONFIG_DICT)


model_players = YOLO(CONFIG_DICT["players_model_path"])
model_keypoints = YOLO(CONFIG_DICT["keypoints_model_path"])

cap = cv2.VideoCapture(VIDEO_PATH)
tactical_map_clean = cv2.imread(TACTICAL_MAP_PATH)
tactical_map_width, tactical_map_height, _ = tactical_map_clean.shape

if CONFIG_DICT["save_output"]:
    last_video_idx = len(os.listdir(CONFIG_DICT["output_path"]))
    output_file_name = f"tactical_tracking_{last_video_idx + 1}"
    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + tactical_map_width
    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + tactical_map_height
    output = cv2.VideoWriter(
        filename=f"{CONFIG_DICT['output_path']}/{output_file_name}.mp4",
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=30,
        frameSize=(output_width, output_height),
    )

tot_nbr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# ==================== THE GREAT LOOP ====================

# Set variable to record the time when we processed last frame
# Set variable to record the time at which we processed current frame
# Number of frames without the ball
new_frame_time, prev_frame_time = 0, 0
nbr_frames_no_ball = 0

# Store the ball track history
ball_track_history = {"src": list(), "dst": list()}

# initialize (mainly for PyCharm to not throw warnings, can be ignored)
keypoints_classes_previous_frame = list()
keypoints_coords_previous_frame = np.array([])
homog = np.array([])

for frame_idx in tqdm(range(tot_nbr_frames)):
    success, frame = cap.read()
    if not success:
        continue

    tactical_map = tactical_map_clean.copy()

    if nbr_frames_no_ball > CONFIG_DICT["nbr_frames_no_ball_thresh"]:
        ball_track_history["src"] = list()
        ball_track_history["dst"] = list()

    detection_players = model_players(
        frame, conf=CONFIG_DICT["players_model_conf_thresh"]
    )
    detection_keypoints = model_keypoints(
        frame, conf=CONFIG_DICT["keypoints_model_conf_thresh"]
    )

    players_bb_xyxy, players_bb_xywh, players_labels, players_confs = (
        get_detection_information(detection_players)
    )
    keypoints_bb_xyxy, keypoints_bb_xywh, keypoints_labels, keypoints_confs = (
        get_detection_information(detection_keypoints)
    )

    keypoints_classes = [KEYPOINTS_CLASSES_DICT[label] for label in keypoints_labels]

    keypoints_coords = np.round(keypoints_bb_xywh[:, :2].astype(int))
    tactical_map_keypoints_coords = np.array(
        [KEYPOINTS_MAP_DICT[keypoint_class] for keypoint_class in keypoints_classes]
    )

    # Calculate Homography transformation matrix if at least 4 keypoints are detected on a frame

    if len(keypoints_classes) >= 4:
        if CONFIG_DICT["update_homography_on_every_frame"] or frame_idx == 0:
            update_homography = True
        else:  # check if the distance of corresponding keypoints between previous and current frame is bigger that a given threshold
            common_keypoints = set(keypoints_classes_previous_frame).intersection(
                set(keypoints_classes)
            )
            if not common_keypoints:
                update_homography = True
            else:
                common_keypoints_indices_prev = [
                    keypoints_classes_previous_frame.index(common_keypoint)
                    for common_keypoint in common_keypoints
                ]
                common_keypoints_indices_current = [
                    keypoints_classes.index(common_keypoint)
                    for common_keypoint in common_keypoints
                ]
                common_keypoints_coords_prev = keypoints_coords_previous_frame[
                    common_keypoints_indices_prev
                ]

                common_keypoints_coords_current = keypoints_coords[
                    common_keypoints_indices_current
                ]

                common_keypoints_mean_distance = mean_squared_error(
                    common_keypoints_coords_prev, common_keypoints_coords_current
                )

                update_homography = (
                    common_keypoints_mean_distance
                    > CONFIG_DICT["keypoints_displacement_mean_tol"]
                )

        if update_homography:
            homog, mask = cv2.findHomography(
                keypoints_coords, tactical_map_keypoints_coords
            )

        # TODO - check if works without .copy()
        keypoints_classes_previous_frame = keypoints_classes.copy()
        keypoints_coords_previous_frame = keypoints_coords.copy()

        # get players and ball coords
        players_classes = np.array(
            [PLAYERS_CLASSES_DICT[label] for label in players_labels]
        )
        players_mask = players_classes == "player"
        ball_mask = players_classes == "ball"

        players_class_player = players_bb_xywh[players_mask]
        players_class_ball = players_bb_xywh[ball_mask]

        # get coordinates of detected players (x_cen, y_cen + h/2)
        # TODO check why + h/2 and determine if needed
        # TODO check if change here doesn't destroy everything
        players_coords_src = (
            players_class_player[:, :2]
            + np.array([0, players_class_player[:, 3] / 2])[:, None].T
        )

        ball_coords_src = (
            players_class_ball[0, :2] if players_class_ball.size != 0 else None
        )

        # TODO check frames not ball mechanism - what is the aim and if it can be replaced with something better
        nbr_frames_no_ball += 1
        if ball_coords_src is not None:
            nbr_frames_no_ball = 0

        # Transform players coordinates to tactical map using homography matrix
        players_dst_coords = get_destination_coords(
            source_coords=players_coords_src, homography_matrix=homog
        )

        if ball_coords_src is not None:
            ball_dst_coords = get_destination_coords(
                source_coords=ball_coords_src, homography_matrix=homog
            )

            # track ball history
            # TODO - check wtf is this and if this is needed
            # TODO- also, check what is in plot_hyperparams and validate with current code


# if __name__ == '__main__':
#     make_tactical_tracking()
