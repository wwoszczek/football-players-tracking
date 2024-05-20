import os
import pandas as pd
import numpy as np
import cv2
import skimage
from PIL import Image
from tqdm import tqdm
import supervision as sv
import uuid
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error
from src.tracking.utils import (
    get_keypoints_possitions_and_class_names,
    get_team_colors,
    get_video_path,
    get_detection_information,
    get_destination_coords,
    update_ball_tracking_history,
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

tracker = sv.ByteTrack()

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

for frame_idx in tqdm(range(tot_nbr_frames)[200:300]):

    # get a specific frame from a video
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if not success:
        continue

    tactical_map = tactical_map_clean.copy()

    if nbr_frames_no_ball > CONFIG_DICT["nbr_frames_no_ball_thresh"]:
        ball_track_history["src"] = list()
        ball_track_history["dst"] = list()

    detection_players = model_players(
        frame,
        conf=CONFIG_DICT["players_model_conf_thresh"],
        iou=CONFIG_DICT["players_model_iou_thresh"],
    )
    detection_keypoints = model_keypoints(
        frame, conf=CONFIG_DICT["keypoints_model_conf_thresh"], iou=0.0
    )

    players_bb_xyxy, players_bb_xywh, players_labels, players_confs = (
        get_detection_information(detection_players)
    )

    players_ultralitics_result = detection_players[0]

    keypoints_bb_xyxy, keypoints_bb_xywh, keypoints_labels, keypoints_confs = (
        get_detection_information(detection_keypoints)
    )

    # TODO - check if limiting to 4 keypoints does not make results better
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
            # TODO check if limiting to 4 keypoints make the transformation better
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
        players_src_coords = (
            players_class_player[:, :2]
            + np.array(
                [[0] * players_class_player.shape[0], players_class_player[:, 3] / 2]
            ).T
        )

        # if detected more than 1 ball, take only 1st one (should not happen)
        ball_src_coords = (
            players_class_ball[:1, :2] if players_class_ball.size != 0 else None
        )

        # TODO check frames not ball mechanism - what is the aim and if it can be replaced with something better
        nbr_frames_no_ball += 1
        if ball_src_coords is not None:
            nbr_frames_no_ball = 0

        # Transform players coordinates to tactical map using homography matrix
        players_dst_coords = get_destination_coords(
            source_coords=players_src_coords, homography_matrix=homog
        )

        # Transform ball coordinates to tactical map using homography matrix
        if ball_src_coords is not None:
            ball_dst_coords = get_destination_coords(
                source_coords=ball_src_coords, homography_matrix=homog
            )

            # track ball history
            # TODO - check wtf is this and if this is needed

            if CONFIG_DICT["show_ball_history"]:
                ball_track_history = update_ball_tracking_history(
                    ball_track_history,
                    ball_src_coords,
                    ball_dst_coords,
                    config_dict=CONFIG_DICT,
                )

    ######### Part 2 ##########
    # Players Team Prediction #
    ###########################

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    obj_palette_list = []  # Initialize players color palette list
    palette_interval = (
        0,
        3,
    )  # Color interval to extract from dominant colors palette (1rd to 3rd color)

    ## Loop over detected players (label 0) and extract dominant colors palette based on defined interval
    for idx, label in enumerate(players_labels):
        if int(label) == 1:
            bbox = players_bb_xyxy[idx, :]  # Get bbox info (x,y,x,y)
            obj_img = frame_rgb[
                int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])
            ]  # Crop bbox out of the frame

            cv2.imwrite(
                f"/Users/wwoszczek/Desktop/studia/magister/football-players-tracking/output/players/player_{idx}_{label}_{uuid.uuid4()}.jpg",
                obj_img,
            )

            obj_img_w, obj_img_h = obj_img.shape[1], obj_img.shape[0]
            center_filter_x1 = np.max([(obj_img_w // 2) - (obj_img_w // 5), 1])
            center_filter_x2 = (obj_img_w // 2) + (obj_img_w // 5)
            center_filter_y1 = np.max([(obj_img_h // 3) - (obj_img_h // 5), 1])
            center_filter_y2 = (obj_img_h // 3) + (obj_img_h // 5)
            center_filter = obj_img[
                center_filter_y1:center_filter_y2, center_filter_x1:center_filter_x2
            ]
            obj_pil_img = Image.fromarray(
                np.uint8(center_filter)
            )  # Convert to pillow image
            reduced = obj_pil_img.convert(
                "P", palette=Image.Palette.WEB
            )  # Convert to web palette (216 colors)
            palette = reduced.getpalette()  # Get palette as [r,g,b,r,g,b,...]
            palette = [
                palette[3 * n : 3 * n + 3] for n in range(256)
            ]  # Group 3 by 3 = [[r,g,b],[r,g,b],...]
            color_count = [
                (n, palette[m]) for n, m in reduced.getcolors()
            ]  # Create list of palette colors with their frequency
            RGB_df = (
                pd.DataFrame(color_count, columns=["cnt", "RGB"])
                .sort_values(
                    # Create dataframe based on defined palette interval
                    by="cnt",
                    ascending=False,
                )
                .iloc[palette_interval[0] : palette_interval[1], :]
            )
            palette = list(
                RGB_df.RGB
            )  # Convert palette to list (for faster processing)

            # Update detected players color palette list
            obj_palette_list.append(palette)

    ## Calculate distances between each color from every detected player color palette and the predefined teams colors
    players_distance_features = []
    # Loop over detected players extracted color palettes
    for palette in obj_palette_list:
        palette_distance = []
        palette_lab = [
            skimage.color.rgb2lab([i / 255 for i in color]) for color in palette
        ]  # Convert colors to L*a*b* space
        # Loop over colors in palette
        for color in palette_lab:
            distance_list = []
            # Loop over predefined list of teams colors
            for c in COLORS_LIST_LAB:
                # distance = np.linalg.norm([i/255 - j/255 for i,j in zip(color,c)])
                distance = skimage.color.deltaE_cie76(
                    color, c
                )  # Calculate Euclidean distance in Lab color space
                distance_list.append(distance)  # Update distance list for current color
            palette_distance.append(
                distance_list
            )  # Update distance list for current palette
        players_distance_features.append(palette_distance)

    ## Predict detected players teams based on distance features
    players_teams_list = []
    # Loop over players distance features
    for distance_feats in players_distance_features:
        vote_list = []
        # Loop over distances for each color
        for dist_list in distance_feats:
            team_idx = dist_list.index(min(dist_list)) // len(
                list(COLORS_DICT.values())[0]
            )  # Assign team index for current color based on min distance
            vote_list.append(
                team_idx
            )  # Update vote voting list with current color team prediction
        players_teams_list.append(max(vote_list, key=vote_list.count))

    #################### Part 3 #####################
    # Updated Frame & Tactical Map With Annotations #
    #################################################

    ball_color_bgr = (0, 0, 255)  # Color (GBR) for ball annotation on tactical map
    players_counter = 0  # Initializing counter of detected players
    annotated_frame = frame

    # get tracks

    tracks = sv.Detections.from_ultralytics(players_ultralitics_result)
    # tracks["names"] = [
    #     model_players.model.names[class_id] for class_id in tracks.class_id
    # ]

    detections_with_tracks = tracker.update_with_detections(tracks)
    track_id_dict = {}
    for frame_detection in detections_with_tracks:
        # bbox_xyxy : track_id
        track_id_dict[str(frame_detection[0])] = frame_detection[4]

    # Loop over all detected object by players detection model
    for i in range(players_bb_xyxy.shape[0]):
        conf = players_confs[i]  # Get confidence of current detected object
        if players_labels[i] == 1:  # Display annotation for detected players (label 0)
            team_name = list(COLORS_DICT.keys())[
                players_teams_list[players_counter]
            ]  # Get detected player team prediction
            track_id = (
                track_id_dict[str(players_bb_xyxy[i])]
                if str(players_bb_xyxy[i]) in track_id_dict
                else 0
            )
            color_rgb = COLORS_DICT[team_name][0]  # Get detected player team color
            color_bgr = color_rgb[::-1]  # Convert color to bgr
            if CONFIG_DICT["show_players_detections"]:
                annotated_frame = cv2.rectangle(
                    annotated_frame,
                    (int(players_bb_xyxy[i, 0]), int(players_bb_xyxy[i, 1])),
                    # Add bbox annotations with team colors
                    (int(players_bb_xyxy[i, 2]), int(players_bb_xyxy[i, 3])),
                    color_bgr,
                    1,
                )

                annotated_frame = cv2.putText(
                    annotated_frame,
                    team_name
                    + f" {conf:.2f} ID: {track_id}",  # Add team name annotations
                    (int(players_bb_xyxy[i, 0]), int(players_bb_xyxy[i, 1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color_bgr,
                    2,
                )

            # Add tactical map player postion color coded annotation if more than 3 field keypoints are detected
            tactical_map = cv2.circle(
                tactical_map,
                (
                    int(players_dst_coords[players_counter][0]),
                    int(players_dst_coords[players_counter][1]),
                ),
                radius=5,
                color=color_bgr,
                thickness=-1,
            )
            tactical_map = cv2.circle(
                tactical_map,
                (
                    int(players_dst_coords[players_counter][0]),
                    int(players_dst_coords[players_counter][1]),
                ),
                radius=5,
                color=(0, 0, 0),
                thickness=1,
            )

            players_counter += 1  # Update players counter
        else:  # Display annotation for otehr detections (label 1, 2)
            annotated_frame = cv2.rectangle(
                annotated_frame,
                (int(players_bb_xyxy[i, 0]), int(players_bb_xyxy[i, 1])),
                # Add white colored bbox annotations
                (int(players_bb_xyxy[i, 2]), int(players_bb_xyxy[i, 3])),
                (255, 255, 255),
                1,
            )
            annotated_frame = cv2.putText(
                annotated_frame,
                PLAYERS_CLASSES_DICT[players_labels[i]] + f" {conf:.2f}",
                # Add white colored label text annotations
                (int(players_bb_xyxy[i, 0]), int(players_bb_xyxy[i, 1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            # Add tactical map ball postion annotation if detected
            if ball_src_coords is not None:
                tactical_map = cv2.circle(
                    tactical_map,
                    (int(ball_dst_coords[0, 0]), int(ball_dst_coords[0, 1])),
                    radius=5,
                    color=ball_color_bgr,
                    thickness=3,
                )
    if CONFIG_DICT["show_keypoints_detections"]:
        for i in range(keypoints_bb_xyxy.shape[0]):
            annotated_frame = cv2.rectangle(
                annotated_frame,
                (int(keypoints_bb_xyxy[i, 0]), int(keypoints_bb_xyxy[i, 1])),
                # Add bbox annotations with team colors
                (int(keypoints_bb_xyxy[i, 2]), int(keypoints_bb_xyxy[i, 3])),
                (0, 0, 0),
                1,
            )
    # Plot the tracks
    if len(ball_track_history["src"]) > 0:
        points = (
            np.hstack(ball_track_history["dst"]).astype(np.int32).reshape((-1, 1, 2))
        )
        tactical_map = cv2.polylines(
            tactical_map, [points], isClosed=False, color=(0, 0, 100), thickness=2
        )

    # Combine annotated frame and tactical map in one image with colored border separation
    border_color = [255, 255, 255]  # Set border color (BGR)
    annotated_frame = cv2.copyMakeBorder(
        annotated_frame,
        40,
        10,
        10,
        10,  # Add borders to annotated frame
        cv2.BORDER_CONSTANT,
        value=border_color,
    )
    tactical_map = cv2.copyMakeBorder(
        tactical_map,
        70,
        50,
        10,
        10,
        cv2.BORDER_CONSTANT,  # Add borders to tactical map
        value=border_color,
    )
    tactical_map = cv2.resize(
        tactical_map, (tactical_map.shape[1], annotated_frame.shape[0])
    )  # Resize tactical map
    final_img = cv2.hconcat((annotated_frame, tactical_map))  # Concatenate both images
    ## Add info annotation
    cv2.putText(
        final_img,
        "Tactical Map",
        (1370, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
    )

    new_frame_time = time.time()  # Get time after finished processing current frame
    prev_frame_time = new_frame_time  # Save current time to be used in next frame

    # cv2.imshow("YOLOv8 Inference", frame)
    if CONFIG_DICT["save_output"]:
        output.write(cv2.resize(final_img, (output_width, output_height)))

    # TODO - try colors differently
    # TODO - dont assign to teams, but just assign color from 8 RGB which is the closest based on color palette
    # TODO - goalkeeper will be different but its ok

    # TODO - tactical map made like in FIFA - zrób mapę przezryczystą i wstaw jak w fife, zmień też mapkę
    # TODO - do non tactical autmated ball possesion tracking bo większa piłka

# if __name__ == '__main__':
#     make_tactical_tracking()
