import streamlit as st
from src.utils.utils import read_video, save_video
from src.tracking.tracker import Tracker
from src.tracking.keypoints_detector import KeypointsDetector
from src.tracking.interpolator import Interpolator
from src.view_transformer.view_transformer import ViewTransformer
from src.field.football_field import FootballField
from src.possession.possession_assigner import PossessionAssigner
from src.drawing.drawer import Drawer
from src.team_assigner.team_assigner import TeamAssigner
import yaml
import tempfile


@st.cache_data
def load_config(camera_view: str):
    if camera_view == "Tactical":
        yaml_name = "tactical_config.yaml"
    elif camera_view == "TV":
        yaml_name = "tv_config.yaml"
    else:
        raise ValueError("Unknown camera type!")
    with open(yaml_name, "r") as tracking_config_stream:
        return yaml.safe_load(tracking_config_stream)


@st.cache_resource
def get_tracker(model_path, confidence_threshold, iou_threshold):
    return Tracker(model_path)


@st.cache_resource
def get_keypoints_detector(model_path):
    return KeypointsDetector(model_path)


@st.cache_data
def process_video(
    config_dict,
    video_path,
    confidence_threshold,
    iou_threshold,
    interpolate_ball,
    draw_keypoints,
    read_from_cache,
):
    # Update model paths based on camera view
    players_model_path = config_dict["players_model_path"]
    keypoints_model_path = config_dict["keypoints_model_path"]

    # read video
    video_frames = read_video(video_path)

    # Initialize Tracker
    tracker = get_tracker(players_model_path, confidence_threshold, iou_threshold)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_cache=read_from_cache,
        cache_path=config_dict["tracking_cache_path"],
    )

    if interpolate_ball:
        # Interpolate ball detections
        interpolator = Interpolator()
        tracks["ball"] = interpolator.interpolate_ball_positions(tracks["ball"])

    # Initialize keypoints detector
    keypoints_detector = get_keypoints_detector(keypoints_model_path)
    keypoints = keypoints_detector.get_keypoints_positions(
        video_frames,
        read_from_cache=read_from_cache,
        cache_path=config_dict["keypoints_cache_path"],
    )

    # Assign Player Teams
    team_assigner = TeamAssigner()
    tracks = team_assigner.assign_players_teams(tracks, video_frames)

    # Assign ball possession
    possession_assigner = PossessionAssigner()
    tracks, team_ball_control = possession_assigner.assign_and_calculate_possession(
        tracks
    )

    # Football field class
    football_field = FootballField()

    # Initialize view transformer which will calculate target points based on homography matrices
    view_transformer = ViewTransformer(keypoints, football_field)
    players_team_1_to_draw, players_team_2_to_draw, referees_to_draw, balls_to_draw = (
        view_transformer.transform_tracks(tracks)
    )

    # Draw output video frames
    drawer = Drawer()
    output_video_frames = drawer.draw_annotations(video_frames, tracks)

    if draw_keypoints:
        output_video_frames = drawer.draw_keypoints(output_video_frames, keypoints)

    output_video_frames = drawer.draw_2d_map(
        output_video_frames,
        football_field,
        team_assigner,
        players_team_1_to_draw,
        players_team_2_to_draw,
        referees_to_draw,
        balls_to_draw,
    )

    output_video_frames = drawer.draw_possession_percentages(
        output_video_frames, team_ball_control
    )

    # Save video to a temporary file
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    save_video(output_video_frames, temp_video_path)

    return temp_video_path


def main():
    st.title("Football Tactical Analysis")

    camera_view = st.selectbox("Camera View", ["TV", "Tactical"])

    config_dict = load_config(camera_view)

    # UI elements to choose parameters
    confidence_threshold = st.slider(
        "Confidence Threshold", 0.0, 1.0, config_dict.get("confidence_threshold", 0.5)
    )
    iou_threshold = st.slider(
        "IOU Threshold", 0.0, 1.0, config_dict.get("iou_threshold", 0.5)
    )
    interpolate_ball = st.checkbox("Interpolate Ball Position", value=True)
    draw_keypoints = st.checkbox("Draw Keypoints", value=False)

    # Display demo video
    st.video(config_dict["test_video_path"])
    use_demo_video = st.checkbox("Use Demo Video", value=True)

    video_file = None
    if not use_demo_video:
        video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    # Button to process video
    if st.button("Process Video"):
        if use_demo_video:
            video_path = config_dict["test_video_path"]
            read_from_cache = True
        else:
            read_from_cache = False
            if video_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(video_file.read())
                    video_path = tmp.name
            else:
                st.error("Please upload a video.")
                return

        output_video_path = process_video(
            config_dict,
            video_path,
            confidence_threshold,
            iou_threshold,
            interpolate_ball,
            draw_keypoints,
            read_from_cache,
        )
        st.video(output_video_path)


if __name__ == "__main__":
    main()
