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

with open("config.yaml", "r") as tracking_config_stream:
    CONFIG_DICT = yaml.safe_load(tracking_config_stream)

if __name__ == "__main__":
    # read video
    video_frames = read_video(CONFIG_DICT["test_video_path"])
    print(f"Frames to process: {len(video_frames)}")

    # Initialize Tracker
    tracker = Tracker(CONFIG_DICT["players_model_path"])
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_cache=True,
        cache_path=CONFIG_DICT["tracking_cache_path"],
    )

    # Interpolate ball detections
    interpolator = Interpolator()
    tracks["ball"] = interpolator.interpolate_ball_positions(tracks["ball"])

    # Initialize keypoints detector
    keypoints_detector = KeypointsDetector(CONFIG_DICT["keypoints_model_path"])
    keypoints = keypoints_detector.get_keypoints_positions(
        video_frames,
        read_from_cache=True,
        cache_path=CONFIG_DICT["keypoints_cache_path"],
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
    # output_video_frames = drawer.draw_keypoints(output_video_frames, keypoints)
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

    # save video
    save_video(
        output_video_frames,
        CONFIG_DICT["output_video_path"],
    )
