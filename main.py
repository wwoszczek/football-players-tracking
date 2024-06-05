from src.utils.utils import read_video, save_video
from src.tracking.tracker import Tracker
from src.drawing.drawer import Drawer
import yaml

with open("config.yaml", "r") as tracking_config_stream:
    CONFIG_DICT = yaml.safe_load(tracking_config_stream)

if __name__ == "__main__":
    # read video
    video_frames = read_video(CONFIG_DICT["test_video_path"])
    print(f"Frames to process: {len(video_frames)}")

    # Initialize Tracker
    tracker = Tracker(CONFIG_DICT["players_model_path"])
    tracks = tracker.get_object_tracks(video_frames, read_from_cache=False)

    # Draw output video frames
    drawer = Drawer()
    output_video_frames = drawer.draw_annotations(video_frames, tracks)

    # save video
    save_video(
        output_video_frames,
        f"{CONFIG_DICT['output_video_path']}/tracking_system_output.mp4",
    )
