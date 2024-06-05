import cv2
from typing import List


def read_video(video_path: str) -> List:
    """
    :param video_path: path of a vide to read
    :return: list of frames from the video of interest
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def save_video(output_video_frames, output_video_path, fps=24):
    """
    Save a sequence of frames as a video file in .mp4 format.

    Parameters:
    - output_video_frames: List of frames to be saved as a video.
    - output_video_path: The path where the output video will be saved.
    - fps: Frames per second for the output video (default is 24).
    """
    if not output_video_frames:
        raise ValueError("The output_video_frames list is empty")

    # Get the height and width of the frames
    height, width, _ = output_video_frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height),
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    return bbox[2] - bbox[0]


def get_bbox_height(bbox):
    return bbox[3] - bbox[1]


def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
