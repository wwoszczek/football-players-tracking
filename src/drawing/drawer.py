import cv2
import numpy as np
from typing import List
from src.utils.utils import get_center_of_bbox, get_bbox_width, get_bbox_height


class Drawer:
    def __init__(self):
        self.rectangle_width = 40
        self.rectangle_height = 20

    def _draw_elipse_with_caption(
        self, frame: np.ndarray, bbox: List, color: tuple, track_id: int = None
    ) -> np.ndarray:
        x_center, y_center = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        height = get_bbox_height(bbox)

        cv2.ellipse(
            frame,
            center=(int(x_center), int(y_center - height // 2)),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        if track_id is not None:
            x1_rect = x_center - self.rectangle_width // 2
            x2_rect = x_center + self.rectangle_width // 2
            y1_rect = (y_center - height // 2 - self.rectangle_height // 2) + 20
            y2_rect = (y_center - height // 2 + self.rectangle_height // 2) + 20

            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_rect + 12), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def _draw_traingle(self, frame: np.ndarray, bbox: List, color: tuple) -> np.ndarray:
        y_upper = int(bbox[1])
        x_center, _ = get_center_of_bbox(bbox)

        triangle_points = np.array(
            [
                [x_center, y_upper],
                [x_center - 10, y_upper - 20],
                [x_center + 10, y_upper - 20],
            ]
        )
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control=None):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self._draw_elipse_with_caption(
                    frame, player["bbox"], color, track_id
                )

                # if player.get("has_ball", False):
                #     frame = self._draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for referee in referee_dict.values():
                frame = self._draw_elipse_with_caption(
                    frame, referee["bbox"], (0, 255, 255)
                )

            # Draw ball
            for ball in ball_dict.values():
                frame = self._draw_traingle(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames
