import cv2
import numpy as np
from typing import List
import supervision as sv

from src.field.football_field import FootballField
from src.team_assigner.team_assigner import TeamAssigner
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
            center=(int(x_center), int(y_center + height // 2)),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_8,
        )

        if track_id is not None:
            x1_rect = x_center - self.rectangle_width // 2
            x2_rect = x_center + self.rectangle_width // 2
            y1_rect = (y_center + height // 2 - self.rectangle_height // 2) + 20
            y2_rect = (y_center + height // 2 + self.rectangle_height // 2) + 20

            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )
            padding = 2 if track_id >= 100 else 12

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_rect + padding), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_DUPLEX,
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

    def _draw_triangle_xy_with_caption(
        self, frame: np.ndarray, xy: List, caption: str = None
    ) -> np.ndarray:
        """
        mainly for drawing keypoints with the labels to check if the order has to be changed
        """
        x, y = xy
        x, y = int(x), int(y)

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ]
        )

        cv2.drawContours(frame, [triangle_points], 0, (255, 255, 255), cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        cv2.putText(
            frame,
            f"{caption}",
            (int(x), int(y - 25)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        return frame

    def draw_annotations(self, video_frames, tracks):
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

                if player.get("has_ball", False):
                    frame = self._draw_traingle(
                        frame, player["bbox"], player["team_color"]
                    )

            # Draw Referee
            for referee in referee_dict.values():
                frame = self._draw_elipse_with_caption(
                    frame, referee["bbox"], (0, 255, 255)
                )

            # Draw ball
            for ball in ball_dict.values():
                frame = self._draw_traingle(frame, ball["bbox"], (191, 126, 58))

            output_video_frames.append(frame)

        return output_video_frames

    def draw_keypoints(self, video_frames, keypoints):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            keypoints_dict = keypoints["keypoints"][frame_num]

            # Draw keypoints
            for keypoint_id, keypoint in keypoints_dict.items():
                frame = self._draw_triangle_xy_with_caption(
                    frame, keypoint, caption=keypoint_id
                )

            output_video_frames.append(frame)

        return output_video_frames

    def draw_2d_map(
        self,
        frames,
        football_field: FootballField,
        team_asigner: TeamAssigner,
        players_team_1_to_draw,
        players_team_2_to_draw,
        referees_to_draw,
        balls_to_draw,
        alpha=0.6,
    ):

        output_video_frames = list()
        for frame_num, frame in enumerate(frames):
            field_2d_frame = football_field.draw_players(
                players_team_1_to_draw[frame_num],
                radius=25,
                face_color=team_asigner.team_colors[1],
            )
            field_2d_frame = football_field.draw_players(
                players_team_2_to_draw[frame_num],
                soccer_field=field_2d_frame,
                face_color=team_asigner.team_colors[2],
                radius=25,
            )
            field_2d_frame = football_field.draw_players(
                referees_to_draw[frame_num],
                soccer_field=field_2d_frame,
                face_color=sv.Color.YELLOW,
                radius=20,
            )
            field_2d_frame = football_field.draw_players(
                balls_to_draw[frame_num],
                soccer_field=field_2d_frame,
                face_color=sv.Color.WHITE,
                radius=15,
            )

            # Resize the field_2d_frame to half its size
            small_height, small_width, _ = field_2d_frame.shape
            field_2d_frame_resized = cv2.resize(
                field_2d_frame, (int(small_width / 2.5), int(small_height / 2.5))
            )

            # Get dimensions of the images
            large_height, large_width, _ = frame.shape
            resized_height, resized_width, _ = field_2d_frame_resized.shape

            # Calculate the position to place the small image (bottom middle)
            x_offset = (large_width - resized_width) // 2
            y_offset = large_height - resized_height

            frame[
                y_offset : y_offset + resized_height,
                x_offset : x_offset + resized_width,
            ] = cv2.addWeighted(
                field_2d_frame_resized,
                alpha,
                frame[
                    y_offset : y_offset + resized_height,
                    x_offset : x_offset + resized_width,
                ],
                1 - alpha,
                0,
            )

            output_video_frames.append(frame)

        return output_video_frames

    def _update_possession_percentages(self, frame_num, frame, team_ball_control):
        height, width, _ = frame.shape

        # Calculate the positions based on frame size
        left_x1, left_y1 = int(width * 0.05), int(height * 0.85)
        left_x2, left_y2 = int(width * 0.35), int(height * 0.95)
        right_x1, right_y1 = int(width * 0.65), int(height * 0.85)
        right_x2, right_y2 = int(width * 0.95), int(height * 0.95)

        overlay = frame.copy()
        cv2.rectangle(
            overlay, (left_x1, left_y1), (left_x2, left_y2), (255, 255, 255), -1
        )
        cv2.rectangle(
            overlay, (right_x1, right_y1), (right_x2, right_y2), (255, 255, 255), -1
        )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[: frame_num + 1]
        # Get the number of times each team had ball control
        team_1_num_frames = team_ball_control_till_frame[
            team_ball_control_till_frame == 1
        ].shape[0]
        team_2_num_frames = team_ball_control_till_frame[
            team_ball_control_till_frame == 2
        ].shape[0]
        total_frames = team_1_num_frames + team_2_num_frames
        team_1 = team_1_num_frames / total_frames
        team_2 = team_2_num_frames / total_frames

        percentage_text_team1 = f"{int(team_1 * 100)}%"
        percentage_text_team2 = f"{int(team_2 * 100)}%"

        # Center the text within the rectangles
        text_size_team1 = cv2.getTextSize(
            f"Team 1: {percentage_text_team1}", cv2.FONT_HERSHEY_DUPLEX, 2, 8
        )[0]
        text_x_team1 = left_x1 + (left_x2 - left_x1 - text_size_team1[0]) // 2
        text_y_team1 = left_y1 + (left_y2 - left_y1 + text_size_team1[1]) // 2

        text_size_team2 = cv2.getTextSize(
            f"Team 1: {percentage_text_team1}", cv2.FONT_HERSHEY_DUPLEX, 2, 8
        )[0]
        text_x_team2 = right_x1 + (right_x2 - right_x1 - text_size_team2[0]) // 2
        text_y_team2 = right_y1 + (right_y2 - right_y1 + text_size_team2[1]) // 2

        # Draw the text with white color and thicker black edges
        cv2.putText(
            frame,
            f"Team 1: {percentage_text_team2}",
            (text_x_team1, text_y_team1),
            cv2.FONT_HERSHEY_DUPLEX,
            2,
            (0, 0, 0),
            10,
        )
        cv2.putText(
            frame,
            f"Team 1: {percentage_text_team2}",
            (text_x_team1, text_y_team1),
            cv2.FONT_HERSHEY_DUPLEX,
            2,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Team 2: {percentage_text_team1}",
            (text_x_team2, text_y_team2),
            cv2.FONT_HERSHEY_DUPLEX,
            2,
            (0, 0, 0),
            10,
        )
        cv2.putText(
            frame,
            f"Team 2: {percentage_text_team1}",
            (text_x_team2, text_y_team2),
            cv2.FONT_HERSHEY_DUPLEX,
            2,
            (255, 255, 255),
            2,
        )

        return frame

    def draw_possession_percentages(self, frames, team_ball_control):
        output_frames = list()
        for frame_num, frame in enumerate(frames):
            output_frames.append(
                self._update_possession_percentages(frame_num, frame, team_ball_control)
            )

        return output_frames
