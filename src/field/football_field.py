from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import cv2
import supervision as sv


@dataclass
class FootballFieldConfiguration:
    width: int = 7000  # [cm]
    length: int = 12000  # [cm]
    penalty_box_width: int = 4100  # [cm]
    penalty_box_length: int = 2015  # [cm]
    goal_box_width: int = 1832  # [cm]
    goal_box_length: int = 550  # [cm]
    centre_circle_radius: int = 915  # [cm]
    penalty_spot_distance: int = 1100  # [cm]

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        return [
            (0, 0),  # 1
            (0, (self.width - self.penalty_box_width) / 2),  # 2
            (0, (self.width - self.goal_box_width) / 2),  # 3
            (0, (self.width + self.goal_box_width) / 2),  # 4
            (0, (self.width + self.penalty_box_width) / 2),  # 5
            (0, self.width),  # 6
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8
            (self.penalty_spot_distance, self.width / 2),  # 9
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13
            (self.length / 2, 0),  # 14
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16
            (self.length / 2, self.width),  # 17
            (
                self.length - self.penalty_box_length,
                (self.width - self.penalty_box_width) / 2,
            ),  # 18
            (
                self.length - self.penalty_box_length,
                (self.width - self.goal_box_width) / 2,
            ),  # 19
            (
                self.length - self.penalty_box_length,
                (self.width + self.goal_box_width) / 2,
            ),  # 20
            (
                self.length - self.penalty_box_length,
                (self.width + self.penalty_box_width) / 2,
            ),  # 21
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22
            (
                self.length - self.goal_box_length,
                (self.width - self.goal_box_width) / 2,
            ),  # 23
            (
                self.length - self.goal_box_length,
                (self.width + self.goal_box_width) / 2,
            ),  # 24
            (self.length, 0),  # 25
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26
            (self.length, (self.width - self.goal_box_width) / 2),  # 27
            (self.length, (self.width + self.goal_box_width) / 2),  # 28
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29
            (self.length, self.width),  # 30
            (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31
            (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32
        ]

    edges: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (7, 8),
            (10, 11),
            (11, 12),
            (12, 13),
            (14, 15),
            (15, 16),
            (16, 17),
            (18, 19),
            (19, 20),
            (20, 21),
            (23, 24),
            (25, 26),
            (26, 27),
            (27, 28),
            (28, 29),
            (29, 30),
            (1, 14),
            (2, 10),
            (3, 7),
            (4, 8),
            (5, 13),
            (6, 17),
            (14, 25),
            (18, 26),
            (23, 27),
            (24, 28),
            (21, 29),
            (17, 30),
        ]
    )

    labels: List[str] = field(
        default_factory=lambda: [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "15",
            "16",
            "17",
            "18",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "14",
            "19",
        ]
    )

    colors: List[str] = field(
        default_factory=lambda: [
            "#FF1493",
            "#FF1493",
            "#FF1493",
            "#FF1493",
            "#FF1493",
            "#FF1493",
            "#FF1493",
            "#FF1493",
            "#FF1493",
            "#FF1493",
            "#FF1493",
            "#FF1493",
            "#FF1493",
            "#00BFFF",
            "#00BFFF",
            "#00BFFF",
            "#00BFFF",
            "#FF6347",
            "#FF6347",
            "#FF6347",
            "#FF6347",
            "#FF6347",
            "#FF6347",
            "#FF6347",
            "#FF6347",
            "#FF6347",
            "#FF6347",
            "#FF6347",
            "#FF6347",
            "#FF6347",
            "#00BFFF",
            "#00BFFF",
        ]
    )


class FootballField:

    def __init__(self):
        self.config = FootballFieldConfiguration()

    def draw_soccer_field(
        self,
        background_color: sv.Color = sv.Color(34, 139, 34),
        line_color: sv.Color = sv.Color.WHITE,
        padding: int = 50,
        line_thickness: int = 4,
        point_radius: int = 8,
        scale: float = 0.1,
    ) -> np.ndarray:
        """
        Draws a soccer field based on the given configuration.

        Args:
            config (SoccerFieldConfiguration): Configuration of the soccer field.
            background_color (sv.Color, optional): Background color of the field.
                Defaults to sv.Color(34, 139, 34).
            line_color (sv.Color, optional): Color of the field lines.
                Defaults to sv.Color.WHITE.
            padding (int, optional): Padding around the field. Defaults to 50.
            line_thickness (int, optional): Thickness of the field lines.
                Defaults to 4.
            point_radius (int, optional): Radius of the points. Defaults to 8.
            scale (float, optional): Scale factor for the field dimensions.
                Defaults to 0.1.

        Returns:
            np.ndarray: Image of the soccer field.
        """
        scaled_width = int(self.config.width * scale)
        scaled_length = int(self.config.length * scale)
        scaled_padding = padding
        scaled_circle_radius = int(self.config.centre_circle_radius * scale)
        scaled_penalty_spot_distance = int(self.config.penalty_spot_distance * scale)

        field_image = np.ones(
            (scaled_width + 2 * scaled_padding, scaled_length + 2 * scaled_padding, 3),
            dtype=np.uint8,
        ) * np.array(background_color.as_bgr(), dtype=np.uint8)

        for start, end in self.config.edges:
            point1 = (
                int(self.config.vertices[start - 1][0] * scale) + scaled_padding,
                int(self.config.vertices[start - 1][1] * scale) + scaled_padding,
            )
            point2 = (
                int(self.config.vertices[end - 1][0] * scale) + scaled_padding,
                int(self.config.vertices[end - 1][1] * scale) + scaled_padding,
            )
            cv2.line(
                img=field_image,
                pt1=point1,
                pt2=point2,
                color=line_color.as_bgr(),
                thickness=line_thickness,
            )

        centre_circle_center = (
            scaled_length // 2 + scaled_padding,
            scaled_width // 2 + scaled_padding,
        )
        cv2.circle(
            img=field_image,
            center=centre_circle_center,
            radius=scaled_circle_radius,
            color=line_color.as_bgr(),
            thickness=line_thickness,
        )

        penalty_spots = [
            (
                scaled_penalty_spot_distance + scaled_padding,
                scaled_width // 2 + scaled_padding,
            ),
            (
                scaled_length - scaled_penalty_spot_distance + scaled_padding,
                scaled_width // 2 + scaled_padding,
            ),
        ]
        for spot in penalty_spots:
            cv2.circle(
                img=field_image,
                center=spot,
                radius=point_radius,
                color=line_color.as_bgr(),
                thickness=-1,
            )

        return field_image

    def draw_players(
        self,
        xy: np.ndarray,
        face_color: sv.Color = sv.Color.RED,
        edge_color: sv.Color = sv.Color.BLACK,
        radius: int = 10,
        thickness: int = 2,
        padding: int = 50,
        scale: float = 0.1,
        soccer_field: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draws players on the soccer field.

        Args:
            config (SoccerFieldConfiguration): Configuration of the soccer field.
            xy (np.ndarray): Array of player positions with shape (N, 2).
            face_color (sv.Color, optional): Fill color for the players.
                Defaults to sv.Color.RED.
            edge_color (sv.Color, optional): Edge color for the players.
                Defaults to sv.Color.BLACK.
            radius (int, optional): Radius of the player circles. Defaults to 10.
            thickness (int, optional): Thickness of the edge lines. Defaults to 2.
            padding (int, optional): Padding around the field. Defaults to 50.
            scale (float, optional): Scale factor for the field dimensions.
                Defaults to 0.1.
            soccer_field (Optional[np.ndarray], optional): Pre-drawn soccer field
                map. If None, a new field is drawn. Defaults to None.

        Returns:
            np.ndarray: Image of the soccer field with players.
        """
        if soccer_field is None:
            soccer_field = self.draw_soccer_field(padding=padding, scale=scale)

        try:
            face_color = face_color.as_bgr()
        except AttributeError:
            pass

        scaled_padding = padding
        for position in xy:
            point = (
                int(position[0] * scale) + scaled_padding,
                int(position[1] * scale) + scaled_padding,
            )
            cv2.circle(
                img=soccer_field,
                center=point,
                radius=radius,
                color=face_color,
                thickness=-1,
            )
            cv2.circle(
                img=soccer_field,
                center=point,
                radius=radius,
                color=edge_color.as_bgr(),
                thickness=thickness,
            )

        return soccer_field
