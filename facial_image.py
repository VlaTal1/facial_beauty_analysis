from math import dist

import cv2
import mediapipe as mp

from points import points
from ratios import Ratios
from utils import *


def convert_to_numpy_array(ratios: dict[str, float]) -> numpy.array:
    arr = []
    for k, v in ratios.items():
        arr.append(v)
    return numpy.array(arr)


class FacialImage:
    def __init__(self):
        self.landmarks = []
        self.width = 0
        self.height = 0

    def get_point_coordinates(self, point_num):
        point = self.landmarks[point_num]
        return point.x * self.width, point.y * self.height

    def get_distance_two_points(self, point_num_1, point_num_2):
        first = self.get_point_coordinates(point_num_1)
        second = self.get_point_coordinates(point_num_2)
        return dist(first, second)

    def get_ratio_between_two(self, ratio1_name, ratio2_name):
        first = self.get_distance_two_points(points[ratio1_name][0], points[ratio1_name][1])
        second = self.get_distance_two_points(points[ratio2_name][0], points[ratio2_name][1])
        return first / second

    def get_landmarks(self, image: numpy.ndarray):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()

        # Image
        height, width, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Facial landmarks
        result = face_mesh.process(rgb_image)

        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

        self.landmarks = result.multi_face_landmarks[0].landmark
        self.width = width
        self.height = height

    def calculate_ratios(self, image: numpy.ndarray, is_normalized=False):
        self.get_landmarks(image)

        ratios = {
            'Under eyes/Interocular': self.get_ratio_between_two(Ratios.UNDER_EYES, Ratios.INTEROCULAR),
            'Under eyes/Nose width': self.get_ratio_between_two(Ratios.UNDER_EYES, Ratios.NOSE_WIDTH),
            'Mouth width/Interocular': self.get_ratio_between_two(Ratios.MOUTH_WIDTH, Ratios.INTEROCULAR),
            'Upper lip-jaw/Interocular': self.get_ratio_between_two(Ratios.UPPER_LIP_TO_JAW, Ratios.INTEROCULAR),
            'Upper lip-jaw/Nose width': self.get_ratio_between_two(Ratios.UPPER_LIP_TO_JAW, Ratios.NOSE_WIDTH),
            'Interocular/Lip height': self.get_ratio_between_two(Ratios.INTEROCULAR, Ratios.LIPS_HEIGHT),
            'Nose width/Interocular': self.get_ratio_between_two(Ratios.NOSE_WIDTH, Ratios.INTEROCULAR),
            'Nose width/Upper lip height': self.get_ratio_between_two(Ratios.NOSE_WIDTH, Ratios.UPPER_LIP_HEIGHT),
            'Interocular/Nose mouth height': self.get_ratio_between_two(Ratios.INTEROCULAR,
                                                                        Ratios.NOSE_TO_MOUTH_HEIGHT),
            'Face top-eyebrows/Eyebrows-Nose': self.get_ratio_between_two(Ratios.FACE_TOP_TO_EYEBROWS,
                                                                          Ratios.EYEBROWS_TO_NOSE),
            'Eyebrows-nose/Nose-jaw': self.get_ratio_between_two(Ratios.EYEBROWS_TO_NOSE, Ratios.NOSE_TO_JAW),
            'Face top-eyebrows/Nose-Jaw': self.get_ratio_between_two(Ratios.FACE_TOP_TO_EYEBROWS, Ratios.NOSE_TO_JAW),
            'Interocular/Nose width': self.get_ratio_between_two(Ratios.INTEROCULAR, Ratios.NOSE_WIDTH),
            'Face height/Face width': self.get_ratio_between_two(Ratios.FACE_HEIGHT, Ratios.FACE_WIDTH),
            'Lower eyebrow length': self.get_ratio_between_two(Ratios.LEFT_LOWER_EYEBROW_LENGTH,
                                                               Ratios.RIGHT_LOWER_EYEBROW_LENGTH),
            'Lower lip length': self.get_ratio_between_two(Ratios.LEFT_LOWER_LIP_LENGTH, Ratios.RIGHT_LOWER_LIP_LENGTH),
            'Upper eyebrow': self.get_ratio_between_two(Ratios.LEFT_UPPER_EYEBROW_LENGTH,
                                                        Ratios.RIGHT_UPPER_EYEBROW_LENGTH),
            'Upper lip': self.get_ratio_between_two(Ratios.LEFT_UPPER_LIP_LENGTH, Ratios.RIGHT_UPPER_LIP_LENGTH),
            'Nose': self.get_ratio_between_two(Ratios.LEFT_NOSE_WIDTH, Ratios.RIGHT_NOSE_WIDTH)
        }

        if is_normalized:
            return normalization(ratios), image

        return ratios, image
