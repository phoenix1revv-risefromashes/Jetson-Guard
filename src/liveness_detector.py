import math
import time

import face_recognition


class LivenessDetector:
    def __init__(self):
        self.eye_closed_threshold = 0.19
        self.eye_open_threshold = 0.24

        self.required_open_frames_before_blink = 5
        self.required_closed_frames = 3
        self.required_open_frames_after_blink = 4

        self.spoof_timeout_seconds = 3.0
        self.live_window_seconds = 3.0

        self.challenge_start_time = None
        self.live_until_time = 0

        self.open_before_count = 0
        self.closed_count = 0
        self.open_after_count = 0

        self.state = "waiting_for_open_eyes"

    def reset(self):
        self.challenge_start_time = None
        self.live_until_time = 0

        self.open_before_count = 0
        self.closed_count = 0
        self.open_after_count = 0

        self.state = "waiting_for_open_eyes"

    def distance(self, point_a, point_b):
        return math.sqrt(
            (point_a[0] - point_b[0]) ** 2
            + (point_a[1] - point_b[1]) ** 2
        )

    def eye_aspect_ratio(self, eye_points):
        if len(eye_points) < 6:
            return None

        p1, p2, p3, p4, p5, p6 = eye_points[:6]

        vertical_1 = self.distance(p2, p6)
        vertical_2 = self.distance(p3, p5)
        horizontal = self.distance(p1, p4)

        if horizontal == 0:
            return None

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def get_average_eye_ratio(self, face_crop_rgb):
        landmarks_list = face_recognition.face_landmarks(face_crop_rgb)

        if len(landmarks_list) == 0:
            return None

        landmarks = landmarks_list[0]

        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")

        if left_eye is None or right_eye is None:
            return None

        left_eye_ratio = self.eye_aspect_ratio(left_eye)
        right_eye_ratio = self.eye_aspect_ratio(right_eye)

        if left_eye_ratio is None or right_eye_ratio is None:
            return None

        return (left_eye_ratio + right_eye_ratio) / 2.0

    def update(self, face_crop_rgb):
        current_time = time.monotonic()

        if current_time < self.live_until_time:
            return True, "liveness_confirmed"

        if self.challenge_start_time is None:
            self.challenge_start_time = current_time

        elapsed_time = current_time - self.challenge_start_time

        if elapsed_time >= self.spoof_timeout_seconds:
            return False, "spoof_detected"

        eye_ratio = self.get_average_eye_ratio(face_crop_rgb)

        if eye_ratio is None:
            return False, "face_not_clear"

        eyes_are_open = eye_ratio >= self.eye_open_threshold
        eyes_are_closed = eye_ratio <= self.eye_closed_threshold

        if self.state == "waiting_for_open_eyes":
            if eyes_are_open:
                self.open_before_count += 1
            else:
                self.open_before_count = 0

            if self.open_before_count >= self.required_open_frames_before_blink:
                self.state = "waiting_for_closed_eyes"

            return False, "blink_required"

        if self.state == "waiting_for_closed_eyes":
            if eyes_are_closed:
                self.closed_count += 1
            else:
                self.closed_count = 0

            if self.closed_count >= self.required_closed_frames:
                self.state = "waiting_for_reopen_eyes"

            return False, "blink_required"

        if self.state == "waiting_for_reopen_eyes":
            if eyes_are_open:
                self.open_after_count += 1
            else:
                self.open_after_count = 0

            if self.open_after_count >= self.required_open_frames_after_blink:
                self.live_until_time = current_time + self.live_window_seconds

                self.challenge_start_time = None
                self.open_before_count = 0
                self.closed_count = 0
                self.open_after_count = 0
                self.state = "waiting_for_open_eyes"

                return True, "blink_detected"

            return False, "eyes_closed"

        return False, "blink_required"