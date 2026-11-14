#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import os
import time

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from utils import CvFpsCalc
from model import KeyPointClassifier

try:
    import pygame
except ImportError:
    pygame = None

# Gamepad buttons (SDL/Xbox mapping): LB=4, RB=5
BUMPER_BUTTONS = (4, 5)

class KerasKeyPointClassifier(object):
    """Keypoint classifier backed by a full Keras model (.keras / .h5)."""

    def __init__(self, model_path):
        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path)

        input_shape = self.model.input_shape
        if len(input_shape) != 2 or input_shape[-1] != 42:
            raise ValueError(
                "Unexpected input shape {} for the keypoint classifier "
                "(expected (None, 42) = 21 landmarks x 2 coordinates).".format(
                    input_shape))

    def __call__(self, landmark_list):
        x = np.array([landmark_list], dtype=np.float32)
        scores = self.model(x, training=False)
        return int(np.argmax(np.squeeze(scores.numpy())))


def load_keypoint_classifier(model_path):
    """Load the gesture classifier; .keras/.h5 use Keras, others use TFLite."""
    if model_path.lower().endswith(('.keras', '.h5', '.hdf5')):
        return KerasKeyPointClassifier(model_path)
    return KeyPointClassifier(model_path=model_path)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--model_path",
                        help='hand landmarker .task model path',
                        default='model/hand_landmarker.task')

    parser.add_argument("--classifier_path",
                        help='keypoint classifier model (.tflite or .keras)',
                        default='model/keypoint_classifier/'
                               'keypoint_classifier.tflite')
    parser.add_argument("--labels_path",
                        help='gesture labels csv path',
                        default='model/keypoint_classifier/'
                                'keypoint_classifier_label.csv')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    model_path = args.model_path
    classifier_path = args.classifier_path
    labels_path = args.labels_path
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    hand_landmarker_options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(
        hand_landmarker_options)

    if not os.path.isfile(classifier_path):
        raise SystemExit(
            "Keypoint classifier not found: {} "
            "(train it first with keypoint_classification_EN.ipynb).".format(
                classifier_path))
    keypoint_classifier = load_keypoint_classifier(classifier_path)
    print("Keypoint classifier loaded: {}".format(classifier_path))

    # Read labels ###########################################################
    with open(labels_path, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [
            row[0] for row in csv.reader(f)
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Gamepad ####################################################
    gamepad = None
    if pygame is not None:
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            gamepad = pygame.joystick.Joystick(0)
            print("Gamepad connected: {}".format(gamepad.get_name()))
        else:
            print("No gamepad detected - keyboard control only.")
    elif pygame is None:
        print("pygame not installed - keyboard control only.")

    mode = 0
    category = 0            # movement category 1-4 (0 = not chosen yet)
    prev_bumper = False     # bumper edge detection
    prev_hat = (0, 0)       # d-pad edge detection
    capture_pending = False # one-frame capture requested by bumper
    captured_count = 0      # frames saved since start
    capture_flash_until = 0.0

    # Timestamps for the VIDEO running mode must be monotonically increasing
    start_time = time.monotonic()
    last_timestamp_ms = -1

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)
        if 1 <= number <= 4:  # keyboard fallback for category selection
            category = number

        # Gamepad input ##########################################################
        if gamepad is not None:
            pygame.event.pump()

            # D-pad up/down: choose movement category 1-4 (remembered + shown)
            hat = gamepad.get_hat(0)
            if hat[1] == 1 and prev_hat[1] != 1:
                category = 1 if category >= 4 else category + 1
            elif hat[1] == -1 and prev_hat[1] != -1:
                category = 4 if category <= 1 else category - 1
            prev_hat = hat

            # Bumper press (LB or RB): capture exactly one frame
            bumper = any(
                gamepad.get_button(b) for b in BUMPER_BUTTONS
                if b < gamepad.get_numbuttons())
            if bumper and not prev_bumper:
                if mode == 1:
                    capture_pending = True
                else:
                    print("Bumper pressed, but mode is not 1 "
                          "(press 'k' to start recording).")
            prev_bumper = bumper

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        #  ####################################################################
        if results.hand_landmarks:
            for hand_landmarks, handedness in zip(results.hand_landmarks,
                                                  results.handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

                # Write to the dataset file
                # One frame per bumper press, class id = category - 1 (0-3)
                if capture_pending and mode == 1 and 1 <= category <= 4:
                    logging_csv(category - 1, pre_processed_landmark_list)
                    capture_pending = False
                    captured_count += 1
                    capture_flash_until = time.monotonic() + 0.6
                    print("Saved frame -> category {} (class {}) (total {})"
                          .format(category, category - 1, captured_count))

                # Hand sign classification
                hand_sign_id = int(keypoint_classifier(
                    pre_processed_landmark_list))
                hand_sign_text = (
                    keypoint_classifier_labels[hand_sign_id]
                    if hand_sign_id < len(keypoint_classifier_labels)
                    else str(hand_sign_id))

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness[0].category_name,
                    hand_sign_text,
                )

        debug_image = draw_info(debug_image, fps, mode, number, category,
                                captured_count,
                                time.monotonic() < capture_flash_until,
                                capture_pending)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    hand_landmarker.close()
    cv.destroyAllWindows()
    if gamepad is not None:
        gamepad.quit()
    if pygame is not None:
        pygame.quit()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def logging_csv(category, landmark_list):
    csv_path = 'model/keypoint_classifier/keypoint.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([category, *landmark_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        radius = 8 if index in (4, 8, 12, 16, 20) else 5  # fingertips are bigger
        cv.circle(image, (landmark[0], landmark[1]), radius, (255, 255, 255),
                  -1)
        cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness_label, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness_label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_info(image, fps, mode, number, category, captured_count, just_saved,
              capture_pending):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    if mode == 1:
        cv.putText(image, "MODE:Logging Key Point", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)

    # Movement category (1-4), remembered between frames
    if category > 0:
        cv.putText(image, "CATEGORY:" + str(category), (10, 140),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "CATEGORY:" + str(category), (10, 140),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
                   cv.LINE_AA)
    else:
        cv.putText(image, "CATEGORY: - (d-pad up/down)", (10, 140),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "CATEGORY: - (d-pad up/down)", (10, 140),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)

    # Capture status
    if just_saved:
        cv.putText(image, "SAVED!", (10, 175), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                   (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "SAVED!", (10, 175), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                   (0, 255, 0), 2, cv.LINE_AA)
    elif capture_pending and mode == 1:
        cv.putText(image, "waiting for hand...", (10, 175),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "waiting for hand...", (10, 175),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
                   cv.LINE_AA)

    if captured_count > 0:
        cv.putText(image, "SAVED FRAMES:" + str(captured_count), (10, 205),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "SAVED FRAMES:" + str(captured_count), (10, 205),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()
