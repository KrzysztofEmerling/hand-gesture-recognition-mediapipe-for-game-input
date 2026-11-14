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

# One gesture sample = graphs of BOTH hands stored in a single class row
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
FEATURES_PER_HAND = LANDMARKS_PER_HAND * 2  # 21 landmarks x (x, y)
NUM_FEATURES = NUM_HANDS * FEATURES_PER_HAND  # 84


class KerasKeyPointClassifier(object):
    """Keypoint classifier backed by a full Keras model (.keras / .h5)."""

    def __init__(self, model_path):
        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path)

        input_shape = self.model.input_shape
        if len(input_shape) != 2 or input_shape[-1] != NUM_FEATURES:
            raise ValueError(
                "Unexpected input shape {} for the keypoint classifier "
                "(expected (None, {}) = {} hands x {} landmarks x 2).".format(
                    input_shape, NUM_FEATURES, NUM_HANDS, LANDMARKS_PER_HAND))

    def __call__(self, landmark_list):
        x = np.array([landmark_list], dtype=np.float32)
        scores = self.model(x, training=False)
        return int(np.argmax(np.squeeze(scores.numpy())))


def load_keypoint_classifier(model_path):
    """Load the gesture classifier; .keras/.h5 use Keras, others use TFLite."""
    if model_path.lower().endswith(('.keras', '.h5', '.hdf5')):
        return KerasKeyPointClassifier(model_path)

    classifier = KeyPointClassifier(model_path=model_path)
    input_shape = classifier.input_details[0]['shape']
    if int(input_shape[-1]) != NUM_FEATURES:
        raise ValueError(
            "Keypoint classifier {} expects {} features, but the model was "
            "built for {} (retrain it with keypoint_classification_EN.ipynb "
            "on two-hand data).".format(
                model_path, NUM_FEATURES, int(input_shape[-1])))
    return classifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_hands", type=int, default=2)

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
    parser.add_argument("--no_classification",
                        help='run WITHOUT the keypoint classifier (hand '
                             'detection + sample recording only)',
                        action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--height_diff_threshold",
                        help='min allowed vertical distance (px) between the '
                             'two hands; a smaller difference forces the '
                             'gesture to inactive (0 = disabled)',
                        type=int,
                        default=100)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    number_of_hands = args.num_hands
    if number_of_hands < NUM_HANDS:
        print("--num_hands {} < {}: a gesture sample needs {} hands, "
              "raising to {}.".format(number_of_hands, NUM_HANDS, NUM_HANDS,
                                      NUM_HANDS))
        number_of_hands = NUM_HANDS
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    model_path = args.model_path
    classifier_path = args.classifier_path
    labels_path = args.labels_path
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    height_diff_threshold = args.height_diff_threshold

    # --no_classification: run the app WITHOUT the keypoint classifier
    use_classifier = not args.no_classification

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    hand_landmarker_options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=number_of_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(
        hand_landmarker_options)

    if use_classifier:
        if not os.path.isfile(classifier_path):
            raise SystemExit(
                "Keypoint classifier not found: {} "
                "(train it first with keypoint_classification_EN.ipynb, or "
                "run with --no_classification to start without it).".format(
                    classifier_path))
        keypoint_classifier = load_keypoint_classifier(classifier_path)
        print("Keypoint classifier loaded: {}".format(classifier_path))
    else:
        keypoint_classifier = None
        print("--no_classification: running WITHOUT the keypoint "
              "classifier (detection + recording only).")

    # Read labels ###########################################################
    with open(labels_path, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [
            row[0] for row in csv.reader(f) if row
        ]

    # Gesture classes are numbered 1..N (0 = not chosen yet)
    num_categories = len(keypoint_classifier_labels)
    if num_categories < 1:
        raise SystemExit("No labels found in {}".format(labels_path))

    # Label used whenever fewer than two hands are visible
    inactive_label = ('inactive'
                      if 'inactive' in keypoint_classifier_labels
                      else keypoint_classifier_labels[-1])

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
    category = 0            # movement category 1-N (0 = not chosen yet)
    prev_bumper = False     # bumper edge detection
    prev_hat = (0, 0)       # d-pad edge detection
    capture_pending = False # one-frame capture requested by bumper
    captured_count = 0      # frames saved since start
    capture_flash_until = 0.0
    capture_missing_hand_until = 0.0

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
        if 1 <= number <= 9:  # keyboard fallback for category selection
            category = min(number, num_categories)

        # Gamepad input ##########################################################
        if gamepad is not None:
            pygame.event.pump()

            # D-pad up/down: choose movement category (remembered + shown)
            hat = gamepad.get_hat(0)
            if hat[1] == 1 and prev_hat[1] != 1:
                category = (1 if category >= num_categories
                            else category + 1)
            elif hat[1] == -1 and prev_hat[1] != -1:
                category = (num_categories if category <= 1
                            else category - 1)
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
        # Collect every detected hand first, then treat them as ONE sample:
        # one gesture = two hand graphs (84 features) stored in one class.
        hands = []
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

                hands.append({
                    'handedness': handedness[0].category_name,
                    'brect': brect,
                    'landmark_list': landmark_list,
                    'pre_processed': pre_processed_landmark_list,
                    # Vertical position of the hand (mean y of landmarks)
                    'height': sum(p[1] for p in landmark_list)
                              / len(landmark_list),
                })

        # Deterministic hand order: "Left" before "Right" (alphabetical),
        # ties broken by wrist x so the feature vector never swaps places.
        hands.sort(key=lambda hand: (
            hand['handedness'],
            hand['landmark_list'][0][0],
        ))

        # Two hand graphs concatenated into a single class sample
        combined_landmark_list = []
        for hand in hands[:NUM_HANDS]:
            combined_landmark_list.extend(hand['pre_processed'])

        both_hands_visible = len(hands) >= NUM_HANDS

        # Vertical distance between BOTH hands (px, mean landmark y).
        # A bigger difference than the threshold -> gesture forced inactive.
        if both_hands_visible:
            height_diff = abs(hands[0]['height'] - hands[1]['height'])
            hands_at_same_height = height_diff >= height_diff_threshold
        else:
            height_diff = 0.0
            hands_at_same_height = False

        # Write to the dataset file
        # One frame per bumper press, class id = category - 1 (0-4)
        # Requires BOTH hands so every row keeps the same 84-feature shape.
        if capture_pending and mode == 1 and 1 <= category <= num_categories:
            if both_hands_visible:
                logging_csv(category - 1, combined_landmark_list)
                capture_pending = False
                captured_count += 1
                capture_flash_until = time.monotonic() + 0.6
                print("Saved frame -> category {} (class {}) (total {})"
                      .format(category, category - 1, captured_count))
            else:
                capture_missing_hand_until = time.monotonic() + 0.6

        # Hand sign classification (needs both hand graphs at similar height)
        if not use_classifier:
            hand_sign_text = inactive_label if not both_hands_visible \
                else 'classification off'
        elif both_hands_visible and hands_at_same_height:
            hand_sign_id = int(keypoint_classifier(combined_landmark_list))
            hand_sign_text = (
                keypoint_classifier_labels[hand_sign_id]
                if hand_sign_id < len(keypoint_classifier_labels)
                else str(hand_sign_id))
        else:
            # Not enough hands, or hands too far apart vertically
            # -> inactive (no real classification)
            hand_sign_text = inactive_label

        # Drawing part
        for hand in hands:
            debug_image = draw_bounding_rect(use_brect, debug_image,
                                             hand['brect'])
            debug_image = draw_landmarks(debug_image, hand['landmark_list'])
            debug_image = draw_info_text(
                debug_image,
                hand['brect'],
                hand['handedness'],
                hand_sign_text
                if both_hands_visible and use_classifier else '',
            )

        debug_image = draw_info(debug_image, fps, mode, number, category,
                                captured_count,
                                time.monotonic() < capture_flash_until,
                                capture_pending,
                                hand_sign_text,
                                time.monotonic() < capture_missing_hand_until,
                                both_hands_visible,
                                keypoint_classifier_labels,
                                hand_heights=[hand['height'] for hand in
                                              hands[:NUM_HANDS]],
                                height_diff=height_diff,
                                height_threshold=height_diff_threshold,
                                hands_at_same_height=hands_at_same_height)

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
    if len(landmark_list) != NUM_FEATURES:
        raise ValueError(
            "Refusing to save sample with {} features (expected {} = "
            "{} hands x {} landmarks x 2).".format(
                len(landmark_list), NUM_FEATURES, NUM_HANDS,
                LANDMARKS_PER_HAND))
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
              capture_pending, gesture_text, missing_hand, both_hands_visible,
              category_labels=None, hand_heights=None, height_diff=0.0,
              height_threshold=0, hands_at_same_height=True):
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

    # Movement category (1-N), remembered between frames
    if category > 0:
        category_text = "CATEGORY:" + str(category)
        if category_labels and category <= len(category_labels):
            category_text += " " + category_labels[category - 1]
        cv.putText(image, category_text, (10, 140),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, category_text, (10, 140),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
                   cv.LINE_AA)
    else:
        cv.putText(image, "CATEGORY: - (d-pad up/down)", (10, 140),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "CATEGORY: - (d-pad up/down)", (10, 140),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)

    # Vertical position of both hands + distance between them
    if hand_heights:
        height_text = "HANDS Y:" + "/".join("{:.0f}".format(h)
                                             for h in hand_heights)
        if both_hands_visible:
            height_text += "  dY:{:.0f}px".format(height_diff)
            if height_threshold > 0:
                height_text += "/{}".format(height_threshold)
        cv.putText(image, height_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 0, 0), 4, cv.LINE_AA)
        same_height = (not both_hands_visible) or hands_at_same_height
        cv.putText(image, height_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255) if same_height else (0, 165, 255), 1,
                   cv.LINE_AA)

    # Recognized gesture (both hands combined into one sample)
    gesture_line = "GESTURE:" + gesture_text
    cv.putText(image, gesture_line, (10, 175), cv.FONT_HERSHEY_SIMPLEX, 0.9,
               (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, gesture_line, (10, 175), cv.FONT_HERSHEY_SIMPLEX, 0.9,
               (0, 255, 0) if both_hands_visible and hands_at_same_height
               else (0, 165, 255), 2,
               cv.LINE_AA)

    # Capture status
    if just_saved:
        cv.putText(image, "SAVED!", (10, 210), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                   (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "SAVED!", (10, 210), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                   (0, 255, 0), 2, cv.LINE_AA)
    elif capture_pending and mode == 1:
        if missing_hand:
            waiting_text = "both hands required!"
        else:
            waiting_text = "waiting for both hands..."
        cv.putText(image, waiting_text, (10, 210),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, waiting_text, (10, 210),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
                   cv.LINE_AA)

    if captured_count > 0:
        cv.putText(image, "SAVED FRAMES:" + str(captured_count), (10, 240),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "SAVED FRAMES:" + str(captured_count), (10, 240),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()
