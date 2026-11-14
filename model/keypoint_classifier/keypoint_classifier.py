#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Klasyfikacja gestów dłoni z użyciem landmarków MediaPipe.

Moduł zawiera:

* ``KeyPointClassifier`` - klasyfikator TFLite,
* ``KerasKeyPointClassifier`` - klasyfikator oparty o model Keras,
* ``TwoHandGestureClassifier`` - warstwę wysokopoziomową, która składa
  landmarki obu dłoni w jeden wektor cech i pilnuje progu pionowego
  rozstawienia dłoni (``height_diff_threshold``).
"""

import csv
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

# Jeden gest = jeden wektor cech z OBU dłoni.
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
FEATURES_PER_HAND = LANDMARKS_PER_HAND * 2  # 21 punktów x (x, y)
NUM_FEATURES = NUM_HANDS * FEATURES_PER_HAND  # 84

# Domyślny minimalny odstęp pionowy (px) między dłońmi. Gdy dłonie są
# bliżej siebie niż ten próg, gest jest traktowany jako nieaktywny.
# Wartość 0 wyłącza sprawdzanie progu.
DEFAULT_HEIGHT_DIFF_THRESHOLD = 60

INACTIVE_GESTURE_LABEL = 'inactive'
CLASSIFICATION_OFF_LABEL = 'classification off'


def load_labels(labels_path):
    """Wczytuje etykiety gestów z pliku CSV (pierwsza kolumna)."""
    with open(labels_path, encoding='utf-8-sig') as labels_file:
        labels = [
            row[0].strip()
            for row in csv.reader(labels_file)
            if row and row[0].strip()
        ]

    if not labels:
        raise ValueError("No labels found in {}".format(labels_path))

    return labels


class KeyPointClassifier(object):
    """Klasyfikator gestów uruchamiany przez TensorFlow Lite."""

    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.num_features = int(self.input_details[0]['shape'][-1])
        self.num_classes = int(self.output_details[0]['shape'][-1])

    def _prepare_input(self, landmark_list):
        values = np.asarray(landmark_list, dtype=np.float32).reshape(-1)

        if values.size != self.num_features:
            raise ValueError(
                "Classifier expects {} features, got {}.".format(
                    self.num_features, values.size))

        return values.reshape(1, -1)

    def predict_proba(self, landmark_list):
        """Zwraca wektor prawdopodobieństw dla pojedynczej próbki."""
        input_tensor = self._prepare_input(landmark_list)

        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_details_tensor_index, input_tensor)
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)

        return np.asarray(np.squeeze(result), dtype=np.float32).reshape(-1)

    def predict(self, landmark_list):
        """Zwraca ``(class_id, confidence)``."""
        probabilities = self.predict_proba(landmark_list)
        class_id = int(np.argmax(probabilities))
        return class_id, float(probabilities[class_id])

    def __call__(self, landmark_list):
        return int(np.argmax(self.predict_proba(landmark_list)))


class KerasKeyPointClassifier(object):
    """Klasyfikator gestów oparty o pełny model Keras (.keras / .h5)."""

    def __init__(self, model_path):
        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path)

        input_shape = self.model.input_shape
        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected input shape {} for the keypoint classifier "
                "(expected (None, {}) = {} hands x {} landmarks x 2).".format(
                    input_shape, NUM_FEATURES, NUM_HANDS, LANDMARKS_PER_HAND))

        self.num_features = int(input_shape[-1])
        self.num_classes = int(self.model.output_shape[-1])

    def _prepare_input(self, landmark_list):
        values = np.asarray(landmark_list, dtype=np.float32).reshape(-1)

        if values.size != self.num_features:
            raise ValueError(
                "Classifier expects {} features, got {}.".format(
                    self.num_features, values.size))

        return values.reshape(1, -1)

    def predict_proba(self, landmark_list):
        """Zwraca wektor prawdopodobieństw dla pojedynczej próbki."""
        input_tensor = self._prepare_input(landmark_list)

        scores = self.model(input_tensor, training=False)
        if hasattr(scores, 'numpy'):
            scores = scores.numpy()

        return np.asarray(np.squeeze(scores), dtype=np.float32).reshape(-1)

    def predict(self, landmark_list):
        """Zwraca ``(class_id, confidence)``."""
        probabilities = self.predict_proba(landmark_list)
        class_id = int(np.argmax(probabilities))
        return class_id, float(probabilities[class_id])

    def __call__(self, landmark_list):
        return int(np.argmax(self.predict_proba(landmark_list)))


def load_keypoint_classifier(model_path, expected_features=NUM_FEATURES):
    """Ładuje klasyfikator; .keras/.h5 używają Keras, pozostałe TFLite."""
    model_path = str(model_path)

    if not os.path.isfile(model_path):
        raise FileNotFoundError("Keypoint classifier not found: {}".format(
            model_path))

    if model_path.lower().endswith(('.keras', '.h5', '.hdf5')):
        classifier = KerasKeyPointClassifier(model_path)
    else:
        classifier = KeyPointClassifier(model_path=model_path)

    if expected_features and classifier.num_features != expected_features:
        raise ValueError(
            "Keypoint classifier {} expects {} features, but the model was "
            "built for {} (retrain it with keypoint_classification_EN.ipynb "
            "on two-hand data).".format(
                model_path, expected_features, classifier.num_features))

    return classifier


def calc_landmark_list(landmarks, image_width, image_height):
    """Zamienia znormalizowane landmarki MediaPipe na piksele."""
    landmark_point = []

    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    """Przesuwa punkt odniesienia do nadgarstka i normalizuje zakres."""
    temp_landmark_list = [list(point) for point in landmark_list]

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    flattened = [coordinate
                 for landmark_point in temp_landmark_list
                 for coordinate in landmark_point]

    max_value = max((abs(value) for value in flattened), default=0.0)
    if max_value == 0:
        return [0.0] * len(flattened)

    return [value / max_value for value in flattened]


@dataclass
class HandSample:
    """Pojedyncza dłoń przygotowana do klasyfikacji."""

    handedness: str
    landmark_list: List[List[int]]
    pre_processed: List[float]
    brect: Optional[List[int]] = None

    @classmethod
    def from_mediapipe(
        cls,
        landmarks,
        handedness,
        image_width,
        image_height,
        brect=None,
    ):
        landmark_list = calc_landmark_list(landmarks, image_width, image_height)

        return cls(
            handedness=handedness,
            landmark_list=landmark_list,
            pre_processed=pre_process_landmark(landmark_list),
            brect=brect,
        )

    @property
    def height(self):
        """Średnia pozycja Y landmarków w pikselach."""
        if not self.landmark_list:
            return 0.0

        return sum(point[1] for point in self.landmark_list) / len(
            self.landmark_list)

@dataclass
class GesturePrediction:
    """Wynik klasyfikacji jednej klatki (z obiema dłońmi lub bez)."""

    hand_sign_id: int
    label: str
    confidence: float
    both_hands_visible: bool
    height_threshold_met: bool
    classified: bool
    is_inactive: bool
    height_diff: float
    height_diff_threshold: int
    features: List[float]
    hands: List[HandSample]

    @property
    def active(self):
        """True tylko dla rozpoznanego, aktywnego gestu (nie ``inactive``)."""
        return (self.classified and self.height_threshold_met
                and not self.is_inactive)

class TwoHandGestureClassifier(object):
    """Składa obie dłonie w jeden gest i stosuje próg ``height_diff``.

    Parametry
    ---------
    classifier:
        Obiekt z metodą ``predict_proba(landmark_list)`` lub ``None``.
        ``None`` oznacza tryb bez klasyfikacji (samo wykrywanie dłoni).
    labels:
        Nazwy klas w kolejności odpowiadającej wyjściu modelu.
    height_diff_threshold:
        Minimalna różnica pionowa (px) między dłońmi wymagana do
        klasyfikacji. Mniejsza różnica => gest ``inactive``.
        ``0`` wyłącza próg.
    """

    def __init__(
        self,
        classifier=None,
        labels=None,
        inactive_label=None,
        height_diff_threshold=DEFAULT_HEIGHT_DIFF_THRESHOLD,
    ):
        self.classifier = classifier
        self.labels = list(labels) if labels is not None else []

        if inactive_label is None:
            if INACTIVE_GESTURE_LABEL in self.labels:
                inactive_label = INACTIVE_GESTURE_LABEL
            elif self.labels:
                inactive_label = self.labels[-1]
            else:
                inactive_label = INACTIVE_GESTURE_LABEL

        self.inactive_label = inactive_label
        self.inactive_id = (self.labels.index(inactive_label)
                            if inactive_label in self.labels else -1)
        self.height_diff_threshold = int(height_diff_threshold)

        if self.classifier is not None:
            num_features = getattr(self.classifier, 'num_features',
                                   NUM_FEATURES)
            if num_features != NUM_FEATURES:
                raise ValueError(
                    "Classifier expects {} features, but a two-hand gesture "
                    "needs {} ({} hands x {} landmarks x 2).".format(
                        num_features, NUM_FEATURES, NUM_HANDS,
                        LANDMARKS_PER_HAND))

            num_classes = getattr(self.classifier, 'num_classes', None)
            if num_classes is not None and self.labels:
                if num_classes != len(self.labels):
                    raise ValueError(
                        "Classifier returns {} classes, but {} contains {} "
                        "labels.".format(num_classes, 'labels', len(
                            self.labels)))

    @property
    def num_features(self):
        return NUM_FEATURES

    @staticmethod
    def order_hands(hands):
        """Ustala deterministyczną kolejność dłoni (Left przed Right)."""
        return sorted(
            hands,
            key=lambda hand: (
                hand.handedness,
                hand.landmark_list[0][0] if hand.landmark_list else 0,
            ),
        )

    @staticmethod
    def combine_features(hands):
        """Skleja cechy maksymalnie dwóch dłoni w jeden wektor."""
        features = []

        for hand in list(hands)[:NUM_HANDS]:
            if len(hand.pre_processed) != FEATURES_PER_HAND:
                raise ValueError(
                    "A hand must contain {} features, got {}.".format(
                        FEATURES_PER_HAND, len(hand.pre_processed)))

            features.extend(hand.pre_processed)

        return features

    def _inactive_prediction(
        self,
        ordered_hands,
        features,
        threshold,
        both_hands_visible,
        height_diff=0.0,
        height_threshold_met=False,
        label=None,
    ):
        return GesturePrediction(
            hand_sign_id=self.inactive_id,
            label=label if label is not None else self.inactive_label,
            confidence=0.0,
            both_hands_visible=both_hands_visible,
            height_threshold_met=height_threshold_met,
            classified=False,
            is_inactive=True,
            height_diff=height_diff,
            height_diff_threshold=threshold,
            features=features,
            hands=ordered_hands,
        )

    def classify(self, hands, height_diff_threshold=None):
        """Klasyfikuje gest z listy obiektów :class:`HandSample`."""
        threshold = (self.height_diff_threshold
                     if height_diff_threshold is None
                     else int(height_diff_threshold))

        ordered_hands = self.order_hands(hands)
        features = self.combine_features(ordered_hands)
        both_hands_visible = len(ordered_hands) >= NUM_HANDS

        if not both_hands_visible:
            return self._inactive_prediction(
                ordered_hands, features, threshold, both_hands_visible)

        height_diff = abs(ordered_hands[0].height - ordered_hands[1].height)
        height_threshold_met = (height_diff >= threshold)

        if not height_threshold_met:
            return self._inactive_prediction(
                ordered_hands,
                features,
                threshold,
                both_hands_visible,
                height_diff=height_diff,
                height_threshold_met=False,
            )

        # Tryb bez klasyfikatora (samo wykrywanie dłoni).
        if self.classifier is None:
            return self._inactive_prediction(
                ordered_hands,
                features,
                threshold,
                both_hands_visible,
                height_diff=height_diff,
                height_threshold_met=True,
                label=CLASSIFICATION_OFF_LABEL,
            )

        probabilities = self.classifier.predict_proba(features)
        hand_sign_id = int(np.argmax(probabilities))

        if hand_sign_id < len(self.labels):
            label = self.labels[hand_sign_id]
        else:
            label = str(hand_sign_id)

        is_inactive = (hand_sign_id == self.inactive_id
                       or label == self.inactive_label)

        return GesturePrediction(
            hand_sign_id=hand_sign_id,
            label=label,
            confidence=float(probabilities[hand_sign_id]),
            both_hands_visible=both_hands_visible,
            height_threshold_met=height_threshold_met,
            classified=True,
            is_inactive=is_inactive,
            height_diff=height_diff,
            height_diff_threshold=threshold,
            features=features,
            hands=ordered_hands,
        )
