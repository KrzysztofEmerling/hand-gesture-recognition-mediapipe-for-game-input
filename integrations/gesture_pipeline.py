#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pipeline rozpoznawania gestów dłoni i wysyłania ich do Unity.

Pipeline:

1. pobiera klatki z kamery,
2. wykrywa dłonie przez MediaPipe Hand Landmarker,
3. składa landmarki obu dłoni i klasyfikuje gest przez
   :class:`model.keypoint_classifier.keypoint_classifier.TwoHandGestureClassifier`
   (wspólny próg ``height_diff_threshold`` z ``app.py``),
4. wysyła zdarzenia ``gesture_started`` / ``gesture_ended`` do Unity przez
   :class:`integrations.event_sender.EventSender` (importowanego przez
   ``integrations/send_event.py``).

Uruchomienie::

    python integrations/gesture_pipeline.py --host 127.0.0.1 --port 5555

lub bez wysyłania (podgląd w konsoli)::

    python integrations/gesture_pipeline.py --dry_run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.send_event import EventSender  
from model import (
    DEFAULT_HEIGHT_DIFF_THRESHOLD,
    INACTIVE_GESTURE_LABEL,
    NUM_HANDS,
    GesturePrediction,
    HandSample,
    TwoHandGestureClassifier,
    load_keypoint_classifier,
    load_labels,
)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "hand_landmarker.task"
DEFAULT_CLASSIFIER_PATH = (PROJECT_ROOT / "model" / "keypoint_classifier"
                           / "keypoint_classifier.tflite")
DEFAULT_LABELS_PATH = (PROJECT_ROOT / "model" / "keypoint_classifier"
                       / "keypoint_classifier_label.csv")

DEFAULT_UNITY_HOST = "127.0.0.1"
DEFAULT_UNITY_PORT = 5555


class GesturePipeline:
    """Wykrywanie dłoni + klasyfikacja gestu dla pojedynczej klatki."""

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        classifier_path=DEFAULT_CLASSIFIER_PATH,
        labels_path=DEFAULT_LABELS_PATH,
        height_diff_threshold=DEFAULT_HEIGHT_DIFF_THRESHOLD,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        num_hands=NUM_HANDS,
    ) -> None:
        self.labels = load_labels(labels_path)
        self.inactive_label = (
            INACTIVE_GESTURE_LABEL
            if INACTIVE_GESTURE_LABEL in self.labels
            else self.labels[-1]
        )

        classifier = load_keypoint_classifier(classifier_path)
        self.gesture_classifier = TwoHandGestureClassifier(
            classifier=classifier,
            labels=self.labels,
            inactive_label=self.inactive_label,
            height_diff_threshold=height_diff_threshold,
        )

        num_hands = max(int(num_hands), NUM_HANDS)

        hand_landmarker_options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hand_landmarker = (
            vision.HandLandmarker.create_from_options(
                hand_landmarker_options))

        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1

    def detect(
        self,
        image_bgr,
        timestamp_ms: Optional[int] = None,
    ) -> GesturePrediction:
        """Przetwarza klatkę BGR i zwraca wynik klasyfikacji gestu."""
        if timestamp_ms is None:
            timestamp_ms = int((time.monotonic() - self._start_time) * 1000)

        # Timestamp musi rosnąć monotonicznie w trybie VIDEO.
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb_image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = self.hand_landmarker.detect_for_video(
            mp_image, timestamp_ms)

        image_height, image_width = image_bgr.shape[:2]
        hands = []
        for hand_landmarks, handedness in zip(results.hand_landmarks,
                                              results.handedness):
            hands.append(HandSample.from_mediapipe(
                hand_landmarks,
                handedness[0].category_name,
                image_width,
                image_height,
            ))

        return self.gesture_classifier.classify(hands)

    def close(self) -> None:
        self.hand_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class UnityGestureBridge:
    """Zamienia wyniki klasyfikacji na zdarzenia start/koniec gestu."""

    def __init__(
        self,
        sender,
        min_confidence: float = 0.0,
        inactive_label: str = INACTIVE_GESTURE_LABEL,
        verbose: bool = False,
    ) -> None:
        self.sender = sender
        self.min_confidence = float(min_confidence)
        self.inactive_label = inactive_label
        self.verbose = verbose

        self._current_gesture: Optional[str] = None
        self._current_confidence = 0.0
        self._started_at = 0.0

    def update(self, prediction: GesturePrediction) -> Optional[str]:
        """Aktualizuje stan gestu; zwraca nazwę wysłanego zdarzenia."""
        now = time.monotonic()

        gesture = None
        if (prediction.active
                and prediction.label != self.inactive_label
                and prediction.confidence >= self.min_confidence):
            gesture = prediction.label

        if gesture == self._current_gesture:
            return None

        event_name = None

        if self._current_gesture is not None:
            duration = max(0.0, now - self._started_at)
            self.sender.send_gesture_ended(
                self._current_gesture,
                duration=duration,
                confidence=self._current_confidence,
            )
            self._log("gesture_ended", self._current_gesture, duration)
            event_name = "gesture_ended"

        if gesture is not None:
            self.sender.send_gesture_started(
                gesture,
                confidence=prediction.confidence,
                hand_sign_id=prediction.hand_sign_id,
            )
            self._log("gesture_started", gesture, prediction.confidence)
            event_name = "gesture_started"

        self._current_gesture = gesture
        self._current_confidence = prediction.confidence
        self._started_at = now

        return event_name

    def close(self) -> None:
        """Kończy aktualnie trwający gest przed zamknięciem pipeline'u."""
        if self._current_gesture is None:
            return

        duration = max(0.0, time.monotonic() - self._started_at)
        self.sender.send_gesture_ended(
            self._current_gesture,
            duration=duration,
            confidence=self._current_confidence,
        )
        self._log("gesture_ended", self._current_gesture, duration)
        self._current_gesture = None

    def _log(self, event: str, gesture: str, value: float) -> None:
        if not self.verbose:
            return

        print("[UnityGestureBridge] {} gesture={} ({:.3f})".format(
            event, gesture, value))


class DryRunSender:
    """Zamiennik EventSender do testowania pipeline'u bez Unity."""

    def send_gesture_started(self, gesture, confidence=None, **extra) -> bool:
        print("[dry-run] gesture_started gesture={} confidence={}".format(
            gesture, confidence))
        return True

    def send_gesture_ended(self, gesture, duration=None, **extra) -> bool:
        print("[dry-run] gesture_ended gesture={} duration={}".format(
            gesture, duration))
        return True

    def close(self) -> None:
        pass


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Rozpoznawanie gestów dłoni i wysyłanie ich do Unity.")

    parser.add_argument("--host", default=DEFAULT_UNITY_HOST,
                        help="adres Unity TCP (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_UNITY_PORT,
                        help="port Unity TCP (default: %(default)s)")
    parser.add_argument("--reconnect_delay", type=float, default=2.0)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)

    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--classifier_path",
                        default=str(DEFAULT_CLASSIFIER_PATH))
    parser.add_argument("--labels_path", default=str(DEFAULT_LABELS_PATH))

    parser.add_argument("--num_hands", type=int, default=NUM_HANDS)
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument("--height_diff_threshold", type=int,
                        default=DEFAULT_HEIGHT_DIFF_THRESHOLD,
                        help="minimalna różnica pionowa (px) między dłońmi; "
                             "mniejsza => inactive (0 = wyłączone)")
    parser.add_argument("--min_gesture_confidence", type=float, default=0.0,
                        help="minimalna pewność, aby wysłać gest do Unity")

    parser.add_argument("--no_display", action="store_true",
                        help="nie otwieraj okna podglądu OpenCV")
    parser.add_argument("--no_flip", action="store_true",
                        help="nie odbijaj lustrzanie obrazu z kamery")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="zakończ po N klatkach (0 = bez limitu)")
    parser.add_argument("--dry_run", action="store_true",
                        help="nie łącz się z Unity, wypisuj zdarzenia")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args(argv)


def run_camera(args) -> None:
    pipeline = GesturePipeline(
        model_path=args.model_path,
        classifier_path=args.classifier_path,
        labels_path=args.labels_path,
        height_diff_threshold=args.height_diff_threshold,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        num_hands=args.num_hands,
    )

    if args.dry_run:
        sender = DryRunSender()
    else:
        sender = EventSender(
            host=args.host,
            port=args.port,
            reconnect_delay=args.reconnect_delay,
        )
        if not sender.connect():
            print("[gesture_pipeline] Unity niedostępne - będzie ponawiane "
                  "przy wysyłaniu gestów.")

    bridge = UnityGestureBridge(
        sender=sender,
        min_confidence=args.min_gesture_confidence,
        inactive_label=pipeline.inactive_label,
        verbose=args.verbose or args.dry_run,
    )

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        pipeline.close()
        sender.close()
        raise SystemExit("Nie można otworzyć kamery {}".format(args.device))

    frame_count = 0

    try:
        while True:
            ok, image = cap.read()
            if not ok:
                print("[gesture_pipeline] Brak klatki z kamery - koniec.")
                break

            if not args.no_flip:
                image = cv.flip(image, 1)

            prediction = pipeline.detect(image)
            bridge.update(prediction)

            frame_count += 1

            if not args.no_display:
                draw_status(image, prediction)
                cv.imshow("Gesture Pipeline", image)
                key = cv.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("\n[gesture_pipeline] Przerwano przez użytkownika.")
    finally:
        bridge.close()
        pipeline.close()
        cap.release()
        cv.destroyAllWindows()
        sender.close()


def draw_status(image, prediction: GesturePrediction) -> None:
    color = (0, 255, 0) if prediction.active else (0, 165, 255)
    gesture_text = "GESTURE: {}".format(prediction.label)

    cv.putText(image, gesture_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9,
               (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, gesture_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9,
               color, 2, cv.LINE_AA)

    if prediction.both_hands_visible:
        height_text = "dY: {:.0f}px / {}".format(
            prediction.height_diff, prediction.height_diff_threshold)
    else:
        height_text = "waiting for both hands..."

    cv.putText(image, height_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6,
               (255, 255, 255), 1, cv.LINE_AA)

    if prediction.classified:
        confidence_text = "CONF: {:.2f}".format(prediction.confidence)
        cv.putText(image, confidence_text, (10, 85), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1, cv.LINE_AA)


def main(argv=None) -> int:
    args = get_args(argv)
    run_camera(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
