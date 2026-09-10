import time
import threading

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "./model/hand_landmarker.task"

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

mp_drawing = vision.drawing_utils
mp_drawing_styles = vision.drawing_styles
mp_hands_connections = vision.HandLandmarksConnections

latest_result = None
result_lock = threading.Lock()


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)

    if detection_result is None:
        return annotated_image

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    for hand_landmarks, handedness in zip(
        hand_landmarks_list,
        handedness_list
    ):
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands_connections.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        height, width, _ = annotated_image.shape

        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]

        text_x = max(0, int(min(x_coordinates) * width))
        text_y = max(
            30,
            int(min(y_coordinates) * height) - MARGIN
        )

        hand_name = handedness[0].category_name
        confidence = handedness[0].score

        label = f"{hand_name} ({confidence:.2f})"

        cv2.putText(
            annotated_image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA
        )

    return annotated_image


def save_result(
    result: vision.HandLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int
):
    global latest_result

    with result_lock:
        latest_result = result


options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=save_result
)


with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Error on open camera")

    start_time = time.monotonic()

    try:
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                print("Não foi possível capturar o frame.")
                break

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            landmarker.detect_async(
                mp_image,
                timestamp_ms
            )

            with result_lock:
                current_result = latest_result

            annotated_image = draw_landmarks_on_image(
                rgb_frame,
                current_result
            )

            bgr_annotated_image = cv2.cvtColor(
                annotated_image,
                cv2.COLOR_RGB2BGR
            )

            cv2.imshow(
                "MediaPipe Hands",
                bgr_annotated_image
            )

            # Pressione ESC para encerrar
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import cv2
# import time
# import numpy as np
#
# mpModel = "./model/hand_landmarker.task"
#
# BaseOptions = python.BaseOptions
# HandLandmarker = vision.HandLandmarker
# HandLandmarkerOptions = vision.HandLandmarkerOptions
#
# HandLandmarkerResult = vision.HandLandmarkerResult
# VisionRunningMode = vision.RunningMode
#
#
# mp_drawing = vision.drawing_utils
# mp_drawing_styles = vision.drawing_styles
# mp_hands_connections = vision.HandLandmarksConnections
#
# MARGIN = 10  # pixels, space above the handedness label
# FONT_SIZE = 1
# FONT_THICKNESS = 1
# HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
#
#
# def draw_landmarks_on_image(rgb_image, detection_result):
#     hand_landmarks_list = detection_result.hand_landmarks
#     handedness_list = detection_result.handedness
#     annotated_image = np.copy(rgb_image)
#
#     for idx in range(len(hand_landmarks_list)):
#         hand_landmarks = hand_landmarks_list[idx]
#         handedness = handedness_list[idx]
#
#         # Draw the 21 landmark points + skeleton connections
#         mp_drawing.draw_landmarks(
#             annotated_image,
#             hand_landmarks,
#             mp_hands_connections.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style()
#         )
#
#         # Compute where to place the "Left"/"Right" label —
#         # just above the topmost landmark of this hand
#         height, width, _ = annotated_image.shape
#         x_coordinates = [lm.x for lm in hand_landmarks]
#         y_coordinates = [lm.y for lm in hand_landmarks]
#         text_x = int(min(x_coordinates) * width)
#         text_y = int(min(y_coordinates) * height) - MARGIN
#
#         cv2.putText(annotated_image, f"{handedness[0].category_name}",
#                     (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
#                     FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
#
#     return annotated_image
#
#
# def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     """
#     Imprimir os resultados da captura da cãmera
#
#     Args:
#         result (HandLandmarkerResult):
#         output_image (mp.Image):
#         timestamp_ms (int):
#     """
#     print('hand landmarker result: {}\n'.format(result))
#
#
# options = HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=mpModel),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     result_callback=print_result)
#
# with HandLandmarker.create_from_options(options) as landmarker:
#
#     # Init Video capture with openCV(cv2)
#     cap = cv2.VideoCapture(0)
#
#     # Loop for read frames
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print("Ignoring empty camera frame.")
#             continue
#
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
#
#         timestamp_ms = int(time.time() * 1000)
#         landmarker_result = landmarker.detect_async(mp_image, timestamp_ms)
#
#         annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), landmarker_result)
#         cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
#
#         # cv2.imshow("MediaPipe hands", frame)
#
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
