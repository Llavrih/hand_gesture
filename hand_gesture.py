import time

import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Get screen size for mapping coordinates
screen_width, screen_height = pyautogui.size()

# Variables for double-click gesture tracking
last_click_time = 0
click_threshold = 0.4  # seconds; adjust as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally and convert to RGB
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Draw the hand annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the tips of the index and middle fingers, and the thumb tip
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP
            ]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_pip = hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_PIP
            ]

            # Map the finger coordinates to screen coordinates
            cursor_x = int(index_tip.x * screen_width)
            cursor_y = int(index_tip.y * screen_height)

            # Move the cursor
            pyautogui.moveTo(cursor_x, cursor_y)

            # Check for double-click gesture (middle finger close to thumb)
            if (
                abs(middle_tip.x - thumb_tip.x) < 0.05
                and abs(middle_tip.y - thumb_tip.y) < 0.05
            ):
                current_time = time.time()
                if current_time - last_click_time <= click_threshold:
                    pyautogui.doubleClick()
                    last_click_time = 0
                else:
                    last_click_time = current_time

            # Check if the tips of the index and middle fingers are close
            elif (
                abs(index_tip.x - middle_tip.x) < 0.06
                and abs(index_tip.y - middle_tip.y) < 0.06
            ):
                # Check if the tips are lower than the knuckles
                if index_tip.y > index_pip.y and middle_tip.y > middle_pip.y:
                    # Scroll down
                    pyautogui.scroll(-30)
                else:
                    # Scroll up
                    pyautogui.scroll(30)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
