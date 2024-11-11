from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pyautogui
import time

app = Flask(__name__)

# Initialize MediaPipe and gesture detection
drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

# Gesture detection variables
prev = -1
start_init = False

# Function to count fingers based on landmarks
def count_fingers(lst):
    cnt = 0
    thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2
    if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6:
        cnt += 1
    return cnt

# Function to generate video frames
def generate_frames():
    global prev, start_init
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            hand_keyPoints = res.multi_hand_landmarks[0]
            cnt = count_fingers(hand_keyPoints)
            if prev != cnt:
                if not start_init:
                    start_time = time.time()
                    start_init = True
                elif (time.time() - start_time) > 0.2:
                    # Perform actions based on the number of fingers counted
                    if cnt == 1:
                        pyautogui.press("right")
                    elif cnt == 2:
                        pyautogui.press("left")
                    elif cnt == 3:
                        pyautogui.press("up")
                    elif cnt == 4:
                        pyautogui.press("down")
                    elif cnt == 5:
                        pyautogui.press("space")
                    prev = cnt
                    start_init = False
            drawing.draw_landmarks(frame, hand_keyPoints, mp.solutions.hands.HAND_CONNECTIONS)

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
