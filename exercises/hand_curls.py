import time
import cv2
import mediapipe as mp
import numpy as np

CALORIES_PER_REP = 0.2

def detect():
    start_time = time.time()

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0
    stage = None
    calories_burnt = 0
    form = None

    def calculate_angle(a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
            np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180.0:
            angle = 360-angle

        return angle

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates, check if landmarks are detected
                leftshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y] if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] else None
                leftelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y] if landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value] else None
                leftwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y] if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value] else None

                rightshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y] if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] else None
                rightelbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y] if landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value] else None
                rightwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y] if landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value] else None
                righthip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y] if landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value] else None

                # Check if all required landmarks are detected
                if leftshoulder and leftelbow and leftwrist and rightshoulder and rightelbow and rightwrist and righthip:
                    # Calculate angles
                    angle1 = calculate_angle(leftshoulder, leftelbow, leftwrist)
                    angle2 = calculate_angle(rightshoulder, rightelbow, rightwrist)
                    angle3 = calculate_angle(rightelbow, rightshoulder, righthip)

                    # Visualize angles
                    cv2.putText(image, str(angle1),
                                tuple(np.multiply(leftelbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(angle2),
                                tuple(np.multiply(rightelbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    # Curl counter logic
                    if angle1 > 160 and angle2 > 160:
                        stage = "UP"
                        form = "CORRECT"
                    if angle1 < 90 and angle2 < 90 and stage == 'UP':
                        stage = "DOWN"
                        form = "CORRECT"
                        counter += 1
                        print(counter)

                    elif angle3 > 25 and stage == "DOWN" or stage == "up":
                        form = "INCORRECT"

                    if counter > 0:
                        calories_burnt = counter * CALORIES_PER_REP

                    # Draw circles at the points
                    cv2.circle(image, tuple(np.multiply(leftshoulder, [640, 480]).astype(int)), 5, (0, 0, 255), -1)
                    cv2.circle(image, tuple(np.multiply(leftwrist, [640, 480]).astype(int)), 5, (0, 0, 255), -1)
                    cv2.circle(image, tuple(np.multiply(leftelbow, [640, 480]).astype(int)), 5, (0, 0, 255), -1)

                    # Draw circles at the points for right arm
                    cv2.circle(image, tuple(np.multiply(rightshoulder, [640, 480]).astype(int)), 5, (0, 0, 255), -1)
                    cv2.circle(image, tuple(np.multiply(rightwrist, [640, 480]).astype(int)), 5, (0, 0, 255), -1)
                    cv2.circle(image, tuple(np.multiply(rightelbow, [640, 480]).astype(int)), 5, (0, 0, 255), -1)

                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(
                                                  color=(245, 117, 66), thickness=1, circle_radius=2),
                                              mp_drawing.DrawingSpec(
                                                  color=(245, 66, 230), thickness=1, circle_radius=2)
                                              )

                else:
                    print("Some landmarks are not detected.")
            except Exception as e:
                print(f"Error occurred: {e}")

            # Setup status box
            cv2.rectangle(image, (0, 0), (150, 120), (0, 0, 0), -1)

            cv2.putText(image, 'HAND CURLS', (275, 25),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 215, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, '  STAGE', (65, 12),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 215, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'FORM', (15, 90),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 215, 0), 1, cv2.LINE_AA)
            cv2.putText(image, form,
                        (15, 115),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Show the image
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time
        elapsed_time_minutes = elapsed_time / 60
        elapsed_time_formatted = "{:.2f}".format(elapsed_time_minutes)
        print("Elapsed time:", elapsed_time_formatted, "Minutes")

        print("Calories Burnt:", calories_burnt)
        cap.release()
        cv2.destroyAllWindows()

        return "HAND CURLS DETECTION RESULT", counter, elapsed_time_formatted, calories_burnt
