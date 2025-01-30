import time
import cv2
import mediapipe as mp
import numpy as np

CALORIES_PER_REP = 0.35


def detect():
    start_time = time.time()

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0
    stage = None
    form = None
    calories_burnt = 0

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

                # Get coordinates
                lefthip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                leftshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                leftelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                leftwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                righthip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                rightshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                rightelbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                rightshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                rightelbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                rightwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle

                angle2 = calculate_angle(righthip, rightshoulder, rightelbow)
                angle1 = calculate_angle(rightshoulder, rightelbow, rightwrist)
                angle3 = calculate_angle(lefthip, leftshoulder, leftelbow)
                angle4 = calculate_angle(leftshoulder, leftelbow, leftwrist)
                # Visualize angle

                cv2.putText(image, str(angle2),
                            tuple(np.multiply(rightshoulder,
                                              [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                            255, 255), 1, cv2.LINE_AA
                            )

                cv2.putText(image, str(angle1),
                            tuple(np.multiply(rightelbow,
                                              [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                            255, 255), 1, cv2.LINE_AA
                            )
                cv2.putText(image, str(angle4),
                            tuple(np.multiply(leftelbow,
                                              [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                            255, 255), 1, cv2.LINE_AA
                            )

                cv2.putText(image, str(angle3),
                            tuple(np.multiply(leftshoulder,
                                              [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                            255, 255), 1, cv2.LINE_AA
                            )
                # Curl counter logic
                if angle1 > 160 and angle2 > 160 and angle3 > 160 and angle4 > 160:
                    stage = "UP"
                    form = "CORRECT"
                if angle1 < 90 and angle2 > 160 and angle3 > 160 and angle4 < 90 and stage == 'UP':
                    stage = "DOWN"
                    form = "CORRECT"
                    counter += 1
                    print(counter)

                elif angle2 < 160 and angle3 < 160 and stage == 'DOWN':
                    form = "INCORRECT"
                elif angle2 < 160 and angle3 < 160 and stage == 'UP':
                    form = "INCORRECT"

                if counter > 0:
                    calories_burnt = counter * CALORIES_PER_REP

            except:
                pass

            # Adjust the wait time as needed

            # Setup status box
            cv2.rectangle(image, (0, 0), (150, 73), (0, 0, 0), -1)

            cv2.putText(image, 'TRICPES CURLS', (275, 25),
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

            cv2.circle(image, tuple(np.multiply(rightshoulder, [
                       640, 480]).astype(int)), 5, (0, 0, 255), -1)
            cv2.circle(image, tuple(np.multiply(
                rightwrist, [640, 480]).astype(int)), 5, (0, 0, 255), -1)
            cv2.circle(image, tuple(np.multiply(
                rightelbow, [640, 480]).astype(int)), 5, (0, 0, 255), -1)
            cv2.circle(image, tuple(np.multiply(leftshoulder, [
                       640, 480]).astype(int)), 5, (0, 0, 255), -1)
            cv2.circle(image, tuple(np.multiply(
                leftwrist, [640, 480]).astype(int)), 5, (0, 0, 255), -1)
            cv2.circle(image, tuple(np.multiply(
                leftelbow, [640, 480]).astype(int)), 5, (0, 0, 255), -1)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=1, circle_radius=2),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=1, circle_radius=2)
                                      )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):

                break

        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time
        elapsed_time_minutes = elapsed_time/60
        elapsed_time_formatted = "{:.2f}".format(elapsed_time_minutes)
        print("Elapsed time:", elapsed_time_formatted, "Minuts")

        calories_burnt = counter * CALORIES_PER_REP

        print("Calories Burnt:", calories_burnt)

        cap.release()
        cv2.destroyAllWindows()

        return "TRICEPS CURLS DETECTION RESULT", counter, elapsed_time_formatted, calories_burnt
