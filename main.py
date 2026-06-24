import cv2
import mediapipe as mp
import os
import subprocess
import signal


def main():
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Path to the downloaded face landmarker model
    model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

    # Configure Face Landmarker options
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    # Eye contour indices
    LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    # Setup webcam
    cap = cv2.VideoCapture(0)

    # Setup video player
    video_path = os.path.join(os.path.dirname(__file__), "Main.mp4")
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print(f"Warning: Could not open video file at {video_path}")

    # Variables for smoothing gaze ratio (Exponential Moving Average)
    smooth_gaze_x = 0.5
    smooth_gaze_y = 0.5
    alpha = 0.10  # Smoothing factor (lower is smoother, higher is more responsive)

    # Look away hysteresis counters and state
    look_away_counter = 0
    LOOK_AWAY_THRESHOLD = 8  # frames to wait before switching to video (~0.25 seconds)
    state = "TRACKING"       # "TRACKING" or "PLAYING_VIDEO"

    # Audio playback process using system's ffplay (avoids python compilation issues)
    audio_process = None
    is_audio_paused = True

    with FaceLandmarker.create_from_options(options) as landmarker:
        frame_timestamp_ms = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a mirror view
            frame = cv2.flip(frame, 1)

            img_h, img_w, _ = frame.shape

            # Convert BGR to RGB and wrap in a MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect face landmarks (VIDEO mode requires increasing timestamps)
            frame_timestamp_ms += 33  # ~30 fps
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            gaze_direction_h = "CENTER"
            gaze_direction_v = "CENTER"
            face_detected = False

            if results.face_landmarks:
                face_detected = True
                for face_landmarks in results.face_landmarks:
                    num_landmarks = len(face_landmarks)
                    
                    # Ensure we have iris landmarks (indices up to 477)
                    if num_landmarks > 473:
                        left_iris = face_landmarks[468]
                        right_iris = face_landmarks[473]

                        # Get eye contour landmarks
                        left_eye_pts = [face_landmarks[idx] for idx in LEFT_EYE_CONTOUR]
                        right_eye_pts = [face_landmarks[idx] for idx in RIGHT_EYE_CONTOUR]

                        # Left Eye Bounds
                        left_min_x = min(p.x for p in left_eye_pts)
                        left_max_x = max(p.x for p in left_eye_pts)
                        left_min_y = min(p.y for p in left_eye_pts)
                        left_max_y = max(p.y for p in left_eye_pts)

                        # Right Eye Bounds
                        right_min_x = min(p.x for p in right_eye_pts)
                        right_max_x = max(p.x for p in right_eye_pts)
                        right_min_y = min(p.y for p in right_eye_pts)
                        right_max_y = max(p.y for p in right_eye_pts)

                        # Calculate relative positions (0.0 = Left/Top edge, 1.0 = Right/Bottom edge)
                        left_ratio_x = (left_iris.x - left_min_x) / (left_max_x - left_min_x) if (left_max_x - left_min_x) != 0 else 0.5
                        left_ratio_y = (left_iris.y - left_min_y) / (left_max_y - left_min_y) if (left_max_y - left_min_y) != 0 else 0.5

                        right_ratio_x = (right_iris.x - right_min_x) / (right_max_x - right_min_x) if (right_max_x - right_min_x) != 0 else 0.5
                        right_ratio_y = (right_iris.y - right_min_y) / (right_max_y - right_min_y) if (right_max_y - right_min_y) != 0 else 0.5

                        # Average the eye ratios
                        curr_gaze_x = (left_ratio_x + right_ratio_x) / 2.0
                        curr_gaze_y = (left_ratio_y + right_ratio_y) / 2.0

                        # Apply EMA smoothing
                        smooth_gaze_x = (1 - alpha) * smooth_gaze_x + alpha * curr_gaze_x
                        smooth_gaze_y = (1 - alpha) * smooth_gaze_y + alpha * curr_gaze_y

                        # Classify horizontal gaze direction
                        if smooth_gaze_x < 0.4:
                            gaze_direction_h = "LEFT"
                        elif smooth_gaze_x > 0.6:
                            gaze_direction_h = "RIGHT"
                        else:
                            gaze_direction_h = "CENTER"

                        # Classify vertical gaze direction
                        if smooth_gaze_y < 0.40:
                            gaze_direction_v = "UP"
                        elif smooth_gaze_y > 0.60:
                            gaze_direction_v = "DOWN"
                        else:
                            gaze_direction_v = "CENTER"

                        # Draw eye contours (blue)
                        for eye_pts in [left_eye_pts, right_eye_pts]:
                            points = []
                            for p in eye_pts:
                                points.append((int(p.x * img_w), int(p.y * img_h)))
                            
                            # Draw contour as connected lines
                            for i in range(len(points)):
                                cv2.line(frame, points[i], points[(i + 1) % len(points)], (255, 100, 0), 1)

                        # Draw iris centers (red)
                        for iris in [left_iris, right_iris]:
                            ix = int(iris.x * img_w)
                            iy = int(iris.y * img_h)
                            cv2.circle(frame, (ix, iy), 3, (0, 0, 255), -1)

            # Determine if user is looking away
            is_looking_away = (not face_detected) or (gaze_direction_h != "CENTER") or (gaze_direction_v != "CENTER")

            if is_looking_away:
                look_away_counter += 1
            else:
                look_away_counter = 0

            # State transition with threshold checks
            if look_away_counter >= LOOK_AWAY_THRESHOLD:
                state = "PLAYING_VIDEO"
            else:
                state = "TRACKING"

            # Render display frame based on current state
            if state == "PLAYING_VIDEO":
                # Manage audio playback - start playing if stopped
                if audio_process is None or audio_process.poll() is not None:
                    try:
                        audio_process = subprocess.Popen(
                            ["ffplay", "-nodisp", "-autoexit", "-loop", "0", video_path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        is_audio_paused = False
                    except Exception as e:
                        print(f"Error starting audio: {e}")
                # Resume if paused
                elif is_audio_paused:
                    try:
                        audio_process.send_signal(signal.SIGCONT)
                    except Exception:
                        pass
                    is_audio_paused = False

                # Read next frame from Main.mp4
                ret_vid, vid_frame = video_cap.read()
                if not ret_vid:
                    # Loop video back to start
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_vid, vid_frame = video_cap.read()
                    
                    # Restart audio playback to keep it in sync on loop
                    if audio_process is not None:
                        try:
                            audio_process.terminate()
                            audio_process.wait(timeout=1)
                        except Exception:
                            pass
                    try:
                        audio_process = subprocess.Popen(
                            ["ffplay", "-nodisp", "-autoexit", "-loop", "0", video_path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        is_audio_paused = False
                    except Exception as e:
                        print(f"Error restarting audio: {e}")

                if ret_vid and vid_frame is not None:
                    # Resize video frame to match webcam dimensions
                    display_frame = cv2.resize(vid_frame, (img_w, img_h))
                else:
                    display_frame = frame.copy()

                # Overlay status indicator on video frame
                cv2.rectangle(display_frame, (10, 10), (450, 75), (0, 0, 0), -1)
                cv2.putText(display_frame, "PLAYING VIDEO (LOOKED AWAY)", (20, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Small pulse indicator to show it's playing
                if (frame_timestamp_ms // 400) % 2 == 0:
                    cv2.circle(display_frame, (420, 38), 6, (0, 0, 255), -1)

                # Show status overlay in bottom-right corner
                overlay_h, overlay_w = 70, 220
                cv2.rectangle(display_frame, (img_w - overlay_w - 10, img_h - overlay_h - 10), (img_w - 10, img_h - 10), (0, 0, 0), -1)
                cv2.putText(display_frame, f"Gaze X (H): {smooth_gaze_x:.3f}", (img_w - overlay_w, img_h - overlay_h + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                cv2.putText(display_frame, f"Gaze Y (V): {smooth_gaze_y:.3f}", (img_w - overlay_w, img_h - overlay_h + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                cv2.putText(display_frame, f"Look Away Ctr: {look_away_counter}", (img_w - overlay_w, img_h - overlay_h + 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            else:
                # Pause audio if playing
                if audio_process is not None and audio_process.poll() is None and not is_audio_paused:
                    try:
                        audio_process.send_signal(signal.SIGSTOP)
                    except Exception:
                        pass
                    is_audio_paused = True

                display_frame = frame.copy()

                # Combine horizontal and vertical classification
                if gaze_direction_h == "CENTER" and gaze_direction_v == "CENTER":
                    gaze_text = "LOOKING: CENTER"
                elif gaze_direction_h == "CENTER":
                    gaze_text = f"LOOKING: {gaze_direction_v}"
                elif gaze_direction_v == "CENTER":
                    gaze_text = f"LOOKING: {gaze_direction_h}"
                else:
                    gaze_text = f"LOOKING: {gaze_direction_h}-{gaze_direction_v}"

                # Draw HUD Background
                cv2.rectangle(display_frame, (10, 10), (380, 110), (0, 0, 0), -1)

                # Draw text indicators
                cv2.putText(display_frame, gaze_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Gaze X (H): {smooth_gaze_x:.3f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(display_frame, f"Gaze Y (V): {smooth_gaze_y:.3f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Display the result
            cv2.imshow('Eye & Gaze Tracking (Press Q to Quit)', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean up subprocess
    if audio_process is not None:
        try:
            audio_process.terminate()
        except Exception:
            pass

    cap.release()
    video_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
