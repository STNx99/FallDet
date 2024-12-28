import cv2
import numpy as np
from ultralytics import YOLO
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

model = YOLO('yolov8m-pose.pt')

video_path = "falltest3.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False, save_txt=False, verbose=False)

    for result in results:
        keypoints_data = result.keypoints

        if keypoints_data is not None:
            keypoints = keypoints_data.xy[0]

            if keypoints.shape[0] >= 13:
                nose_x, nose_y = keypoints[0][:2]
                left_hip_x, left_hip_y = keypoints[11][:2]
                right_hip_x, right_hip_y = keypoints[12][:2]

                print(f"Nose: x={nose_x:.2f}, y={nose_y:.2f}")
                print(f"Left Hip: x={left_hip_x:.2f}, y={left_hip_y:.2f}")
                print(f"Right Hip: x={right_hip_x:.2f}, y={right_hip_y:.2f}")
                if(nose_y != 0 and left_hip_y != 0 and right_hip_y != 0 ):
                    if (nose_y > left_hip_y
                            or nose_y > right_hip_y
                            or (left_hip_y - nose_y) < 30
                            or (right_hip_y - nose_y) < 30):
                        print("Ngã!")
                        cv2.putText(
                            frame,
                            "Fall Detected!",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA
                        )
                else:
                    cv2.putText(
                        frame,
                        "Chua the nhan dien",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )

    processed_frame = results[0].plot()

    combined_frame = cv2.addWeighted(frame, 0.7, processed_frame, 0.3, 0)

    cv2.imshow('YOLOv8 Pose Estimation - Fall Detection', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()