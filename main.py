import cv2
import numpy as np
from ultralytics import YOLO
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

# Load YOLOv8 model
model = YOLO('yolov8m-pose.pt')

# Load video
video_path = "falltest.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán pose
    results = model.predict(source=frame, save=False, save_txt=False, verbose=False)

    for result in results:
        # Lấy keypoints (pose estimation output)
        keypoints_data = result.keypoints  # Access keypoints

        if keypoints_data is not None:
            keypoints = keypoints_data.xy[0]  # Get keypoints as numpy array

            # Kiểm tra keypoints đủ lớn để xác định
            if keypoints.shape[0] >= 13:
                nose_x, nose_y = keypoints[0][:2]  # Keypoint 0 là mũi
                left_hip_x, left_hip_y = keypoints[11][:2]  # Keypoint 11 là hông trái
                right_hip_x, right_hip_y = keypoints[12][:2]  # Keypoint 12 là hông phải

                # In thông tin keypoints
                print(f"Nose: x={nose_x:.2f}, y={nose_y:.2f}")
                print(f"Left Hip: x={left_hip_x:.2f}, y={left_hip_y:.2f}")
                print(f"Right Hip: x={right_hip_x:.2f}, y={right_hip_y:.2f}")

                # Kiểm tra điều kiện phát hiện ngã
                if (nose_y > left_hip_y or nose_y > right_hip_y or
                    (left_hip_y - nose_y) < 30 or (right_hip_y - nose_y) < 30):
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

    # Vẽ kết quả lên khung hình
    processed_frame = results[0].plot()

    # Kết hợp khung hình gốc với khung hình vẽ kết quả
    combined_frame = cv2.addWeighted(frame, 0.7, processed_frame, 0.3, 0)

    # Hiển thị video
    cv2.imshow('YOLOv8 Pose Estimation - Fall Detection', combined_frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
