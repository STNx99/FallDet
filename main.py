import cv2
from ultralytics import YOLO

model = YOLO('yolov8m-pose.pt')

video_path = 0
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False, save_txt=False)

    for result in results:
        for pose in result.keypoints:
            keypoints = pose.xy[0]

            if keypoints.shape[0] >= 13:
                nose_x, nose_y = keypoints[0][:2]
                left_hip_x, left_hip_y = keypoints[11][:2]
                right_hip_x, right_hip_y = keypoints[12][:2]

                print(f"Nose: x={nose_x.item():.2f}, y={nose_y.item():.2f}")
                print(f"Left Hip: x={left_hip_x.item():.2f}, y={left_hip_y.item():.2f}")
                print(f"Right Hip: x={right_hip_x.item():.2f}, y={right_hip_y.item():.2f}")

                # Kiểm tra điều kiện ngã
                if (nose_y > left_hip_y
                        or nose_y > right_hip_y
                        or (left_hip_y - nose_y)  < 30
                        or (right_hip_y - nose_y) < 30):
                    print("Ngã!")
                    # Hiển thị thông báo trên video
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
                print("Not enough keypoints detected.")

    processed_frame = results[0].plot()

    # Thêm khung hình đã xử lý với thông báo nếu có
    combined_frame = cv2.addWeighted(frame, 0.7, processed_frame, 0.3, 0)

    cv2.imshow('YOLOv8 Pose Estimation', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
