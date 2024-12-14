import cv2
from ultralytics import YOLO

model = YOLO('yolov8m-pose.pt')

video_path = 'gymtest.mp4'
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
            else:
                print("Not enough keypoints detected.")

    processed_frame = results[0].plot()

    cv2.imshow('YOLOv8 Pose Estimation', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
