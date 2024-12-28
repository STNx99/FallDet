import time

import numpy as np

from FallDet.main import cap, model

previous_nose = None
previous_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()  # Lấy thời gian hiện tại
    results = model.predict(source=frame, save=False, save_txt=False, verbose=False)

    for result in results:
        keypoints_data = result.keypoints

        if keypoints_data is not None:
            keypoints = keypoints_data.xy[0]

            if keypoints.shape[0] >= 13:  # Đảm bảo có đủ keypoints
                # Lấy tọa độ mũi hiện tại
                nose_x, nose_y = keypoints[0][:2]

                # Tính tốc độ nếu đã có thông tin từ khung trước
                if previous_nose is not None and previous_time is not None:
                    delta_time = current_time - previous_time
                    distance = np.sqrt((nose_x - previous_nose[0])**2 + (nose_y - previous_nose[1])**2)
                    speed = distance / delta_time  # Tính tốc độ

                    print(f"Tốc độ di chuyển của mũi: {speed:.2f} pixel/giây")
                else:
                    print("Không đủ dữ liệu để tính tốc độ.")

                # Cập nhật giá trị cho khung hình tiếp theo
                previous_nose = (nose_x, nose_y)
                previous_time = current_time
