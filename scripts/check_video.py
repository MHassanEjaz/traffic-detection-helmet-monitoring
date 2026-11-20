import cv2

cap = cv2.VideoCapture("test_videos/3.mp4")

print("Video opened:", cap.isOpened())
print("Total frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    count += 1

print("Successfully read frames:", count)

cap.release()