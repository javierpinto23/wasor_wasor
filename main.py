from ultralytics import YOLO
import cv2

# Read the model
model = YOLO("/Users/pintojav/Desktop/wasor_wasor/wasor_wasor/wasor_wasor/runs/weights/best.pt")
# Capture the video
cap = cv2.VideoCapture("http://192.168.18.16:8080/video")

while True:
    # Read the frames
    ret, frame = cap.read()

    # Read the results
    results = model.predict(frame, imgsz = 640)

    # Display the results
    annotations = results[0].plot()

    # Display the frames and the results video
    cv2.imshow("WASOR", annotations)

    # Close the program
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()