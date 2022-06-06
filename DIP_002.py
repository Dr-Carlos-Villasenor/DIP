# Intensity transformations
import cv2

capture = cv2.VideoCapture(0)

while True:
    isTrue, Frame = capture.read()
    cv2.imshow('video', Frame)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()
