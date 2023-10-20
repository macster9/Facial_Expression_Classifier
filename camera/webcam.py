import numpy as np
import cv2


def bbox(frame, thickness=5, x_size=220, y_size=220):
    y, x, channels = frame.shape
    x_width1 = int((x/2)-(x_size/2)-(thickness/2))
    x_width2 = int((x/2)-(x_size/2)+(thickness-int(thickness/2)))
    y_width1 = int((y / 2) - (y_size / 2) - (thickness / 2))
    y_width2 = int((y / 2) - (y_size / 2) + (thickness - int(thickness / 2)))
    frame[y_width1:-y_width1, x_width1:x_width2, 2] = np.ones(
        frame[y_width1:-y_width1, x_width1:x_width2, 2].shape) * 255
    frame[y_width1:-y_width1, x_width1:x_width2, 0:1] = np.zeros(
        frame[y_width1:-y_width1, x_width1:x_width2, 0:1].shape)

    frame[y_width1:-y_width1, -x_width2:-x_width1, 2] = np.ones(
        frame[y_width1:-y_width1, -x_width2:-x_width1, 2].shape) * 255
    frame[y_width1:-y_width1, -x_width2:-x_width1, 0:1] = np.zeros(
        frame[y_width1:-y_width1, -x_width2:-x_width1, 0:1].shape)

    frame[y_width1:y_width2, x_width1:-x_width1, 2] = np.ones(
        frame[y_width1:y_width2, x_width1:-x_width1, 2].shape) * 255
    frame[y_width1:y_width2, x_width1:-x_width1, 0:1] = np.zeros(
        frame[y_width1:y_width2, x_width1:-x_width1, 0:1].shape)

    frame[-y_width2:-y_width1, x_width2:-x_width2, 2] = np.ones(
        frame[-y_width2:-y_width1, x_width2:-x_width2, 2].shape) * 255
    frame[-y_width2:-y_width1, x_width2:-x_width2, 0:1] = np.zeros(
        frame[-y_width2:-y_width1, x_width2:-x_width2, 0:1].shape)
    return frame


class WebCam:
    def __init__(self):
        super(WebCam, self).__init__()
        self.on = True
        self.vid = cv2.VideoCapture(0)

    def activate(self):
        print("Webcam activated.")
        while self.on:
            ret, frame = self.vid.read()
            frame = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
            frame = bbox(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.vid.release()
        cv2.destroyAllWindows()

    def deactivate(self):
        print("Webcam deactivated")
        self.on = False
