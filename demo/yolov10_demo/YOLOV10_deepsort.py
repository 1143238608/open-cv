import cv2
import numpy as np

video_path = './test_1.mp4'
pts = np.array([(710,200), (1110,200), (810,400), (410, 400)],dtype=np.int32)
if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('ERROR')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(fps, frame_width, frame_height)
    while True:
        success, frame = cap.read()
        if not success:
            print('读取帧失败')
            break
        cv2.line(frame, (0, int(frame_height / 2)), (int(frame_width), int(frame_height / 2)), (0, 0, 255), 2)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
