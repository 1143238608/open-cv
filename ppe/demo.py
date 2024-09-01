import cv2
import torch


class PPE:
    def __init__(self):
        self.model = torch.hub.load('../yolov5', 'custom', path='./yolov5s.pt', source='local')
        self.model.conf = 0.4
        video_path = './sd1660272102_2.mp4'
        self.cap = cv2.VideoCapture(video_path)

    def detect(self):
        while True:
            ret, frame = self.cap.read()

            # 画面反转
            # frame = cv2.flip(frame, 1)
            # 画面转为RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 推理过程
            results = self.model(frame_rgb)
            results_np = results.pandas().xyxy[0].to_numpy()
            print(results_np)

            # 绘制边界框
            for box in results_np:
                l, t, r, b = box[:4].astype('int')
                label_id = box[5]
                if label_id == 0:
                    cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (l, t), (r, b), (255, 0, 255), 2)
            # 显示画面
            cv2.imshow('demo', frame)

            if cv2.waitKey(10) & 0xff == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()


ppe = PPE()
ppe.detect()
