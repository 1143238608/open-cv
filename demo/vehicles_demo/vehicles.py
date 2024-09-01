import cv2
import torch


class PPE:
    def __init__(self):
        # 注意需要与使用的对应yolo项目同级目录
        self.model = torch.hub.load('../../yolov5', 'custom', path='./vehicles.pt', source='local')
        self.model.conf = 0.4
        img_path = './test.jpg'
        self.img = cv2.imread(img_path)
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.num_car = 0

    def detect(self):
        results = self.model(self.img_rgb)
        results_np = results.pandas().xyxy[0].to_numpy()
        # print(results_np)
        vehicles_type = ['big bus', 'big truck', 'bus-l-', 'bus-s-', 'car', 'mid truck', 'small bus', 'small truck',
                         'truck-l-', 'truck-m-', 'truck-s-', 'truck-xl-']
        for box in results_np:
            x1, y1, x2, y2 = box[:4].astype('int')
            label = box[-1]
            if label == 'car': self.num_car += 1
            # print(x1,y1,x2,y2)
            cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(self.img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(self.img, "car number is :" + str(self.num_car), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 255, 150),
                    1)
        cv2.imwrite("test_final.jpg", self.img)


ppe = PPE()
ppe.detect()
