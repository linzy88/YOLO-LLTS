import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"./YOLO-LLTS.yaml")

    model.train(
        data=r'/home/lthpc/student/dataset/GTSDB_night/GTSDB.yaml',
        cache=False,
        imgsz=640,
        epochs=300,
        single_cls=False,
        batch=48,
        workers=0,
        device='0,1,2,3',
        project='runs/train',
        name='exp',
    )