from ultralytics import YOLO
import argparse
import os

if __name__ == '__main__':
    model = YOLO('/home/lthpc/student/lzy/yolov8/runs/train/GTSDB/weights/best.pt') # The trained weight
    model.val(data='/home/lthpc/student/dataset/GTSDB_night/GTSDB.yaml',
              split='val',  
              imgsz=640,
              batch=1,
              workers=0,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='test_gtsdb',
              device='0',
              )
