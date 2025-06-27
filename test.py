from ultralytics import YOLO
import argparse
import os

if __name__ == '__main__':
    model = YOLO('/home/lthpc/student/lzy/yolov8/runs/train/GTSDB/weights/best.pt') # 自己训练结束后的模型权重
    model.val(data='/home/lthpc/student/dataset/GTSDB_night/GTSDB.yaml',
              split='val',  
              imgsz=640,
              batch=1,
              workers=0,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='tt100k_night_enhancement6',
              device='0',
              )


# from ultralytics import YOLO

# # Load the YOLOv9 model
# model = YOLO('/home/lthpc/student/lzy/CCTSDB-weights/YOLOv5/CCTSDB2021/exp15/weights/best.pt') # model = YOLO("custom_model.pt") is failed

# # Export the model to ONNX format
# model.export(format="onnx")  # creates 'yolov9c.onnx'

# # Load the exported ONNX model
# onnx_model = YOLO("yolov9c.onnx")
