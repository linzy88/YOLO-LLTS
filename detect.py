from ultralytics import YOLO
import torch
 
if __name__=="__main__":
    
    pth_path=r"./weights/cntsss.pt"
 
    test_path=r"/home/lthpc/student/dataset/CNTSSS/test"

    device = torch.device('cuda:0')
    model = YOLO(pth_path)
 
    # Predict with the model
    results = model(test_path,save=True,conf=0.6).to(device)
