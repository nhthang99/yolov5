# YOLOv5n
python train.py --weights weights/pretrained_weights/yolov5n.pt \
                --cfg models/yolov5n.yaml \
                --data data/prj_vtcc_logo_recognition.yaml \
                --imgsz 640 \
                --multi-scale \
                --name prj_vtcc_logo_recognition

# YOLOv5s
CUDA_VISIBLE_DEVICES=1
python train.py --weights weights/pretrained_weights/yolov5s.pt \
                --cfg models/yolov5s.yaml \
                --data data/prj_vtcc_logo_recognition.yaml \
                --imgsz 640 \
                --multi-scale \
                --name prj_vtcc_logo_recognition

# YOLOv5m
CUDA_VISIBLE_DEVICES=1,2
python train.py --weights weights/pretrained_weights/yolov5m.pt \
                --cfg models/yolov5m.yaml \
                --data data/prj_vtcc_logo_recognition.yaml \
                --imgsz 640 \
                --multi-scale \
                --name prj_vtcc_logo_recognition
                --batch-size 8

# YOLOv5s6
CUDA_VISIBLE_DEVICES=0,2
python train.py --weights weights/pretrained_weights/yolov5s6.pt \
                --cfg models/hub/yolov5s6.yaml \
                --data data/prj_vtcc_logo_recognition.yaml \
                --imgsz 1280 \
                --multi-scale \
                --name prj_vtcc_logo_recognition