This repo documents my attempts to improve a YOLOv11 instance segmentation model using FiftyOne.

The initial model was trained using a dataset containing 1920x1080 images from roadside surveys 
to monitor coconut rhinoceros beetles on Guam.

#### model training results

<nbsp>| <nbsp>
--- | ---
created with | yolo segment train data=./data.yaml model=yolov8-seg.pt epochs=1000 imgsz=960 patience=300
epochs | 766
training time | 0.281 hours
mAP50 | 0.442
mAP50-95 | 0.304
directory | /home/aubrey/Desktop/sam/runs/segment/train14
model| /home/aubrey/Desktop/sam/runs/segment/train14/weights/best.pt
---



```bash
# activate a virtual environment containing YOLO training tools from Ultralytics
cd /home/aubrey/Desktop/sam
source .venv/bin/activate

# move to the training dataset directory
cd /home/aubrey/Desktop/Guam07-merged_results/YOLO-prepped

# train
yolo segment train \
data=./dataset.yaml \
model=/home/aubrey/Desktop/yolo11n-seg.pt \
epochs=1000 \
imgsz=960 \
patience=300
```

```
Validating /home/aubrey/Desktop/sam/runs/segment/train16/weights/best.pt...
Ultralytics 8.3.192 🚀 Python-3.12.11 torch-2.8.0+cu128 CUDA:0 (NVIDIA GeForce RTX 3080 Laptop GPU, 15993MiB)
YOLO11n-seg summary (fused): 113 layers, 2,835,348 parameters, 0 gradients, 9.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 7.5it/s 0.1s
                   all         26        103      0.437       0.42      0.436      0.323      0.423       0.41      0.428      0.302
               healthy         21         49      0.564      0.502      0.512      0.381      0.565      0.503      0.512      0.375
                  dead          2          3       0.25      0.333      0.349       0.28      0.249      0.333      0.349      0.314
                  vcut         16         27      0.362      0.259      0.261      0.145      0.309      0.222      0.228     0.0915
               damaged         17         24      0.573      0.583      0.623      0.488      0.569      0.583      0.623      0.426
```

```bash
# activate a virtual environment containing YOLO training tools from Ultralytics
cd /home/aubrey/Desktop/sam
source .venv/bin/activate

# move to the training dataset directory
cd /home/aubrey/Desktop/Guam07-merged_results/YOLO-prepped

# train
yolo segment train \
data=./dataset.yaml \
model=/home/aubrey/Desktop/yolo11s-seg.pt \
epochs=10000 \
imgsz=960 \
patience=300
```
```
EarlyStopping: Training stopped early as no improvement observed in last 300 epochs. Best results observed at epoch 303, best model saved as best.pt.
To update EarlyStopping(patience=300) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

603 epochs completed in 0.435 hours.
Optimizer stripped from /home/aubrey/Desktop/sam/runs/segment/train18/weights/last.pt, 20.6MB
Optimizer stripped from /home/aubrey/Desktop/sam/runs/segment/train18/weights/best.pt, 20.6MB

Validating /home/aubrey/Desktop/sam/runs/segment/train18/weights/best.pt...
Ultralytics 8.3.192 🚀 Python-3.12.11 torch-2.8.0+cu128 CUDA:0 (NVIDIA GeForce RTX 3080 Laptop GPU, 15993MiB)
YOLO11s-seg summary (fused): 113 layers, 10,068,364 parameters, 0 gradients, 32.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 4.9it/s 0.2s
                   all         26        103       0.45      0.606      0.549      0.326      0.441      0.598       0.53      0.329
               healthy         21         49      0.531       0.37      0.533      0.367       0.53      0.368      0.516      0.333
                  dead          2          3      0.396      0.667      0.514      0.261      0.399      0.667      0.514      0.413
                  vcut         16         27      0.457      0.556      0.474      0.255      0.399      0.481      0.411      0.162
               damaged         17         24      0.416      0.833      0.674      0.423      0.437      0.875      0.679      0.408
Speed: 0.3ms preprocess, 4.9ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to /home/aubrey/Desktop/sam/runs/segment/train18
```

```bash
# activate a virtual environment containing YOLO training tools from Ultralytics
cd /home/aubrey/Desktop/sam
source .venv/bin/activate

# move to the training dataset directory
cd /home/aubrey/Desktop/Guam07-merged_results/YOLO-prepped

# train
# added batch=8 when GPU ran out of memory when training with yolo11m-seg.pt
yolo segment train \
data=./dataset.yaml \
model=/home/aubrey/Desktop/yolo11m-seg.pt \
epochs=10000 \
imgsz=960 \
patience=300 \
batch=8
```
```
EarlyStopping: Training stopped early as no improvement observed in last 300 epochs. Best results observed at epoch 206, best model saved as best.pt.
To update EarlyStopping(patience=300) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

506 epochs completed in 0.730 hours.
Optimizer stripped from /home/aubrey/Desktop/sam/runs/segment/train22/weights/last.pt, 45.3MB
Optimizer stripped from /home/aubrey/Desktop/sam/runs/segment/train22/weights/best.pt, 45.3MB

Validating /home/aubrey/Desktop/sam/runs/segment/train22/weights/best.pt...
Ultralytics 8.3.192 🚀 Python-3.12.11 torch-2.8.0+cu128 CUDA:0 (NVIDIA GeForce RTX 3080 Laptop GPU, 15993MiB)
YOLO11m-seg summary (fused): 138 layers, 22,338,396 parameters, 0 gradients, 112.9 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 4.9it/s 0.4s
                   all         26        103      0.657      0.402      0.489      0.319      0.657      0.402      0.501      0.333
               healthy         21         49      0.752      0.429      0.609      0.401      0.752      0.429      0.589      0.356
                  dead          2          3      0.507      0.333      0.347      0.243      0.507      0.333       0.43      0.388
                  vcut         16         27      0.798      0.292       0.44       0.24      0.798      0.292       0.44       0.24
               damaged         17         24       0.57      0.553      0.561      0.392       0.57      0.553      0.547      0.346
Speed: 0.3ms preprocess, 12.7ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to /home/aubrey/Desktop/sam/runs/segment/train22
💡 Learn more at https://docs.ultralytics.com/modes/train
```

## Training

dataset | base model | Mask mapAP50 | trained model
:---: | :---: | :---: | :---
YOLO-prepped | yolo11n-seg | 0.408 | /home/aubrey/Desktop/sam/runs/segment/train16/weights/best.pt
YOLO-prepped | yolo11s-seg | 0.530 | /home/aubrey/Desktop/sam/runs/segment/train18/weights/best.pt
YOLO-prepped | yolo11m-seg | 0.501 | /home/aubrey/Desktop/sam/runs/segment/train22/weights/best.pt

## Evaluation

```bash
# activate a virtual environment containing YOLO training tools from Ultralytics
cd /home/aubrey/Desktop/sam
source .venv/bin/activate
# evaluate
yolo segment val \
model=/home/aubrey/Desktop/sam/runs/segment/train16/weights/best.pt \
data=/home/aubrey/Desktop/Guam07-merged_results/YOLO-prepped/dataset.yaml
```

dataset | model | Mask mapAP50
:---: | :---: | :---:
YOLO-prepped | train16/weights/best.pt (yolo11n-seg) | 0.431
YOLO-prepped | train18/weights/best.pt (yolo11s-seg) | 0.526
YOLO-prepped | train22/weights/best.pt (yolo11m-seg) | 0.501













