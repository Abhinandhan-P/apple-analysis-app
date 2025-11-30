# Apple Ripeness Training Instructions (Google Colab)

Follow these steps to train your YOLOv8 model for Apple Ripeness Classification.

## 1. Open Google Colab
Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

## 2. Set up GPU
1. Click on **Runtime** > **Change runtime type**.
2. Select **T4 GPU** (or any available GPU).
3. Click **Save**.

## 3. Install YOLOv8
Copy and paste this into the first code cell and run it:
```python
!pip install ultralytics
from ultralytics import YOLO
import os
from IPython.display import display, Image
```

## 4. Download Dataset
Copy and paste this into the next cell to download your Ripeness dataset:
```bash
!curl -L "https://app.roboflow.com/ds/eWgB1pkPyp?key=eV3V9eKikW" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

## 5. Train the Model
Run this cell to start training. We'll train for 50 epochs to ensure good accuracy.
```python
# Load a pre-trained model
model = YOLO('yolov8n.pt')

# Train the model
# imgsz=640 is standard
# epochs=50 should be sufficient for a good start
results = model.train(data='/content/data.yaml', epochs=50, imgsz=640)
```

## 6. Download the Best Model
After training finishes, run this to download your trained weights:
```python
from google.colab import files
files.download('/content/runs/detect/train/weights/best.pt')
```

## 7. Next Steps
Once you have downloaded `best.pt`:
1. Rename it to `ripeness_best.pt` (so we don't confuse it with the previous model).
2. Place it in your project folder `ultrainstinct/runs/detect/train/weights/` (or just in the root `ultrainstinct` folder).
3. Let me know, and I will update the app to use this new model!
