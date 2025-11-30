from ultralytics import YOLO

def train_model():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='c:/Users/ABHINANDHAN/Desktop/ultrainstinct/data.yaml', epochs=20, imgsz=640)

if __name__ == '__main__':
    train_model()
