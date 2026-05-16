import os
from ultralytics import YOLO

def main():
    # Define dataset path
    data_dir = r"D:\Arohan_Softwares\Google projects\Leaf_Detection\yolo_dataset"
    
    # Check if the path exists
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory not found at {data_dir}")
        return

    print("Loading YOLOv8n-cls model...")
    # Load a pre-trained YOLOv8 classification model (nano size for speed)
    model = YOLO('yolov8n-cls.pt')

    print(f"Starting training on dataset at: {data_dir}")
    # Train the model 
    # YOLO classification uses folder structure directly: train/, val/, test/
    # imgsz=224 is standard for ResNet and YOLO classification
    results = model.train(
        data=data_dir,
        epochs=15,  # You can adjust this based on how long it takes
        imgsz=224,
        batch=16,   # Adjust if memory allows
        device='0', # Try to use GPU if available, otherwise switch to 'cpu'
        project='runs/classify',
        name='leaf_disease_model'
    )
    
    print("\nTraining completed!")
    print(f"Best model weights saved to: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()
