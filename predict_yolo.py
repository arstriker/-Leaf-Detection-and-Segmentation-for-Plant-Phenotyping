import os
import argparse
from ultralytics import YOLO
import cv2

def predict_image(model_path, image_path):
    """
    Runs YOLOv8 classification inference on a single image.
    """
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        print("Please check if the training has completed and generated 'best.pt' in the runs/classify folder.")
        return

    # Load the trained model
    model = YOLO(model_path)

    print(f"Running inference on: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Run inference
    results = model(image_path)

    # Process the results
    result = results[0]  # Get the first result (we only passed one image)
    
    # Get the top prediction
    top_class_id = result.probs.top1
    top_class_name = result.names[top_class_id]
    top_confidence = result.probs.top1conf.item()

    print("\n--- Prediction Results ---")
    print(f"Class: {top_class_name}")
    print(f"Confidence: {top_confidence:.4f} ({top_confidence * 100:.2f}%)")
    print("--------------------------\n")

    # Optional: Display the top 5 predictions
    print("Top 5 Predictions:")
    top5_ids = result.probs.top5
    top5_confs = result.probs.top5conf.tolist()
    for idx, conf in zip(top5_ids, top5_confs):
        print(f"  - {result.names[idx]}: {conf:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLOv8 Classification Inference")
    parser.add_argument('--model', type=str, 
                        default=r"runs\classify\leaf_disease_model2\weights\best.pt", 
                        help="Path to the trained best.pt model file")
    parser.add_argument('--image', type=str, 
                        default=r"D:\Arohan_Softwares\Google projects\Leaf_Detection\yolo_dataset\test\Tomato___Early_blight\0012b9d2-2130-4a06-a834-b1f3af34f57e___RS_Erly.B 8389.JPG", 
                        help="Path to the image to classify")
    
    args = parser.parse_args()
    
    predict_image(args.model, args.image)
