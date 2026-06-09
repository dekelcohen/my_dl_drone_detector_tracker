"""
Usage:
# Train and Eval    
python train_yolo.py --data_yaml ./yolo_dataset/data.yaml --model yolo26s.pt --imgsz 1920 --batch_size -1 --epochs 50 --random_seed 42    

# Run evaluation/validation ONLY on an existing model
python train_yolo.py --data_yaml ./yolo_dataset/data.yaml --model ./runs/detect/TIB_NET_UAV/yolo26_train/weights/best.pt --val

# Run prediction only using a trained model
python train_yolo.py --predict ./data/uav/JPEGImages/sample.jpg --model ./runs/detect/TIB_NET_UAV/yolo26_train/weights/best.pt --output-overlay ./predict_result_yolo.jpg --predict-threshold 0.5
"""

import os
import argparse
import cv2
import random
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train, Validate, or Predict using Ultralytics YOLO26")
    
    # --- Training/Validation parameters ---
    parser.add_argument('--data_yaml', type=str, default=None, 
                        help='Path to the YOLO data.yaml configuration file (e.g., ./yolo_dataset/data.yaml)')
    parser.add_argument('--model', type=str, default='yolo26n.pt', 
                        help='Ultralytics YOLO model version or local path to weights (default: yolo26n.pt)')
    parser.add_argument('--val', action='store_true',
                        help='Run validation only on the dataset provided in --data_yaml (skips training)')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=1080, 
                        help='Image resolution for training (high res recommended for 500m drones)')
    parser.add_argument('--batch_size', type=int, default=-1, 
                        help='Batch size. Set to -1 for AutoBatch to maximize GPU memory (default: -1)')
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='Initial learning rate (default: 0.01)')
    parser.add_argument('--optimizer', type=str, default='auto', choices=['auto', 'SGD', 'Adam', 'AdamW', 'MuSGD'],
                        help='Optimizer to use (auto will pick best)')
    parser.add_argument('--save_period', type=int, default=-1, 
                        help='Save checkpoint every x epochs (default: -1, only saves best/last)')
    parser.add_argument('--random_seed', type=int, default=42, 
                        help='Random seed for training reproducibility (default: 42)')
    
    # --- Prediction / Inference ---
    parser.add_argument('--predict', type=str, default=None,
                        help='Path to an image to run prediction on')
    parser.add_argument('--output-overlay', type=str, default=None,
                        help='Path to save the annotated output image with predicted bboxes')
    parser.add_argument('--predict-threshold', type=float, default=0.25,
                        help='Confidence threshold to display and annotate detections (default: 0.25)')
    
    return parser.parse_args()

def predict(model_path, image_path, output_path, conf_threshold):
    """Loads a YOLO model, runs prediction on the target image, and saves the overlay."""
    print(f"\n--- Running Prediction ---")
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Running inference on image: {image_path} (Confidence threshold: {conf_threshold})")
    results = model.predict(source=image_path, conf=conf_threshold)
    
    # Process the prediction results (expecting one result since source is a single image)
    for result in results:
        # result.plot() handles drawing bboxes and confidence labels automatically
        annotated_img = result.plot(conf=True, line_width=2)
        
        # Determine output file path
        if not output_path:
            base, ext = os.path.splitext(image_path)
            output_file = f"{base}_overlay{ext}"
        else:
            output_file = output_path
            
        # Ensure output directory exists
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        cv2.imwrite(output_file, annotated_img)
        print(f"Successfully saved annotated overlay image to: {output_file}")

def main():
    args = parse_args()
    
    # Apply global random seed for predictability
    random.seed(args.random_seed)
    
    # Validation check: Ensure either training data or predict target is supplied
    if not args.data_yaml and not args.predict:
        print("Error: You must provide either --data_yaml (for training/evaluation) or --predict (for inference).")
        return

    # Track active model path for post-training inference
    active_model_path = args.model
    
    if args.data_yaml:
        print(f"\n--- Initializing YOLO Model ({args.model}) ---")
        model = YOLO(args.model)
        
        # Skip training if --val flag is provided
        if not args.val:
            print(f"\n--- Training Model (Seed: {args.random_seed}) ---")
            model.train(
                data=args.data_yaml,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch_size,     
                lr0=args.lr,               
                optimizer=args.optimizer,  
                save_period=args.save_period, 
                seed=args.random_seed,     # Pass seed natively to ultralytics
                val=True,                  
                project='TIB_NET_UAV',
                name='yolo26_train',
                exist_ok=True
            )
            
            # Post-training, we direct inference to use the newly trained best weights
            active_model_path = os.path.join('TIB_NET_UAV', 'yolo26_train', 'weights', 'best.pt')
        else:
            print("\n--- Skipping Training (Validation Only Mode) ---")
            
        print("\n--- Evaluating Model on Validation Set ---")
        metrics = model.val(data=args.data_yaml)
        
        print("\n--- Evaluation Metrics ---")
        print(f"mAP@50-95: {metrics.box.map:.4f}")
        print(f"mAP@50:    {metrics.box.map50:.4f}")
        print(f"mAP@75:    {metrics.box.map75:.4f}")
    
    # --- Inference Execution ---
    if args.predict:
        predict(
            model_path=active_model_path,
            image_path=args.predict,
            output_path=args.output_overlay,
            conf_threshold=args.predict_threshold
        )

if __name__ == '__main__':
    main()