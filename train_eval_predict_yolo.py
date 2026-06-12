"""
Usage:
# Train and Eval    
python train_yolo.py --data_yaml ./yolo_dataset/data.yaml --model yolo26s.pt --imgsz 1920 --batch_size -1 --epochs 50 --random_seed 42    

# Run evaluation/validation ONLY on an existing model
python train_yolo.py --data_yaml ./yolo_dataset/data.yaml --model ./runs/detect/TIB_NET_UAV/yolo26_train/weights/best.pt --val

# Run prediction on a single image
python train_yolo.py --predict ./data/uav/JPEGImages/sample.jpg --model ./runs/detect/TIB_NET_UAV/yolo26_train/weights/best.pt --output-overlay ./predict_result_yolo.jpg --predict-threshold 0.5

# Run prediction on a folder of images (outputs written to the folder provided)
python train_yolo.py --predict ./data/uav/JPEGImages/ --model ./runs/detect/TIB_NET_UAV/yolo26_train/weights/best.pt --output-overlay ./predictions_out/

# If --output-overlay is omitted, results are saved under "predictions_outputs" next to the image(s):
#  - Single image: <img_dir>/predictions_outputs/<image_name>
#  - Folder:       <input_folder>/predictions_outputs/<image_name>
"""

import os
import argparse
import cv2
import random
from ultralytics import YOLO

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


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
                        help='Path to an image file OR a folder of images to run prediction on')
    parser.add_argument('--output-overlay', type=str, default=None,
                        help='For single image: output file path or a directory. For folder input: a directory to write annotated images')
    parser.add_argument('--predict-threshold', type=float, default=0.25,
                        help='Confidence threshold to display and annotate detections (default: 0.25)')
    
    return parser.parse_args()


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def _gather_images(source_path: str):
    """Return a list of image paths from a file or a directory (non-recursive)."""
    if os.path.isdir(source_path):
        entries = [os.path.join(source_path, f) for f in os.listdir(source_path)]
        return [p for p in entries if os.path.isfile(p) and _is_image_file(p)]
    if os.path.isfile(source_path) and _is_image_file(source_path):
        return [source_path]
    raise FileNotFoundError(f"No images found at: {source_path}")


def _resolve_output_for_single(image_path: str, output_overlay: str | None):
    """Return (output_file, output_dir) for a single image case.
    - If output_overlay is a path with an image extension -> treat as file path
    - Else treat as directory and place file with original name inside
    - If None -> use <img_dir>/predictions_outputs/<image_name>
    """
    image_name = os.path.basename(image_path)
    if not output_overlay:
        out_dir = os.path.join(os.path.dirname(image_path), 'predictions_outputs')
        return os.path.join(out_dir, image_name), out_dir

    lower = output_overlay.lower()
    if lower.endswith(IMAGE_EXTS):
        out_dir = os.path.dirname(output_overlay)
        return output_overlay, out_dir
    # Treat as directory
    out_dir = output_overlay
    return os.path.join(out_dir, image_name), out_dir


def _resolve_output_dir_for_folder(input_dir: str, output_overlay: str | None):
    """Return an output directory for the folder case.
    - If provided -> use as directory
    - If None -> <input_dir>/predictions_outputs
    """
    if output_overlay:
        return output_overlay
    return os.path.join(input_dir, 'predictions_outputs')


def predict(model_path: str, source_path: str, output_path: str | None, conf_threshold: float):
    """Load a YOLO model, run prediction on one image or a folder, and save overlays."""
    print(f"\n--- Running Prediction ---")
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    images = _gather_images(source_path)
    if not images:
        raise FileNotFoundError(f"No images found to run prediction at: {source_path}")

    multiple = len(images) > 1 or os.path.isdir(source_path)
    if multiple:
        # Folder-style output handling
        input_dir = source_path if os.path.isdir(source_path) else os.path.dirname(images[0])
        out_dir = _resolve_output_dir_for_folder(input_dir, output_path)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Running inference on {len(images)} images (conf={conf_threshold}). Output dir: {out_dir}")
        results = model.predict(source=images, conf=conf_threshold)
        for img_path, result in zip(images, results):
            annotated_img = result.plot(conf=True, line_width=2)
            out_file = os.path.join(out_dir, os.path.basename(img_path))
            cv2.imwrite(out_file, annotated_img)
            print(f"Saved: {out_file}")
    else:
        image_path = images[0]
        print(f"Running inference on image: {image_path} (conf={conf_threshold})")
        results = model.predict(source=image_path, conf=conf_threshold)
        for result in results:
            annotated_img = result.plot(conf=True, line_width=2)
            out_file, out_dir = _resolve_output_for_single(image_path, output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(out_file, annotated_img)
            print(f"Successfully saved annotated overlay image to: {out_file}")


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
            source_path=args.predict,
            output_path=args.output_overlay,
            conf_threshold=args.predict_threshold
        )


if __name__ == '__main__':
    main()