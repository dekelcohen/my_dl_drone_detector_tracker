"""
Usage:
# Download TIB NET dataset from github releases 
mkdir data
cd data
wget https://github.com/dekelcohen/my_dl_drone_detector_tracker/releases/download/dataset_tag/uav-20260530T222902Z-3-001.zip
python -m zipfile -e uav-20260530T222902Z-3-001.zip .

# Train and Eval    
python yolo_train_voc_tib_net.py --dataset_dir ./data/uav --model yolo26s.pt --imgsz 1920 --batch_size -1 --epochs 50     

# Run prediction only using a trained model
python yolo_train_voc_tib_net.py --predict ./data/uav/JPEGImages/sample.jpg --model ./runs/detect/TIB_NET_UAV/yolo26_train/weights/best.pt --output-overlay ./predict_result_yolo.jpg --predict-threshold 0.5
"""

import os
import glob
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
import cv2
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO26 on TIB-Net UAV dataset or run prediction")
    
    # --- Modified dataset_dir to be optional to support inference-only execution ---
    parser.add_argument('--dataset_dir', type=str, default=None, 
                        help='Path to the root of TIB_NET_uav (contains Annotations/ and JPEGImages/)')
    parser.add_argument('--output_dir', type=str, default='./yolo_dataset', 
                        help='Directory where the YOLO format dataset will be generated')
    
    # --- Dataset Recreation ---
    parser.add_argument('--recreate_dataset', action='store_true', 
                        help='Force recreation of the YOLO dataset if it already exists (default: False)')
    
    parser.add_argument('--test_split', type=float, default=0.1, 
                        help='Fraction of the dataset to be used for testing/validation (default: 0.1)')
    parser.add_argument('--model', type=str, default='yolo26n.pt', 
                        help='Ultralytics YOLO model version or local path to weights (default: yolo26n.pt)')
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
    
    # --- NEW ARGUMENTS: Prediction / Inference ---
    parser.add_argument('--predict', type=str, default=None,
                        help='Path to an image to run prediction on')
    parser.add_argument('--output-overlay', type=str, default=None,
                        help='Path to save the annotated output image with predicted bboxes')
    parser.add_argument('--predict-threshold', type=float, default=0.25,
                        help='Confidence threshold to display and annotate detections (default: 0.25)')
    
    return parser.parse_args()

def convert_voc_to_yolo(xml_path, classes):
    """Parses a VOC XML and returns a list of YOLO format label strings."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    w = float(size.find('width').text)
    h = float(size.find('height').text)
    
    yolo_labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in classes:
            continue
        
        class_id = classes.index(name)
        xmlbox = obj.find('bndbox')
        
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)
        
        x_center = ((xmin + xmax) / 2.0) / w
        y_center = ((ymin + ymax) / 2.0) / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h
        
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
    return yolo_labels

def prepare_dataset(dataset_dir, output_dir, test_split):
    classes = ['uav']
    
    anno_dir = os.path.join(dataset_dir, 'Annotations')
    img_dir = os.path.join(dataset_dir, 'JPEGImages')
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
    xml_files = glob.glob(os.path.join(anno_dir, '*.xml'))
    if not xml_files:
        raise ValueError(f"No XML files found in {anno_dir}")
        
    random.seed(42)
    random.shuffle(xml_files)
    
    split_idx = int(len(xml_files) * (1 - test_split))
    train_files = xml_files[:split_idx]
    val_files = xml_files[split_idx:]
    
    print(f"Found {len(xml_files)} total valid files.")
    print(f"Splitting into {len(train_files)} train and {len(val_files)} test/val samples.")
    
    def process_files(files, split_name):
        for xml_path in files:
            basename = os.path.splitext(os.path.basename(xml_path))[0]
            img_path = os.path.join(img_dir, f"{basename}.jpg")
            
            if not os.path.exists(img_path):
                continue
            
            yolo_labels = convert_voc_to_yolo(xml_path, classes)
            
            txt_path = os.path.join(output_dir, 'labels', split_name, f"{basename}.txt")
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_labels))
                
            out_img_path = os.path.join(output_dir, 'images', split_name, f"{basename}.jpg")
            if not os.path.exists(out_img_path):
                shutil.copy(img_path, out_img_path)

    print("Processing Training set...")
    process_files(train_files, 'train')
    print("Processing Test set...")
    process_files(val_files, 'val')
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    return yaml_path

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
    
    # Validation check: Ensure at least one primary path is designated
    if not args.dataset_dir and not args.predict:
        print("Error: You must provide either --dataset_dir (for training/evaluation) or --predict (for inference).")
        return

    # Track active model path for post-training inference
    active_model_path = args.model
    
    if args.dataset_dir:
        # Check if dataset already exists
        expected_yaml = os.path.join(args.output_dir, 'data.yaml')
        
        if os.path.exists(expected_yaml) and not args.recreate_dataset:
            print(f"--- Step 1: Dataset already exists at '{args.output_dir}' ---")
            print("Skipping dataset generation. (Use --recreate_dataset to force rebuilding)")
            yaml_path = expected_yaml
        else:
            print("--- Step 1: Preparing Dataset ---")
            if os.path.exists(args.output_dir):
                print(f"Cleaning up existing directory: {args.output_dir}...")
                shutil.rmtree(args.output_dir)
                
            yaml_path = prepare_dataset(args.dataset_dir, args.output_dir, args.test_split)
            print(f"Data configuration saved to: {yaml_path}")
        
        print(f"\n--- Step 2: Initializing YOLO26 Model ({args.model}) ---")
        model = YOLO(args.model)
        
        print("\n--- Step 3: Training Model ---")
        model.train(
            data=yaml_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,     
            lr0=args.lr,               
            optimizer=args.optimizer,  
            save_period=args.save_period, 
            val=True,                  
            project='TIB_NET_UAV',
            name='yolo26_train',
            exist_ok=True
        )
        
        # Post-training, we direct inference to use the newly trained best weights
        active_model_path = os.path.join('TIB_NET_UAV', 'yolo26_train', 'weights', 'best.pt')
        
        print("\n--- Step 4: Evaluating Model on Test Set ---")
        metrics = model.val(data=yaml_path)
        
        print("\n--- Evaluation Metrics ---")
        print(f"mAP@50-95: {metrics.box.map:.4f}")
        print(f"mAP@50:    {metrics.box.map50:.4f}")
        print(f"mAP@75:    {metrics.box.map75:.4f}")
    
    # --- Step 5: Inference Execution ---
    if args.predict:
        predict(
            model_path=active_model_path,
            image_path=args.predict,
            output_path=args.output_overlay,
            conf_threshold=args.predict_threshold
        )

if __name__ == '__main__':
    main()