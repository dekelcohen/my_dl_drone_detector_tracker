"""
Usage:
python prepare_birds_dataset_to_yolo.py --dataset_dir "E:\Vision\Drones\data\Datasets\Birds\Distant Bird Detection for Safe Drone Flight" --output_dir ./yolo_birds_dataset --num_samples 2000
"""

import os
import json
import random
import shutil
import argparse
import cv2
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Bird Dataset to YOLO format with specific sampling rules")
    
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='Path to the root of the Birds dataset containing annotations/ and image folders')
    parser.add_argument('--output_dir', type=str, default='./yolo_birds_dataset', 
                        help='Directory where the YOLO format dataset will be generated')
    parser.add_argument('--num_samples', type=int, default=2000, 
                        help='Number of total images to sample')
    parser.add_argument('--test_split', type=float, default=0.1, 
                        help='Fraction of the sampled dataset to be used for validation (default: 0.1)')
    parser.add_argument('--random_seed', type=int, default=42, 
                        help='Random seed for reproducibility (default: 42)')
    
    return parser.parse_args()

def round_robin_sample(folder_dict, n_samples):
    """Samples items evenly across all available folders to maximize diversity."""
    sampled = []
    folders = list(folder_dict.keys())
    random.shuffle(folders)
    
    # Shuffle the items inside each folder
    for f in folders:
        random.shuffle(folder_dict[f])
        
    pointers = {f: 0 for f in folders}
    active_folders = set(folders)
    
    while len(sampled) < n_samples and active_folders:
        for f in list(active_folders):
            if len(sampled) >= n_samples:
                break
            
            idx = pointers[f]
            if idx < len(folder_dict[f]):
                sampled.append(folder_dict[f][idx])
                pointers[f] += 1
            else:
                active_folders.remove(f)
                
    return sampled

def prepare_dataset():
    args = parse_args()
    random.seed(args.random_seed)
    
    train_json_path = os.path.join(args.dataset_dir, 'annotations', 'train.json')
    val_json_path = os.path.join(args.dataset_dir, 'annotations', 'val.json')
    
    # Combine both annotations to sample from the whole pool
    annotations = []
    for j_path in [train_json_path, val_json_path]:
        if os.path.exists(j_path):
            with open(j_path, 'r') as f:
                annotations.extend(json.load(f))
        else:
            print(f"Warning: Annotation file {j_path} not found.")

    print(f"Total annotations loaded: {len(annotations)}")
    
    # Buckets based on Average BBox Area per image
    # bucket format: dict[folder_name] = [annotation_items]
    bucket_A = defaultdict(list) # < 150 px
    bucket_B = defaultdict(list) # 150 - 300 px
    bucket_C = defaultdict(list) # > 300 px
    
    missing_files_count = 0
    print("Filtering images and categorizing by bounding box sizes...")
    
    for ann in annotations:
        rel_path = ann['path']
        img_path = os.path.join(args.dataset_dir, rel_path)
        
        # Verify if .jpg actually exists
        if not os.path.exists(img_path):
            if missing_files_count < 10:
                print(f"Warning: Image {img_path} not found. Skipping.")
            elif missing_files_count == 10:
                print("Warning: More images not found. Suppressing further warnings...")
            missing_files_count += 1
            continue
            
        bboxes = ann.get('bbox', [])
        if not bboxes:
            continue
            
        # Calculate average area of bboxes in this image
        avg_area = sum(w * h for _, _, w, h in bboxes) / len(bboxes)
        folder = os.path.dirname(rel_path)
        
        if avg_area < 150:
            bucket_A[folder].append(ann)
        elif avg_area <= 300:
            bucket_B[folder].append(ann)
        else:
            bucket_C[folder].append(ann)

    if missing_files_count > 0:
        print(f"Total missing image files skipped: {missing_files_count}")

    # Calculate target samples for each bucket
    target_A = int(args.num_samples * 0.10) # 10%
    target_B = int(args.num_samples * 0.50) # 50%
    target_C = args.num_samples - target_A - target_B # Remaining 40%
    
    print(f"Sampling targets -> <150px: {target_A}, 150-300px: {target_B}, >300px: {target_C}")

    sampled_A = round_robin_sample(bucket_A, target_A)
    sampled_B = round_robin_sample(bucket_B, target_B)
    
    # If A or B didn't have enough images, roll the deficit over to C
    deficit = (target_A - len(sampled_A)) + (target_B - len(sampled_B))
    sampled_C = round_robin_sample(bucket_C, target_C + deficit)
    
    final_samples = sampled_A + sampled_B + sampled_C
    random.shuffle(final_samples)
    
    print(f"Actually sampled -> <150px: {len(sampled_A)}, 150-300px: {len(sampled_B)}, >300px: {len(sampled_C)} | Total: {len(final_samples)}")
    
    if len(final_samples) < args.num_samples:
        print(f"Warning: Only found {len(final_samples)} valid images matching criteria across all datasets.")

    # Prepare output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(args.output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'labels', split), exist_ok=True)
        
    split_idx = int(len(final_samples) * (1 - args.test_split))
    splits = {
        'train': final_samples[:split_idx],
        'val': final_samples[split_idx:]
    }
    
    print(f"Writing {len(splits['train'])} train and {len(splits['val'])} val samples to {args.output_dir}...")
    
    for split_name, samples in splits.items():
        for i, ann in enumerate(samples):
            rel_path = ann['path']
            img_path = os.path.join(args.dataset_dir, rel_path)
            
            # Read image to get width and height for YOLO normalization
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            
            # Create unique filename using folder name to prevent overwrites (e.g., "9_0.jpg")
            safe_basename = rel_path.replace(os.sep, '_').replace('/', '_')
            base_no_ext = os.path.splitext(safe_basename)[0]
            
            out_img_path = os.path.join(args.output_dir, 'images', split_name, safe_basename)
            out_txt_path = os.path.join(args.output_dir, 'labels', split_name, f"{base_no_ext}.txt")
            
            # Copy image
            shutil.copy(img_path, out_img_path)
            
            # Write labels
            yolo_labels = []
            for bbox in ann.get('bbox', []):
                x, y, w, h = bbox
                
                # YOLO format calculations (Clamp values between 0.0 and 1.0 just in case)
                x_center = min(max((x + w / 2.0) / img_w, 0.0), 1.0)
                y_center = min(max((y + h / 2.0) / img_h, 0.0), 1.0)
                w_norm = min(max(w / img_w, 0.0), 1.0)
                h_norm = min(max(h / img_h, 0.0), 1.0)
                
                # class_id is 0 because all objects are 'bird'
                yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                
            with open(out_txt_path, 'w') as f:
                f.write('\n'.join(yolo_labels))
                
    # Create data.yaml
    yaml_path = os.path.join(args.output_dir, 'data.yaml')
    yaml_content = f"""path: {os.path.abspath(args.output_dir)}
train: images/train
val: images/val

nc: 1
names: ['bird']
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"Done! Dataset ready at {args.output_dir}")

if __name__ == "__main__":
    prepare_dataset()