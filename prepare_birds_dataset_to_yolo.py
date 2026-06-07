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
import numpy as np
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Bird Dataset to YOLO format with diversity sampling rules")
    
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
    
    # --- Diversity Parameters ---
    parser.add_argument('--max_per_folder', type=int, default=150, 
                        help='Maximum number of images to sample from a single folder/video (default: 150)')
    parser.add_argument('--dist_thresh', type=float, default=100.0, 
                        help='Min pixel distance a bird must move to be considered a different pose/frame (default: 100)')
    parser.add_argument('--area_ratio_thresh', type=float, default=0.7, 
                        help='If BBox areas are within this ratio and close spatially, frame is rejected (default: 0.7)')
    
    return parser.parse_args()

def is_too_similar(ann1, ann2, dist_thresh, area_ratio_thresh):
    """
    Checks if two annotations from the same folder are too similar
    based on BBox center distances and area similarity.
    """
    bboxes1 = ann1.get('bbox', [])
    bboxes2 = ann2.get('bbox', [])
    
    if not bboxes1 or not bboxes2:
        return False
        
    for b1 in bboxes1:
        x1, y1, w1, h1 = b1
        cx1, cy1 = x1 + w1 / 2.0, y1 + h1 / 2.0
        a1 = w1 * h1
        
        for b2 in bboxes2:
            x2, y2, w2, h2 = b2
            cx2, cy2 = x2 + w2 / 2.0, y2 + h2 / 2.0
            a2 = w2 * h2
            
            # Check spatial distance between centers
            dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
            
            # Check area similarity (ratio of smallest to largest area)
            min_a, max_a = min(a1, a2), max(a1, a2)
            area_ratio = min_a / max_a if max_a > 0 else 0
            
            # If they are spatially close AND roughly the same size, they are redundant
            if dist < dist_thresh and area_ratio > area_ratio_thresh:
                return True
                
    return False

def round_robin_sample(folder_dict, n_samples, global_accepted, max_per_folder, dist_thresh, area_ratio_thresh):
    """
    Samples items evenly across folders while strictly enforcing diversity constraints.
    """
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
                
            # Check if we hit the absolute maximum for this specific folder
            if len(global_accepted[f]) >= max_per_folder:
                active_folders.remove(f)
                continue
            
            found_valid = False
            
            # Try to find a non-redundant candidate in this folder
            while pointers[f] < len(folder_dict[f]):
                candidate = folder_dict[f][pointers[f]]
                pointers[f] += 1
                
                # Compare against all globally accepted images from this folder
                is_redundant = False
                for accepted_ann in global_accepted[f]:
                    if is_too_similar(candidate, accepted_ann, dist_thresh, area_ratio_thresh):
                        is_redundant = True
                        break
                
                if not is_redundant:
                    sampled.append(candidate)
                    global_accepted[f].append(candidate) # Add to global tracking
                    found_valid = True
                    break # Success! Break to move to the next folder in round-robin
            
            # If we exhausted the folder without finding a valid candidate
            if not found_valid:
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
                print("Warning: More missing images found. Suppressing further warnings...")
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
    print("Applying Spatial Diversity and Redundancy filters...")

    # Global tracking to ensure cross-bucket diversity within the same folder
    global_accepted = defaultdict(list)
    
    sampled_A = round_robin_sample(bucket_A, target_A, global_accepted, args.max_per_folder, args.dist_thresh, args.area_ratio_thresh)
    sampled_B = round_robin_sample(bucket_B, target_B, global_accepted, args.max_per_folder, args.dist_thresh, args.area_ratio_thresh)
    
    # If A or B didn't have enough diverse images, roll the deficit over to C
    deficit = (target_A - len(sampled_A)) + (target_B - len(sampled_B))
    sampled_C = round_robin_sample(bucket_C, target_C + deficit, global_accepted, args.max_per_folder, args.dist_thresh, args.area_ratio_thresh)
    
    final_samples = sampled_A + sampled_B + sampled_C
    random.shuffle(final_samples)
    
    print(f"Actually sampled -> <150px: {len(sampled_A)}, 150-300px: {len(sampled_B)}, >300px: {len(sampled_C)} | Total: {len(final_samples)}")
    
    if len(final_samples) < args.num_samples:
        print(f"\n[!] Warning: Strict diversity settings prevented reaching {args.num_samples} samples.")
        print("[!] Resulted in smaller, but highly diverse dataset.")

    # Prepare output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(args.output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'labels', split), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'annotations'), exist_ok=True)
        
    split_idx = int(len(final_samples) * (1 - args.test_split))
    splits = {
        'train': final_samples[:split_idx],
        'val': final_samples[split_idx:]
    }
    
    print(f"Writing {len(splits['train'])} train and {len(splits['val'])} val samples to {args.output_dir}...")
    
    saved_annotations = {'train': [], 'val': []}
    
    for split_name, samples in splits.items():
        for i, ann in enumerate(samples):
            rel_path = ann['path']
            img_path = os.path.join(args.dataset_dir, rel_path)
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            
            # Create unique filename
            safe_basename = rel_path.replace(os.sep, '_').replace('/', '_')
            base_no_ext = os.path.splitext(safe_basename)[0]
            
            out_img_path = os.path.join(args.output_dir, 'images', split_name, safe_basename)
            out_txt_path = os.path.join(args.output_dir, 'labels', split_name, f"{base_no_ext}.txt")
            
            saved_annotations[split_name].append({
                "original_path": rel_path,
                "new_filename": safe_basename,
                "bbox": ann.get('bbox', []),
                "label": ann.get('label', ['bird'])
            })
            
            shutil.copy(img_path, out_img_path)
            
            yolo_labels = []
            for bbox in ann.get('bbox', []):
                x, y, w, h = bbox
                
                # YOLO format norm
                x_center = min(max((x + w / 2.0) / img_w, 0.0), 1.0)
                y_center = min(max((y + h / 2.0) / img_h, 0.0), 1.0)
                w_norm = min(max(w / img_w, 0.0), 1.0)
                h_norm = min(max(h / img_h, 0.0), 1.0)
                
                yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                
            with open(out_txt_path, 'w') as f:
                f.write('\n'.join(yolo_labels))
                
    # Save JSON records
    for split in ['train', 'val']:
        json_out_path = os.path.join(args.output_dir, 'annotations', f'{split}.json')
        with open(json_out_path, 'w') as f:
            json.dump(saved_annotations[split], f, indent=4)
                
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

    print(f"\nDone! Dataset ready at {args.output_dir}")
    print(f"Sampled annotations saved to {os.path.join(args.output_dir, 'annotations')}")

    # ==========================
    # Final Statistics Reporting
    # ==========================
    sampled_areas = []
    
    for ann in final_samples:
        for bbox in ann.get('bbox', []):
            x, y, w, h = bbox
            sampled_areas.append(w * h)
            
    if sampled_areas:
        areas_arr = np.array(sampled_areas)
        print("\n--- Final Sampled BBox Statistics (Area in px) ---")
        print(f"{'Count:':<10} {len(areas_arr)}")
        print(f"{'Mean:':<10} {areas_arr.mean():.2f}")
        print(f"{'Std:':<10} {areas_arr.std():.2f}")
        print(f"{'Min:':<10} {areas_arr.min():.2f}")
        print(f"{'25%:':<10} {np.percentile(areas_arr, 25):.2f}")
        print(f"{'50%:':<10} {np.percentile(areas_arr, 50):.2f}")
        print(f"{'75%:':<10} {np.percentile(areas_arr, 75):.2f}")
        print(f"{'Max:':<10} {areas_arr.max():.2f}")
    else:
        print("\nNo bounding boxes found in sampled data.")
        
    print(f"\nNumber of unique folders sampled from: {len(global_accepted)}")

if __name__ == "__main__":
    prepare_dataset()