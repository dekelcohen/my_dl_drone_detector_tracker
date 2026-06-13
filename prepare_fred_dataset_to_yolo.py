"""
Usage: 
python prepare_fred_dataset_to_yolo.py --dataset_dir "E:\Vision\Drones\data\Datasets\FRED" --output_dir ./fred_yolo_small_bboxes --max_area 60.0 --num_samples 2000
"""

import os
import re
import json
import random
import shutil
import argparse
import cv2
import numpy as np
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Small Drone/Bird BBoxes to YOLO format")
    
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='Path to the root of the dataset containing FRED/ folders')
    parser.add_argument('--output_dir', type=str, default='./yolo_small_bboxes', 
                        help='Directory where the YOLO format dataset will be generated')
    parser.add_argument('--max_area', type=float, default=60.0, 
                        help='Maximum area of bbox (W * H) in pixels to be extracted (default: 60.0)')
    parser.add_argument('--num_samples', type=int, default=2000, 
                        help='Number of total images to sample')
    parser.add_argument('--test_split', type=float, default=0.1, 
                        help='Fraction of the dataset to be used for validation (default: 0.1)')
    parser.add_argument('--random_seed', type=int, default=42, 
                        help='Random seed for reproducibility (default: 42)')
    
    # --- Diversity Parameters ---
    parser.add_argument('--max_per_folder', type=int, default=150, 
                        help='Maximum number of images to sample from a single folder/video (default: 150)')
    parser.add_argument('--dist_thresh', type=float, default=100.0, 
                        help='Min pixel distance a bbox must move to be considered a new frame (default: 100)')
    
    return parser.parse_args()

def get_timestamp_from_filename(filename):
    """ Extracts timestamp float from filenames like Video_22_19_01_47.178196.jpg """
    match = re.search(r'_(\d+\.\d+)\.jpg$', filename)
    if match:
        return float(match.group(1))
    return None

def is_too_similar(ann1, ann2, dist_thresh):
    """ Checks if two annotations are spatially too similar to avoid redundant frames """
    bboxes1 = ann1.get('bbox', [])
    bboxes2 = ann2.get('bbox', [])
    
    if not bboxes1 or not bboxes2:
        return False
        
    for b1 in bboxes1:
        x1, y1, w1, h1 = b1
        cx1, cy1 = x1 + w1 / 2.0, y1 + h1 / 2.0
        
        for b2 in bboxes2:
            x2, y2, w2, h2 = b2
            cx2, cy2 = x2 + w2 / 2.0, y2 + h2 / 2.0
            
            # Check spatial distance between centers
            dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
            if dist < dist_thresh:
                return True
                
    return False

def round_robin_sample(folder_dict, n_samples, max_per_folder, dist_thresh):
    """ Samples items evenly across folders while enforcing diversity and dup removal """
    sampled = []
    global_accepted = defaultdict(list)
    folders = list(folder_dict.keys())
    random.shuffle(folders)
    
    for f in folders:
        random.shuffle(folder_dict[f])
        
    pointers = {f: 0 for f in folders}
    active_folders = set(folders)
    
    while len(sampled) < n_samples and active_folders:
        for f in list(active_folders):
            if len(sampled) >= n_samples:
                break
                
            if len(global_accepted[f]) >= max_per_folder:
                active_folders.remove(f)
                continue
            
            found_valid = False
            while pointers[f] < len(folder_dict[f]):
                candidate = folder_dict[f][pointers[f]]
                pointers[f] += 1
                
                is_redundant = False
                for accepted_ann in global_accepted[f]:
                    if is_too_similar(candidate, accepted_ann, dist_thresh):
                        is_redundant = True
                        break
                
                if not is_redundant:
                    sampled.append(candidate)
                    global_accepted[f].append(candidate)
                    found_valid = True
                    break
            
            if not found_valid:
                active_folders.remove(f)
                
    return sampled, global_accepted

def scan_and_extract(dataset_dir, max_area):
    bucket_small = defaultdict(list)
    
    for root, dirs, files in os.walk(dataset_dir):
        if 'coordinates_rgb.txt' in files:
            txt_path = os.path.join(root, 'coordinates_rgb.txt')
            rgb_dir = os.path.join(root, 'RGB')
            
            if not os.path.isdir(rgb_dir):
                print(f"[!] Warning: Missing RGB folder at {rgb_dir}. Skipping.")
                continue
            
            # 1. Pre-fetch and parse all image timestamps from this directory
            images = [img for img in os.listdir(rgb_dir) if img.endswith('.jpg')]
            img_timestamps = []
            for img in images:
                ts = get_timestamp_from_filename(img)
                if ts is not None:
                    img_timestamps.append((ts, img))
                    
            image_bboxes = defaultdict(list)
            image_labels = defaultdict(list)
            
            # 2. Parse text file
            with open(txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    parts = line.split(':')
                    if len(parts) < 2: continue
                        
                    try:
                        ts_val = float(parts[0].strip())
                        bbox_info = parts[1].split(',')
                        
                        x1 = float(bbox_info[0].strip())
                        y1 = float(bbox_info[1].strip())
                        x2 = float(bbox_info[2].strip())
                        y2 = float(bbox_info[3].strip())
                        
                        class_name = bbox_info[5].strip() if len(bbox_info) > 5 else f"class_{bbox_info[4].strip()}"
                        
                        w = abs(x2 - x1)
                        h = abs(y2 - y1)
                        area = w * h
                        
                        if area > max_area:
                            continue  # Ignore large bboxes
                            
                        # Find closest matching image
                        closest_img = None
                        min_diff = float('inf')
                        for img_ts, img_name in img_timestamps:
                            diff = abs(img_ts - ts_val)
                            if diff < min_diff:
                                min_diff = diff
                                closest_img = img_name
                        
                        # Set a max tolerance of 0.1 seconds offset between txt and filename
                        if min_diff < 0.1 and closest_img is not None:
                            image_bboxes[closest_img].append([min(x1, x2), min(y1, y2), w, h])
                            image_labels[closest_img].append(class_name)
                            
                    except ValueError:
                        continue # Skip malformed lines gracefully
                        
            # 3. Group by image 
            for img_name, bboxes in image_bboxes.items():
                ann = {
                    'path': os.path.join(rgb_dir, img_name),
                    'bbox': bboxes,
                    'label': image_labels[img_name]
                }
                bucket_small[root].append(ann)
                
    return bucket_small

def prepare_dataset():
    args = parse_args()
    random.seed(args.random_seed)
    
    print(f"[*] Scanning recursively in {args.dataset_dir} for < {args.max_area}px area bboxes...")
    bucket_small = scan_and_extract(args.dataset_dir, args.max_area)
    
    total_extracted = sum(len(v) for v in bucket_small.values())
    print(f"[*] Total images with small bboxes found: {total_extracted}")
    
    if total_extracted == 0:
        print("[!] No matching small bounding boxes were found. Check your threshold or directory paths.")
        return

    print("[*] Applying Spatial Diversity and Redundancy filters...")
    final_samples, global_accepted = round_robin_sample(
        bucket_small, 
        args.num_samples, 
        args.max_per_folder, 
        args.dist_thresh
    )
    
    print(f"[*] Actually sampled -> Total: {len(final_samples)}")
    
    # Prepare directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(args.output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'labels', split), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'annotations'), exist_ok=True)
        
    split_idx = int(len(final_samples) * (1 - args.test_split))
    splits = {
        'train': final_samples[:split_idx],
        'val': final_samples[split_idx:]
    }
    
    print(f"[*] Writing {len(splits['train'])} train and {len(splits['val'])} val samples to {args.output_dir}...")
    
    saved_annotations = {'train': [], 'val': []}
    class_mapping = {}  # Resolves arbitrary class names/ids to YOLO 0-indexing 
    
    for split_name, samples in splits.items():
        for ann in samples:
            img_path = ann['path']
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            
            # Normalize filename globally to prevent collisions across folders
            rel_path = os.path.relpath(img_path, args.dataset_dir)
            safe_basename = rel_path.replace(os.sep, '_').replace('/', '_').replace(':', '_')
            base_no_ext = os.path.splitext(safe_basename)[0]
            
            out_img_path = os.path.join(args.output_dir, 'images', split_name, safe_basename)
            out_txt_path = os.path.join(args.output_dir, 'labels', split_name, f"{base_no_ext}.txt")
            
            saved_annotations[split_name].append({
                "original_path": img_path,
                "new_filename": safe_basename,
                "bbox": ann.get('bbox', []),
                "label": ann.get('label', [])
            })
            
            shutil.copy(img_path, out_img_path)
            
            yolo_labels = []
            for bbox, label_name in zip(ann.get('bbox', []), ann.get('label', [])):
                x1, y1, w, h = bbox
                
                # Setup mapping
                if label_name not in class_mapping:
                    class_mapping[label_name] = len(class_mapping)
                yolo_id = class_mapping[label_name]
                
                # Norm mapping
                x_center = min(max((x1 + w / 2.0) / img_w, 0.0), 1.0)
                y_center = min(max((y1 + h / 2.0) / img_h, 0.0), 1.0)
                w_norm = min(max(w / img_w, 0.0), 1.0)
                h_norm = min(max(h / img_h, 0.0), 1.0)
                
                yolo_labels.append(f"{yolo_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                
            with open(out_txt_path, 'w') as f:
                f.write('\n'.join(yolo_labels))
                
    # Generate backup JSON annotations (so you don't lose the metadata footprint)
    for split in ['train', 'val']:
        json_out_path = os.path.join(args.output_dir, 'annotations', f'{split}.json')
        with open(json_out_path, 'w') as f:
            json.dump(saved_annotations[split], f, indent=4)
                
    # Data yaml
    yaml_path = os.path.join(args.output_dir, 'data.yaml')
    classes_str = ", ".join(f"'{name}'" for name in class_mapping.keys()) if class_mapping else "'target'"
    yaml_content = f"path: {os.path.abspath(args.output_dir)}\n" \
                   f"train: images/train\n" \
                   f"val: images/val\n\n" \
                   f"nc: {max(1, len(class_mapping))}\n" \
                   f"names: [{classes_str}]\n"
                   
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n[*] Done! Dataset generated at {args.output_dir}")

    # ==========================
    # Final Statistics
    # ==========================
    sampled_areas = []
    for ann in final_samples:
        for bbox in ann.get('bbox', []):
            _, _, w, h = bbox
            sampled_areas.append(w * h)
            
    if sampled_areas:
        areas_arr = np.array(sampled_areas)
        print("\n--- Final Extracted Small BBox Statistics (Area px) ---")
        print(f"{'Count:':<10} {len(areas_arr)}")
        print(f"{'Mean:':<10} {areas_arr.mean():.2f}")
        print(f"{'Std:':<10} {areas_arr.std():.2f}")
        print(f"{'Min:':<10} {areas_arr.min():.2f}")
        print(f"{'25%:':<10} {np.percentile(areas_arr, 25):.2f}")
        print(f"{'50%:':<10} {np.percentile(areas_arr, 50):.2f}")
        print(f"{'75%:':<10} {np.percentile(areas_arr, 75):.2f}")
        print(f"{'Max:':<10} {areas_arr.max():.2f}")
    
    print(f"\n[*] Number of unique folders sampled from: {len(global_accepted)}")

if __name__ == "__main__":
    prepare_dataset()