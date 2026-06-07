"""
Python script tailored to merge any two YOLO datasets. It automatically solves the issues below:
1) Filename Collisions: It prefixes images and labels with ds1_ and ds2_ so 0.jpg from both datasets don't overwrite each other.
2) Class ID Shifting: It reads the data.yaml of both datasets. If Dataset 1 has ['uav'] (ID 0) and Dataset 2 has ['bird'] (ID 0), it will automatically assign bird to ID 1 and safely rewrite all the .txt label files in Dataset 2 to reflect this new ID. (Bonus: If both datasets share a class name, it will merge them into the same ID!)
3) Generates a Unified data.yaml: It produces a new configuration file ready for training.

Usage:
python merge_yolo_datasets.py \
    --ds1 ./yolo_dataset \
    --ds2 ./yolo_birds_dataset \
    --output ./yolo_merged_uav_birds
"""

import os
import shutil
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Merge two YOLO format datasets safely.")
    
    parser.add_argument('--ds1', type=str, required=True, 
                        help='Path to the first YOLO dataset (must contain data.yaml)')
    parser.add_argument('--ds2', type=str, required=True, 
                        help='Path to the second YOLO dataset (must contain data.yaml)')
    parser.add_argument('--output', type=str, required=True, 
                        help='Path to the output directory for the merged dataset')
    
    return parser.parse_args()

def normalize_classes(names_data):
    """Normalizes the 'names' field from data.yaml into a dictionary {id: 'class_name'}"""
    if isinstance(names_data, list):
        return {i: str(name) for i, name in enumerate(names_data)}
    elif isinstance(names_data, dict):
        return {int(k): str(v) for k, v in names_data.items()}
    else:
        raise ValueError("Unknown 'names' format in yaml file.")

def build_class_mapping(ds1_classes, ds2_classes):
    """
    Creates a unified class dictionary and maps old IDs to new IDs.
    Returns: merged_classes_list, ds1_id_map, ds2_id_map
    """
    merged_classes = {}
    ds1_map = {}
    ds2_map = {}
    current_id = 0

    # 1. Map Dataset 1 classes
    for old_id, name in ds1_classes.items():
        merged_classes[current_id] = name
        ds1_map[old_id] = current_id
        current_id += 1

    # 2. Map Dataset 2 classes (Checking for shared classes)
    for old_id, name in ds2_classes.items():
        # Check if this class name already exists from Dataset 1
        existing_id = None
        for m_id, m_name in merged_classes.items():
            if m_name.lower() == name.lower():
                existing_id = m_id
                break
        
        if existing_id is not None:
            # Shared class, map to the existing ID
            ds2_map[old_id] = existing_id
        else:
            # New class, create a new ID
            merged_classes[current_id] = name
            ds2_map[old_id] = current_id
            current_id += 1
            
    # Convert merged_classes to a flat list for the output yaml
    merged_classes_list = [merged_classes[i] for i in range(len(merged_classes))]
    
    return merged_classes_list, ds1_map, ds2_map

def copy_and_remap_dataset(ds_dir, out_dir, prefix, class_map):
    """Copies images and rewrites label files with updated class IDs."""
    print(f"Processing dataset: {ds_dir} ...")
    
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(ds_dir, 'images', split)
        lbl_dir = os.path.join(ds_dir, 'labels', split)
        
        if not os.path.exists(img_dir):
            continue
            
        out_img_dir = os.path.join(out_dir, 'images', split)
        out_lbl_dir = os.path.join(out_dir, 'labels', split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)
        
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in images:
            # 1. Copy Image with prefix to avoid overwriting
            src_img = os.path.join(img_dir, img_name)
            dst_img = os.path.join(out_img_dir, f"{prefix}{img_name}")
            shutil.copy(src_img, dst_img)
            
            # 2. Process corresponding Label file
            base_name = os.path.splitext(img_name)[0]
            src_lbl = os.path.join(lbl_dir, f"{base_name}.txt")
            dst_lbl = os.path.join(out_lbl_dir, f"{prefix}{base_name}.txt")
            
            if os.path.exists(src_lbl):
                with open(src_lbl, 'r') as f_in, open(dst_lbl, 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        
                        # Read old class ID, map it to new class ID
                        old_id = int(parts[0])
                        new_id = class_map.get(old_id, old_id) # fallback to old_id just in case
                        
                        # Reconstruct the line
                        new_line = f"{new_id} {' '.join(parts[1:])}\n"
                        f_out.write(new_line)

def main():
    args = parse_args()
    
    # Check for Yaml files
    yaml1_path = os.path.join(args.ds1, 'data.yaml')
    yaml2_path = os.path.join(args.ds2, 'data.yaml')
    
    if not os.path.exists(yaml1_path):
        raise FileNotFoundError(f"data.yaml not found in {args.ds1}")
    if not os.path.exists(yaml2_path):
        raise FileNotFoundError(f"data.yaml not found in {args.ds2}")
        
    # Read classes
    with open(yaml1_path, 'r') as f1, open(yaml2_path, 'r') as f2:
        ds1_yaml = yaml.safe_load(f1)
        ds2_yaml = yaml.safe_load(f2)
        
    ds1_classes = normalize_classes(ds1_yaml.get('names', []))
    ds2_classes = normalize_classes(ds2_yaml.get('names', []))
    
    # Build maps
    merged_names, map1, map2 = build_class_mapping(ds1_classes, ds2_classes)
    
    print("\n--- Class Mapping Resolution ---")
    print(f"Dataset 1 Original: {ds1_classes}")
    print(f"Dataset 1 Map: {map1}")
    print(f"Dataset 2 Original: {ds2_classes}")
    print(f"Dataset 2 Map: {map2}")
    print(f"Final Merged Classes: {merged_names}\n")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process both datasets
    copy_and_remap_dataset(args.ds1, args.output, prefix="ds1_", class_map=map1)
    copy_and_remap_dataset(args.ds2, args.output, prefix="ds2_", class_map=map2)
    
    # Write merged data.yaml
    merged_yaml_path = os.path.join(args.output, 'data.yaml')
    yaml_content = f"""path: {os.path.abspath(args.output)}
train: images/train
val: images/val

nc: {len(merged_names)}
names: {merged_names}
"""
    with open(merged_yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"\nSuccessfully merged datasets into: {args.output}")
    print(f"You can now train using: yolo train data={merged_yaml_path}")

if __name__ == '__main__':
    main()