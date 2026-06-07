"""
Usage:
# Download TIB NET dataset from github releases 
mkdir data
cd data
wget https://github.com/dekelcohen/my_dl_drone_detector_tracker/releases/download/dataset_tag/uav-20260530T222902Z-3-001.zip
python -m zipfile -e uav-20260530T222902Z-3-001.zip .
cd ..

# Prepare the dataset into YOLO format
python prepare_tib_dataset_to_yolo.py --dataset_dir ./data/uav --output_dir ./yolo_dataset --test_split 0.1 --random_seed 42
"""

import os
import glob
import shutil
import random
import argparse
import xml.etree.ElementTree as ET

def parse_args():
    parser = argparse.ArgumentParser(description="Convert TIB-Net UAV VOC dataset to YOLO format")
    
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='Path to the root of TIB_NET_uav (contains Annotations/ and JPEGImages/)')
    parser.add_argument('--output_dir', type=str, default='./yolo_dataset', 
                        help='Directory where the YOLO format dataset will be generated')
    
    parser.add_argument('--recreate_dataset', action='store_true', 
                        help='Force recreation of the YOLO dataset if it already exists (default: False)')
    parser.add_argument('--test_split', type=float, default=0.1, 
                        help='Fraction of the dataset to be used for testing/validation (default: 0.1)')
    parser.add_argument('--random_seed', type=int, default=42, 
                        help='Random seed for dataset splitting (default: 42)')
    
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

def prepare_dataset(dataset_dir, output_dir, test_split, random_seed):
    classes = ['uav']
    
    anno_dir = os.path.join(dataset_dir, 'Annotations')
    img_dir = os.path.join(dataset_dir, 'JPEGImages')
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
    xml_files = glob.glob(os.path.join(anno_dir, '*.xml'))
    if not xml_files:
        raise ValueError(f"No XML files found in {anno_dir}")
        
    random.seed(random_seed)
    random.shuffle(xml_files)
    
    split_idx = int(len(xml_files) * (1 - test_split))
    train_files = xml_files[:split_idx]
    val_files = xml_files[split_idx:]
    
    print(f"Found {len(xml_files)} total valid files.")
    print(f"Splitting into {len(train_files)} train and {len(val_files)} test/val samples (seed: {random_seed}).")
    
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

def main():
    args = parse_args()
    
    expected_yaml = os.path.join(args.output_dir, 'data.yaml')
    
    if os.path.exists(expected_yaml) and not args.recreate_dataset:
        print(f"--- Dataset already exists at '{args.output_dir}' ---")
        print("Skipping dataset generation. (Use --recreate_dataset to force rebuilding)")
        yaml_path = expected_yaml
    else:
        print("--- Preparing Dataset ---")
        if os.path.exists(args.output_dir):
            print(f"Cleaning up existing directory: {args.output_dir}...")
            shutil.rmtree(args.output_dir)
            
        yaml_path = prepare_dataset(args.dataset_dir, args.output_dir, args.test_split, args.random_seed)
    
    print(f"Data configuration saved to: {yaml_path}")

if __name__ == '__main__':
    main()