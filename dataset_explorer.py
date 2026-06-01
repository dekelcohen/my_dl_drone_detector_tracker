#!/usr/bin/env python3
"""
Usage:
python dataset_explorer.py --dataset_dir ./data/uav
"""

import os
import glob
import argparse
import xml.etree.ElementTree as ET
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a Pascal VOC dataset to compute bounding box area distribution statistics."
    )
    parser.add_argument(
        '--dataset_dir', 
        type=str, 
        required=True, 
        help='Path to the root of the VOC dataset (should contain Annotations/ directory)'
    )
    return parser.parse_args()

def collect_bbox_areas(dataset_dir):
    """
    Parses VOC XML files to extract bounding box areas in pixels.
    Looks inside `dataset_dir/Annotations/` first, and falls back to `dataset_dir/` if not found.
    """
    anno_dir = os.path.join(dataset_dir, 'Annotations')
    if not os.path.isdir(anno_dir):
        print(f"Annotations directory not found at '{anno_dir}'. Searching in root '{dataset_dir}'...")
        anno_dir = dataset_dir
        
    xml_files = glob.glob(os.path.join(anno_dir, '*.xml'))
    if not xml_files:
        # Recursive fallback search
        xml_files = glob.glob(os.path.join(anno_dir, '**', '*.xml'), recursive=True)
        
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in '{anno_dir}' or its subdirectories.")
        
    areas = []
    xml_count = 0
    missing_bndbox_count = 0
    invalid_bndbox_count = 0
    
    print(f"Scanning XML files in '{anno_dir}'...")
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            xml_count += 1
            
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    missing_bndbox_count += 1
                    continue
                    
                xmin_elem = bndbox.find('xmin')
                ymin_elem = bndbox.find('ymin')
                xmax_elem = bndbox.find('xmax')
                ymax_elem = bndbox.find('ymax')
                
                if None in (xmin_elem, ymin_elem, xmax_elem, ymax_elem):
                    missing_bndbox_count += 1
                    continue
                
                # Convert coordinate values to float to support sub-pixel annotations
                xmin = float(xmin_elem.text)
                ymin = float(ymin_elem.text)
                xmax = float(xmax_elem.text)
                ymax = float(ymax_elem.text)
                
                width = xmax - xmin
                height = ymax - ymin
                
                if width <= 0 or height <= 0:
                    invalid_bndbox_count += 1
                    continue
                    
                area = width * height
                areas.append(area)
                
        except Exception as e:
            print(f"Warning: Failed to parse file {xml_path}: {e}")
            
    return areas, xml_count, missing_bndbox_count, invalid_bndbox_count

def print_statistics(areas, xml_count, missing_bndbox_count, invalid_bndbox_count):
    if not areas:
        print("\nNo valid bounding boxes found to analyze.")
        return
        
    areas = np.array(areas)
    
    # Compute requested statistics
    count = len(areas)
    min_val = np.min(areas)
    p5 = np.percentile(areas, 5)
    p10 = np.percentile(areas, 10)
    p25 = np.percentile(areas, 25)
    p50 = np.percentile(areas, 50)  # Median
    mean_val = np.mean(areas)
    p75 = np.percentile(areas, 75)
    p85 = np.percentile(areas, 85)
    p95 = np.percentile(areas, 95)
    p99 = np.percentile(areas, 99)
    max_val = np.max(areas)

    print("=" * 60)
    print(" VOC Dataset Bounding Box Area Explorer (in Pixels) ")
    print("=" * 60)
    print(f"Total XML files scanned:       {xml_count}")
    print(f"Total valid bounding boxes:    {count}")
    if missing_bndbox_count > 0:
        print(f"Objects missing bndbox tags:   {missing_bndbox_count}")
    if invalid_bndbox_count > 0:
        print(f"Bounding boxes with <=0 area:  {invalid_bndbox_count}")
    print("-" * 60)
    print(f"{'Statistic':<28} | {'Value (px^2)':<25}")
    print("-" * 60)
    print(f"{'Count':<28} | {count:<25,}")
    print(f"{'Minimum':<28} | {min_val:<25,.2f}")
    print(f"{'5% Percentile':<28} | {p5:<25,.2f}")
    print(f"{'10% Percentile':<28} | {p10:<25,.2f}")
    print(f"{'25% Percentile':<28} | {p25:<25,.2f}")
    print(f"{'50% Percentile (Median)':<28} | {p50:<25,.2f}")
    print(f"{'Mean':<28} | {mean_val:<25,.2f}")
    print(f"{'75% Percentile':<28} | {p75:<25,.2f}")
    print(f"{'85% Percentile':<28} | {p85:<25,.2f}")
    print(f"{'95% Percentile':<28} | {p95:<25,.2f}")
    print(f"{'99% Percentile':<28} | {p99:<25,.2f}")
    print(f"{'Maximum':<28} | {max_val:<25,.2f}")
    print("=" * 60)
    
    # Calculate equivalent square side lengths to offer a spatial sense of scale
    print(f"Equivalent Square Bounding Box Dimensions (px):")
    print(f"  Min Box Size:    {np.sqrt(min_val):.1f} x {np.sqrt(min_val):.1f} px")
    print(f"  Median Box Size: {np.sqrt(p50):.1f} x {np.sqrt(p50):.1f} px")
    print(f"  Mean Box Size:   {np.sqrt(mean_val):.1f} x {np.sqrt(mean_val):.1f} px")
    print(f"  Max Box Size:    {np.sqrt(max_val):.1f} x {np.sqrt(max_val):.1f} px")
    print("=" * 60)

def main():
    args = parse_args()
    try:
        areas, xml_count, missing, invalid = collect_bbox_areas(args.dataset_dir)
        print_statistics(areas, xml_count, missing, invalid)
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == '__main__':
    main()