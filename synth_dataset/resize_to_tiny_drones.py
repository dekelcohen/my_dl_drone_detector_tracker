"""
Tiny Drone Simulator for YOLO Training

This script synthetically inserts a high-resolution drone image into a background image,
applying physically accurate optical blur, motion blur, sensor downsampling (INTER_AREA),
sensor noise, and JPEG compression to realistically simulate a tiny distant drone.

Usage Examples:

python synth_dataset\resize_to_tiny_drones.py --bg_img ..\data\drone_crops\bk_tree.png --drone_img outputs\rendered_drones\render_0007.png  --x 200 --y 20 --scale 25 --quality 90 --out_img outputs\resize_drone_bk_tree.png

1. Basic usage with random placement (great for testing):
   python resize_to_tiny_drones.py --bg_img sky.jpg --drone_img drone.png --random_pos

2. Specific placement with custom scale and JPEG quality:
   

Arguments:
  --bg_img     : Path to the background image (e.g., 640x640 sky/landscape).
  --drone_img  : Path to high-res drone image (PNG with transparent background recommended).
  --out_img    : Path to save the output image (default: output_simulated.jpg).
  --scale      : Downsample factor. E.g., 10 means a 50x50 drone becomes 5x5 (default: 10).
  --quality    : JPEG compression quality (0-100) (default: 60).
  --x, --y     : Explicit center pixel coordinates to place the drone.
  --random_pos : Flag to randomly place the drone (overrides --x and --y).
"""

import argparse
import os
import cv2
import numpy as np

GLOBAL_SENSOR_NOISE = False 

def generate_realistic_tiny_drone(bg_img, hr_drone, hr_drone_alpha, center_x, center_y, scale_factor=10, jpeg_quality=60):
    """
    Physically accurate tiny object simulator for YOLO training.
    Returns: final_image, bbox, debug_mask
    """
    bg_h, bg_w = bg_img.shape[:2]
    hr_h, hr_w = hr_drone.shape[:2]
    
    # 1. Clean up invisible dust from 3D renders (Sanitize input)
    hr_drone_alpha[hr_drone_alpha < 0.05] = 0.0
    
    # 2. Premultiply alpha (Fixes dark halos on edges when blurring/resizing)
    hr_drone_pre = hr_drone * hr_drone_alpha[..., np.newaxis]
    hr_rgba = np.concatenate((hr_drone_pre, hr_drone_alpha[..., np.newaxis]), axis=-1)
    
    # 3. Pad the high-res drone so the blur has room to fade out naturally
    pad = max(hr_w, hr_h) // 2
    hr_rgba_padded = cv2.copyMakeBorder(hr_rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0,0,0,0])
    
    # 4. Apply Optical Physics (Blur ONLY the drone)
    if np.random.rand() < 0.3:
        motion_kernel = np.zeros((7, 7))
        motion_kernel[3, :] = 1/7
        angle = np.random.uniform(0, 180)
        M = cv2.getRotationMatrix2D((3, 3), angle, 1)
        motion_kernel = cv2.warpAffine(motion_kernel, M, (7, 7))
        hr_rgba_padded = cv2.filter2D(hr_rgba_padded, -1, motion_kernel)
        
    sigma = np.random.uniform(0.8, 1.5)
    hr_rgba_padded = cv2.GaussianBlur(hr_rgba_padded, (0, 0), sigma)
    
    # 5. Downsample (Sensor simulation)
    new_w = hr_rgba_padded.shape[1] // scale_factor
    new_h = hr_rgba_padded.shape[0] // scale_factor
    lr_rgba = cv2.resize(hr_rgba_padded, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    lr_drone_pre = lr_rgba[..., :3]
    lr_alpha = lr_rgba[..., 3]
    
    # 6. SURGICAL EXTRACTION: Kill the Gaussian "glass box" tail
    # Any pixel with less than 3% opacity is snapped to absolute zero.
    lr_alpha[lr_alpha < 0.03] = 0.0
    
    # Initialize the full-size debug mask
    debug_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    # 7. Find tight bounding box contour around the actual drone pixels
    y_coords, x_coords = np.where(lr_alpha > 0)
    if len(y_coords) == 0:
        return bg_img, (center_x, center_y, 0, 0), debug_mask # Drone is fully transparent/gone

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Crop the drone tightly to its visible pixels
    drone_patch_pre = lr_drone_pre[y_min:y_max+1, x_min:x_max+1]
    drone_patch_alpha = lr_alpha[y_min:y_max+1, x_min:x_max+1, np.newaxis]
    patch_h, patch_w = drone_patch_alpha.shape[:2]
    
    # Calculate placement bounds on background
    x1 = center_x - patch_w // 2
    y1 = center_y - patch_h // 2
    x2 = x1 + patch_w
    y2 = y1 + patch_h
    
    # Boundary clamping (in case the drone is half off-screen)
    bg_x1, bg_x2 = max(0, x1), min(bg_w, x2)
    bg_y1, bg_y2 = max(0, y1), min(bg_h, y2)
    
    if bg_x1 >= bg_x2 or bg_y1 >= bg_y2:
        return bg_img, (center_x, center_y, 0, 0), debug_mask

    patch_x1 = bg_x1 - x1
    patch_x2 = patch_w - (x2 - bg_x2)
    patch_y1 = bg_y1 - y1
    patch_y2 = patch_h - (y2 - bg_y2)

    # Extract the cropped region matching the screen bounds
    d_pre = drone_patch_pre[patch_y1:patch_y2, patch_x1:patch_x2]
    d_alpha = drone_patch_alpha[patch_y1:patch_y2, patch_x1:patch_x2]
    
    final_img = bg_img.copy()
    
    # 8. PIXEL-PERFECT BLEND
    bg_patch = final_img[bg_y1:bg_y2, bg_x1:bg_x2].astype(np.float32)
    
    # Alpha blend math
    blended = d_pre * 255.0 + bg_patch * (1.0 - d_alpha)
    blended_uint8 = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    
    # 9. CONTOUR MASKING
    # Only overwrite the background strictly where the drone's shape exists!
    # Background pixels outside the drone's blob are mathematically untouched.
    mask_3d = np.repeat(d_alpha > 0, 3, axis=2)
    
    bg_patch_uint8 = final_img[bg_y1:bg_y2, bg_x1:bg_x2]
    np.copyto(bg_patch_uint8, blended_uint8, where=mask_3d)

    # 10. POPULATE DEBUG MASK
    # Save the exact boolean footprint to prove there is no rectangle
    debug_mask[bg_y1:bg_y2, bg_x1:bg_x2] = (d_alpha.squeeze(-1) > 0).astype(np.uint8) * 255

    # 11. Global noise and compression
    if GLOBAL_SENSOR_NOISE:
        final_float = final_img.astype(np.float32) / 255.0
        noisy = np.random.poisson(final_float * 10000.0) / 10000.0
        gauss = np.random.normal(0, 0.001, noisy.shape)
        final_img = np.clip(np.round((noisy + gauss) * 255.0), 0, 255).astype(np.uint8)
        
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, encimg = cv2.imencode('.jpg', final_img, encode_param)
    final_output = cv2.imdecode(encimg, 1)
    
    # Return tight bounding box of the inserted object for YOLO
    bbox_w = bg_x2 - bg_x1
    bbox_h = bg_y2 - bg_y1
    bbox_cx = bg_x1 + bbox_w // 2
    bbox_cy = bg_y1 + bbox_h // 2
    bbox = (bbox_cx, bbox_cy, bbox_w, bbox_h)
    
    return final_output, bbox, debug_mask

def main():
    parser = argparse.ArgumentParser(description="Simulate realistic tiny drones for YOLO detection.")
    parser.add_argument("--bg_img", type=str, required=True, help="Path to the background image (e.g. 640x640 sky/landscape).")
    parser.add_argument("--drone_img", type=str, required=True, help="Path to high-res drone image.")
    parser.add_argument("--out_img", type=str, default="output_simulated.jpg", help="Path to save the output image.")
    
    parser.add_argument("--scale", type=int, default=10, help="Downsample factor (e.g. 10 means 50x50 drone becomes 5x5).")
    parser.add_argument("--quality", type=int, default=60, help="JPEG compression quality (0-100).")
    
    # Placement arguments
    parser.add_argument("--x", type=int, default=-1, help="X coordinate for drone center.")
    parser.add_argument("--y", type=int, default=-1, help="Y coordinate for drone center.")
    parser.add_argument("--random_pos", action="store_true", help="Place the drone at a random location (overrides --x and --y).")
    
    args = parser.parse_args()

    if not os.path.exists(args.bg_img):
        raise FileNotFoundError(f"Background image not found: {args.bg_img}")
    if not os.path.exists(args.drone_img):
        raise FileNotFoundError(f"Drone image not found: {args.drone_img}")

    bg_img = cv2.imread(args.bg_img, cv2.IMREAD_COLOR)
    drone_img_raw = cv2.imread(args.drone_img, cv2.IMREAD_UNCHANGED) 

    if drone_img_raw is not None and drone_img_raw.shape[-1] == 4:
        hr_drone = drone_img_raw[:, :, :3].astype(np.float32) / 255.0
        hr_drone_alpha = (drone_img_raw[:, :, 3].astype(np.float32) / 255.0)
    else:
        print("⚠️ Warning: Drone image has no alpha channel. Assuming full opacity.")
        hr_drone = drone_img_raw.astype(np.float32) / 255.0
        hr_drone_alpha = np.ones(hr_drone.shape[:2], dtype=np.float32)

    bg_h, bg_w = bg_img.shape[:2]
    
    if args.random_pos or (args.x == -1 and args.y == -1):
        center_x = np.random.randint(0, bg_w)
        center_y = np.random.randint(0, bg_h)
    else:
        center_x = args.x
        center_y = args.y

    print(f"⚙️ Simulating tiny drone at ({center_x}, {center_y}) with scale factor {args.scale}x...")
    final_img, bbox, debug_mask = generate_realistic_tiny_drone(
        bg_img=bg_img, 
        hr_drone=hr_drone, 
        hr_drone_alpha=hr_drone_alpha, 
        center_x=center_x, 
        center_y=center_y, 
        scale_factor=args.scale,
        jpeg_quality=args.quality
    )

    # Save output image
    cv2.imwrite(args.out_img, final_img)
    print(f"✅ Saved output to: {args.out_img}")
    
    # Save the debug mask image
    base, ext = os.path.splitext(args.out_img)
    mask_out_path = f"{base}_mask.png" # Force PNG so compression doesn't blur the mask edges
    cv2.imwrite(mask_out_path, debug_mask)
    print(f"✅ Saved debug mask to: {mask_out_path}")
    
    yolo_x = bbox[0] / bg_w
    yolo_y = bbox[1] / bg_h
    yolo_w = bbox[2] / bg_w
    yolo_h = bbox[3] / bg_h
    print(f"📝 YOLO Normalized Label: 0 {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}")

if __name__ == "__main__":
    main()