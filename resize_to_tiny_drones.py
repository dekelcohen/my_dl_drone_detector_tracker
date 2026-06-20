"""
Tiny Drone Simulator for YOLO Training

This script synthetically inserts a high-resolution drone image into a background image,
applying physically accurate optical blur, motion blur, sensor downsampling (INTER_AREA),
sensor noise, and JPEG compression to realistically simulate a tiny distant drone.

Usage Examples:

python resize_to_tiny_drones.py --bg_img sky.jpg --drone_img drone.png --x 300 --y 200 --scale 8 --quality 90 --out_img my_custom_output.jpg

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
    """
    bg_h, bg_w = bg_img.shape[:2]
    hr_h, hr_w = hr_drone.shape[:2]
    lr_h, lr_w = hr_h // scale_factor, hr_w // scale_factor
    
    # To avoid edge artifacts during blur, we need a "padding" around the drone
    pad = hr_w // 2  
    hr_patch_size = hr_w + (pad * 2)
    lr_patch_size = hr_patch_size // scale_factor
    
    # --- Boundary Safety Check ---
    # Ensure the patch doesn't try to crop outside the background image
    half_patch = lr_patch_size // 2
    center_x = int(np.clip(center_x, half_patch, bg_w - half_patch - 1))
    center_y = int(np.clip(center_y, half_patch, bg_h - half_patch - 1))
    
    # Extract the LR patch from the background
    x1 = center_x - half_patch
    y1 = center_y - half_patch
    x2 = x1 + lr_patch_size
    y2 = y1 + lr_patch_size
    
    lr_bg_patch = bg_img[y1:y2, x1:x2].astype(np.float32) / 255.0
    
    # Upscale the background patch to HR to match the drone
    hr_bg_patch = cv2.resize(lr_bg_patch, (hr_patch_size, hr_patch_size), interpolation=cv2.INTER_LINEAR)
    
    # Paste the HR Drone into the center of the HR Background Patch
    dx1 = pad
    dy1 = pad
    dx2 = dx1 + hr_w
    dy2 = dy1 + hr_h
    
    # Alpha blending
    for c in range(3):
        hr_bg_patch[dy1:dy2, dx1:dx2, c] = (
            hr_drone[:, :, c] * hr_drone_alpha +
            hr_bg_patch[dy1:dy2, dx1:dx2, c] * (1.0 - hr_drone_alpha)
        )
        
    composite_hr = hr_bg_patch 
    
    # Apply Optical Physics (Blur before downsampling)
    # Optional: Motion blur (simulate 1D movement)
    if np.random.rand() < 0.3: # 30% chance
        motion_kernel = np.zeros((7, 7))
        motion_kernel[3, :] = 1/7
        angle = np.random.uniform(0, 180)
        M = cv2.getRotationMatrix2D((3, 3), angle, 1)
        motion_kernel = cv2.warpAffine(motion_kernel, M, (7, 7))
        composite_hr = cv2.filter2D(composite_hr, -1, motion_kernel)
        
    # Mandatory: Optical PSF (Lens Blur / Anti-aliasing)
    sigma = np.random.uniform(0.8, 1.5)
    composite_hr = cv2.GaussianBlur(composite_hr, (0, 0), sigma)
    
    # Sensor Integration (Downsample via INTER_AREA)
    composite_lr = cv2.resize(composite_hr, (lr_patch_size, lr_patch_size), interpolation=cv2.INTER_AREA)
    
    # Put the perfectly integrated LR patch back into the full image
    result_img = bg_img.astype(np.float32) / 255.0
    result_img[y1:y2, x1:x2] = composite_lr
    
    
    if GLOBAL_SENSOR_NOISE:
        # Global Sensor Noise (Calibrated for daylight/low-ISO)
        # 10000.0 simulates a high photon count (smooth sky)
        noisy = np.random.poisson(result_img * 10000.0) / 10000.0
        
        # Lower the read noise (0.01 was way too high, 0.001 is much cleaner)
        gauss = np.random.normal(0, 0.001, noisy.shape)
        result_img = np.clip(noisy + gauss, 0, 1.0)
        
    
    final_uint8 = (result_img * 255).astype(np.uint8)    
    
    # Global JPEG artifacts
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, encimg = cv2.imencode('.jpg', final_uint8, encode_param)
    final_image = cv2.imdecode(encimg, 1)
    
    # Return bounding box of the small drone for YOLO label format (x_center, y_center, w, h)
    drone_w_lr, drone_h_lr = lr_w, lr_h
    bbox = (center_x, center_y, drone_w_lr, drone_h_lr)
    
    return final_image, bbox

def main():
    parser = argparse.ArgumentParser(description="Simulate realistic tiny drones for YOLO detection.")
    parser.add_argument("--bg_img", type=str, required=True, help="Path to the background image (e.g. 640x640 sky/landscape).")
    parser.add_argument("--drone_img", type=str, required=True, help="Path to high-res drone image (preferably PNG with transparent background).")
    parser.add_argument("--out_img", type=str, default="output_simulated.jpg", help="Path to save the output image.")
    
    parser.add_argument("--scale", type=int, default=10, help="Downsample factor (e.g. 10 means 50x50 drone becomes 5x5).")
    parser.add_argument("--quality", type=int, default=60, help="JPEG compression quality (0-100).")
    
    # Placement arguments
    parser.add_argument("--x", type=int, default=-1, help="X coordinate for drone center.")
    parser.add_argument("--y", type=int, default=-1, help="Y coordinate for drone center.")
    parser.add_argument("--random_pos", action="store_true", help="Place the drone at a random location (overrides --x and --y).")
    
    args = parser.parse_args()

    # 1. Validate file paths
    if not os.path.exists(args.bg_img):
        raise FileNotFoundError(f"Background image not found: {args.bg_img}")
    if not os.path.exists(args.drone_img):
        raise FileNotFoundError(f"Drone image not found: {args.drone_img}")

    # 2. Load images
    bg_img = cv2.imread(args.bg_img, cv2.IMREAD_COLOR)
    # IMREAD_UNCHANGED ensures we load the 4th Alpha channel if it exists (PNG format)
    drone_img_raw = cv2.imread(args.drone_img, cv2.IMREAD_UNCHANGED) 

    # 3. Extract RGB and Alpha from drone image
    if drone_img_raw.shape[-1] == 4:
        hr_drone = drone_img_raw[:, :, :3].astype(np.float32) / 255.0
        hr_drone_alpha = (drone_img_raw[:, :, 3].astype(np.float32) / 255.0)
    else:
        print("⚠️ Warning: Drone image has no alpha channel. Assuming full opacity.")
        hr_drone = drone_img_raw.astype(np.float32) / 255.0
        hr_drone_alpha = np.ones(hr_drone.shape[:2], dtype=np.float32)

    # 4. Determine Placement
    bg_h, bg_w = bg_img.shape[:2]
    
    if args.random_pos or (args.x == -1 and args.y == -1):
        center_x = np.random.randint(0, bg_w)
        center_y = np.random.randint(0, bg_h)
    else:
        center_x = args.x
        center_y = args.y

    # 5. Run the Simulation
    print(f"⚙️ Simulating tiny drone at ({center_x}, {center_y}) with scale factor {args.scale}x...")
    final_img, bbox = generate_realistic_tiny_drone(
        bg_img=bg_img, 
        hr_drone=hr_drone, 
        hr_drone_alpha=hr_drone_alpha, 
        center_x=center_x, 
        center_y=center_y, 
        scale_factor=args.scale,
        jpeg_quality=args.quality
    )

    # 6. Save output
    cv2.imwrite(args.out_img, final_img)
    print(f"✅ Saved to: {args.out_img}")
    print(f"📍 YOLO Bounding Box (cx, cy, w, h in pixels): {bbox}")
    
    # To format for YOLO txt file (normalized coords):
    yolo_x = bbox[0] / bg_w
    yolo_y = bbox[1] / bg_h
    yolo_w = bbox[2] / bg_w
    yolo_h = bbox[3] / bg_h
    print(f"📝 YOLO Normalized Label: 0 {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}")

if __name__ == "__main__":
    main()