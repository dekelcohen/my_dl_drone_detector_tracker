"""
Usage:
The script renders an object (.obj file) with textures from --textures <folder> in Blender (install it first) with different poses, --res resolution 
Adjust path to installed blender:
"D:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --python synth_dataset\render_drones.py -- --obj "..\data\drone_3d_models\fpv-drone\source\FPV Drone.obj" --textures "..\data\drone_3d_models\fpv-drone\textures" --out "outputs\rendered_drones" --renders 10 --res 128
"""
import bpy
import math
import random
import os
import sys

# --- ARGUMENT PARSING ---
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse
parser = argparse.ArgumentParser(description="Universal 3D to 2D Synthetic Generator")
parser.add_argument("--obj", type=str, required=True, help="Path to .obj file")
parser.add_argument("--textures", type=str, default="", help="Path to textures folder")
parser.add_argument("--out", type=str, default="data/rendered_objects", help="Output directory")
parser.add_argument("--renders", type=int, default=10, help="Number of images to generate")
parser.add_argument("--res", type=int, default=128, help="Resolution of the output PNG (default 128)")
parser.add_argument("--zoom", type=float, default=0.8, help="Zoom factor. 0.8 leaves a 20% safe transparent border (default 0.8)")
args = parser.parse_args(argv)

os.makedirs(args.out, exist_ok=True)

# --- 1. CLEAR THE SCENE ---
bpy.ops.wm.read_factory_settings(use_empty=True)

# --- 2. SETUP RENDER ENGINE ---
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE' 
scene.render.resolution_x = args.res
scene.render.resolution_y = args.res
scene.render.film_transparent = True       
scene.render.image_settings.color_mode = 'RGBA'
scene.render.image_settings.file_format = 'PNG'

# --- 3. IMPORT THE MODEL ---
print(f"📥 Importing {args.obj}...")
try:
    bpy.ops.wm.obj_import(filepath=args.obj)
except AttributeError:
    bpy.ops.import_scene.obj(filepath=args.obj)

# --- 4. AUTO-CENTER & AUTO-SCALE ---
bpy.ops.object.select_all(action='DESELECT')
meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

# Create a controller empty
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
controller = bpy.context.active_object

# Parent all meshes and calculate bounding box max size
max_dim = 0.001
for obj in meshes:
    obj.parent = controller
    for corner in obj.bound_box:
        for coord in corner:
            if abs(coord) > max_dim:
                max_dim = abs(coord)

# Scale the model. The * args.zoom ensures it has a transparent border and doesn't get clipped.
scale_factor = (2.0 / max_dim) * args.zoom
controller.scale = (scale_factor, scale_factor, scale_factor)
print(f"📐 Auto-scaled model by factor of {scale_factor:.4f} (Zoom: {args.zoom})")

# --- 5. AUTO-FIX BROKEN TEXTURES ---
if args.textures and os.path.exists(args.textures):
    print(f"🔍 Scanning for textures in: {args.textures}")
    
    actual_files = os.listdir(args.textures)
    
    def normalize_name(filename):
        base = os.path.splitext(filename)[0]
        return base.lower().replace(" ", "_")
        
    file_map = {normalize_name(f): os.path.join(args.textures, f) for f in actual_files}
    
    fixed_count = 0
    for img in bpy.data.images:
        if not img.filepath: continue
        requested_name = normalize_name(os.path.basename(img.filepath))
        
        if requested_name in file_map:
            absolute_path = os.path.abspath(file_map[requested_name])
            img.filepath = absolute_path
            img.reload()
            fixed_count += 1
            
    print(f"🎨 Successfully repaired {fixed_count} texture links!")

# --- 6. CAMERA & LIGHTING ---
bpy.ops.object.camera_add(location=(0, -5, 0), rotation=(math.radians(90), 0, 0))
camera = bpy.context.active_object
scene.camera = camera

bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
sun = bpy.context.active_object

# --- 7. RENDER LOOP ---
print(f"🚀 Starting {args.renders} renders at {args.res}x{args.res} resolution...")

for i in range(args.renders):
    # Randomize Object Rotation
    pitch = math.radians(random.uniform(-30, 30)) 
    roll = math.radians(random.uniform(-30, 30))  
    yaw = math.radians(random.uniform(0, 360))    
    controller.rotation_euler = (pitch, roll, yaw)
    
    # Randomize Sun Lighting (Creates realistic shadows across the texture)
    sun.rotation_euler = (math.radians(random.uniform(0, 90)), 0, math.radians(random.uniform(0, 360)))
    sun.data.energy = random.uniform(1.5, 4.0)

    # Save
    filename = f"render_{i:04d}.png"
    filepath = os.path.join(args.out, filename)
    scene.render.filepath = os.path.abspath(filepath)
    
    bpy.ops.render.render(write_still=True)

print("✅ Finished rendering!")