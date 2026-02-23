#npy_animation_using_frames


import bpy
import numpy as np
import os

folder = "path_of_the_folder_that_contains_frames"

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

coords = np.load(os.path.join(folder, "frame_0000.npy"))

curve_data = bpy.data.curves.new("PolymerCurve", type='CURVE')
curve_data.dimensions = '3D'

spline = curve_data.splines.new('POLY')
spline.points.add(len(coords) - 1)

curve_obj = bpy.data.objects.new("Polymer", curve_data)
bpy.context.collection.objects.link(curve_obj)
curve_data.bevel_depth = 0.1

def update_polymer(scene):
    frame = scene.frame_current
    file_path = os.path.join(folder, f"frame_{frame:04d}.npy")
    
    if os.path.exists(file_path):
        coords = np.load(file_path)
        for i, coord in enumerate(coords):
            spline.points[i].co = (coord[0], coord[1], coord[2], 1)

bpy.app.handlers.frame_change_post.clear()

bpy.app.handlers.frame_change_post.append(update_polymer)

bpy.context.scene.frame_start = 0 #you can change the start number
bpy.context.scene.frame_end = 99 #you can change the end number

print("Live animation ready.")
