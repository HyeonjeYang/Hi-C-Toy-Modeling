#for blender (not .csv, you have to use .npy)
#you can get .npy from
#polychrom_simu_get_h5.ipynb --> get_coordinates_for_blender.ipynb

#copy the code below and paste it to Blender --> Scripting --> New

import bpy
import numpy as np

file_path = "your_npy_file_path"

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

coords = np.load(file_path)

# coords = coords * 0.2

curve_data = bpy.data.curves.new(name="PolymerCurve", type='CURVE')
curve_data.dimensions = '3D'
curve_data.resolution_u = 2

spline = curve_data.splines.new('POLY')
spline.points.add(len(coords) - 1)

for i, coord in enumerate(coords):
    x, y, z = coord
    spline.points[i].co = (x, y, z, 1)  # w=1

curve_data.bevel_depth = 0.15
curve_data.bevel_resolution = 3

curve_obj = bpy.data.objects.new("Polymer", curve_data)
bpy.context.collection.objects.link(curve_obj)

print("Polymer curve created.")
