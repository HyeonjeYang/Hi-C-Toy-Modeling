import bpy
import csv

# 기존 오브젝트 삭제
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

coords = []
with open("csv_file_path") as f:
    reader = csv.reader(f)
    for row in reader:
        coords.append((float(row[0]), float(row[1]), float(row[2])))

# Curve 생성
curve = bpy.data.curves.new("Polymer", type='CURVE')
curve.dimensions = '3D'
spline = curve.splines.new('POLY')
spline.points.add(len(coords)-1)

for i, (x,y,z) in enumerate(coords):
    spline.points[i].co = (x,y,z,1)

obj = bpy.data.objects.new("Polymer", curve)
bpy.context.collection.objects.link(obj)

# 두께 추가
curve.bevel_depth = 0.05
curve.bevel_resolution = 3
