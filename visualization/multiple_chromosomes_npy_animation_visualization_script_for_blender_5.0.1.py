import bpy
import numpy as np
import colorsys

# =============================
# LOAD DATA
# =============================
data = np.load(r"C:/Users/user/Downloads/spherical_multiple_chrom_dynamics_20260223.npy")

FRAME_STRIDE = 1
MONOMER_STRIDE = 1

data = data[::FRAME_STRIDE]
frames, chains, monomers, _ = data.shape

print("Data shape:", data.shape)

# ============================= 
# CLEAN SCENE
# =============================
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.app.handlers.frame_change_post.clear()

chain_objects = []

# =============================
# CREATE CURVES + OPTIONS
# =============================
for c in range(chains):

    curve_data = bpy.data.curves.new(f"Chain_{c}", type='CURVE')
    curve_data.dimensions = '3D'

    curve_data.fill_mode = 'FULL'
    curve_data.use_radius = True

    # Geometry
    curve_data.extrude = 0.10  #you can adjust the numbers
    curve_data.offset = 0.0

    # Bevel
    curve_data.bevel_mode = 'ROUND'
    curve_data.bevel_depth = 0.13
    curve_data.bevel_resolution = 3
    curve_data.use_fill_caps = False

    # -----------------------------

    obj = bpy.data.objects.new(f"Chain_{c}", curve_data)
    bpy.context.collection.objects.link(obj)

    # -------- Material --------
    mat = bpy.data.materials.new(name=f"Mat_{c}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]

    hue = c / chains
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.65)
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.5

    obj.data.materials.append(mat)
    chain_objects.append(obj)

print("Created chains:", len(chain_objects))


# =============================
# FRAME HANDLER
# =============================
def update_scene(scene):
    frame = scene.frame_current
    if frame >= frames:
        return

    for idx, obj in enumerate(chain_objects):

        coords = data[frame, idx][::MONOMER_STRIDE]

        curve_data = obj.data
        curve_data.splines.clear()

        spline = curve_data.splines.new('POLY')
        spline.points.add(len(coords) - 1)

        for i, point in enumerate(coords):
            spline.points[i].co = (
                float(point[0]),
                float(point[1]),
                float(point[2]),
                1.0
            )

    bpy.context.view_layer.update()

bpy.app.handlers.frame_change_post.append(update_scene)

bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = frames - 1

print("Animation ready with UI-style geometry settings.")
