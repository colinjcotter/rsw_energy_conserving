# render_fast.py
from paraview.simple import *
import os
import subprocess

fields_to_render = ['vorticity', 'psi_noise', 'velocity']

for f in fields_to_render:
    os.makedirs(f"frames/{f}", exist_ok=True)

reader = OpenDataFile("w6_rsw_output_SFLT.pvd")
reader.UpdatePipeline()

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1920, 1080]
view.Background = [0.2, 0.2, 0.3]
view.OrientationAxesVisibility = 0  # hide XYZ axes

display = Show(reader, view)
timesteps = reader.TimestepValues

for field in fields_to_render:
    print(f"Rendering {field}...")
    
    # Hide all color bars first
    HideScalarBarIfNotNeeded(GetColorTransferFunction(field), view)
    
    if field == 'velocity':
        ColorBy(display, ('POINTS', field, 'Magnitude'))
    else:
        ColorBy(display, ('POINTS', field))
    
    # Reset to correct data range
    display.RescaleTransferFunctionToDataRange(True, True)
    display.SetScalarBarVisibility(view, True)
    
    # Hide all OTHER scalar bars
    for other in fields_to_render:
        if other != field:
            try:
                HideScalarBarIfNotNeeded(GetColorTransferFunction(other), view)
            except:
                pass
    
    Render()

    count = 0
    for i, t in enumerate(timesteps):
        if i % 1 == 0:
            GetAnimationScene().AnimationTime = t
            Render()
            SaveScreenshot(f"frames/{field}/{field}_{count:04d}.png", view)
            count += 1

    print(f"  Done: {count} frames")

# Convert to videos
print("Converting to videos...")
for field in fields_to_render:
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '1',
        '-i', f'frames/{field}/{field}_%04d.png',
        '-c:v', 'libx264', '-crf', '20',
        f'{field}.mp4'
    ])
    print(f"  Created {field}.mp4")

print("All done!")





















# # check_fields.py — run with: pvbatch check_fields.py
# from paraview.simple import *

# reader = OpenDataFile("w6_rsw_output_SFLT.pvd")
# reader.UpdatePipeline()

# print("=== POINTS fields ===")
# for i in range(reader.GetPointDataInformation().GetNumberOfArrays()):
#     print(reader.GetPointDataInformation().GetArray(i).GetName())

# print("=== CELLS fields ===")
# for i in range(reader.GetCellDataInformation().GetNumberOfArrays()):
#     print(reader.GetCellDataInformation().GetArray(i).GetName())



# # render_all.py — run with: pvbatch render_all.py
# from paraview.simple import *
# import os

# # Create output directories
# fields = ['velocity', 'F', 'D', 'eta', 'psi_noise', 'psi_perp', 'u_noise', 'vorticity']
# for f in fields:
#     os.makedirs(f"frames/{f}", exist_ok=True)

# # Load data
# reader = OpenDataFile("w6_rsw_output_SFLT.pvd")
# reader.UpdatePipeline()

# view = GetActiveViewOrCreate('RenderView')
# view.ViewSize = [1920, 1080]
# view.Background = [0.2, 0.2, 0.3]  # dark background like your ParaView screenshots

# display = Show(reader, view)

# timesteps = reader.TimestepValues
# print(f"Found {len(timesteps)} timesteps")

# # Scalar fields (ColorBy scalar)
# scalar_fields = {
#     'vorticity': 'POINTS',
#     'eta': 'POINTS',
#     'D': 'POINTS',
#     'F': 'POINTS',
#     'psi_noise': 'POINTS',
#     'psi_perp': 'POINTS',
# }

# # Vector fields (ColorBy Magnitude)
# vector_fields = {
#     'velocity': 'POINTS',
#     'u_noise': 'POINTS',
# }

# # Render scalar fields
# for field, assoc in scalar_fields.items():
#     print(f"Rendering {field}...")
#     ColorBy(display, (assoc, field))
#     display.RescaleTransferFunctionToDataRange(True, False)
#     display.SetScalarBarVisibility(view, True)
#     Render()

#     for i, t in enumerate(timesteps):
#         GetAnimationScene().AnimationTime = t
#         Render()
#         SaveScreenshot(f"frames/{field}/{field}_{i:04d}.png", view)
    
#     print(f"  Done: {len(timesteps)} frames")

# # Render vector fields (magnitude)
# for field, assoc in vector_fields.items():
#     print(f"Rendering {field} (magnitude)...")
#     ColorBy(display, (assoc, field, 'Magnitude'))
#     display.RescaleTransferFunctionToDataRange(True, False)
#     display.SetScalarBarVisibility(view, True)
#     Render()

#     for i, t in enumerate(timesteps):
#         GetAnimationScene().AnimationTime = t
#         Render()
#         SaveScreenshot(f"frames/{field}/{field}_{i:04d}.png", view)
    
#     print(f"  Done: {len(timesteps)} frames")

# print("All frames rendered. Now convert to videos with:")
# print("  for f in velocity F D eta psi_noise psi_perp u_noise vorticity; do")
# print("    ffmpeg -framerate 10 -i frames/$f/${f}_%04d.png -c:v libx264 -crf 20 ${f}.mp4")
# print("  done")