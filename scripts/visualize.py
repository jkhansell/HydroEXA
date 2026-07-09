# based on traces generated using paraview version 5.8.0
# Modified to handle shallow water data: bathymetry, h, hu, hv, roughness
# Added dynamic AMReX inputs parser to automatically compute domain center and camera zoom

#### import the simple module from the paraview
from paraview.simple import *

import subprocess
import glob
import argparse
import os

ffmpeg_path = "/lustre/orion/geo161/scratch/jkhansell/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--frame_rate', type=int, default=15, help="Frame rate for generating movies.")
parser.add_argument('-r', '--resolution', type=int, default=1024, help="(Square) resolution of output movie.")
parser.add_argument('-d', '--spacedim', type=int, default=2, help="Dimensionality of the problem: 2 or 3")
parser.add_argument('-v', '--variable', type=str, default='h', choices=['bathymetry', 'h', 'hu', 'hv', 'roughness'], help="Variable to use for color mapping.")
parser.add_argument('-i', '--inputs_file', type=str, default='inputs', help="Path to the AMReX inputs file to parse domain geometry.")
args = parser.parse_args()

SWE_VARIABLES = ['bathymetry', 'h', 'hu', 'hv', 'roughness']

def parse_amrex_geometry(inputs_filename):
    """
    Parses AMReX inputs file to extract domain boundaries.
    Defaults to a fallback [-100, 100] window if calculation parameters cannot be found.
    """
    # Healthy defaults based on your physical 200x200 box configuration
    prob_lo = [-100.0, -100.0, 0.0]
    prob_hi = [100.0, 100.0, 1.0]
    
    if os.path.exists(inputs_filename):
        print(f"Parsing AMReX domain parameters from: {inputs_filename}")
        with open(inputs_filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                if 'geometry.prob_lo' in line:
                    parts = line.split('=')[1].split()
                    prob_lo = [float(x) for x in parts[:3]]
                elif 'geometry.prob_hi' in line:
                    parts = line.split('=')[1].split()
                    prob_hi = [float(x) for x in parts[:3]]
    else:
        print(f"Warning: '{inputs_filename}' not detected. Employing default geometry [-100, 100].")

    # Compute bounds metrics
    center_x = (prob_lo[0] + prob_hi[0]) / 2.0
    center_y = (prob_lo[1] + prob_hi[1]) / 2.0
    
    dx = prob_hi[0] - prob_lo[0]
    dy = prob_hi[1] - prob_lo[1]
    max_dim = max(dx, dy)
    
    # In 2D parallel projection views, CameraParallelScale dictates half of the visible vertical height.
    # We add a 10% padding margin factor around the box edge boundary to prevent visual clipping.
    parallel_scale = (max_dim / 2.0) * 1.1 
    
    return [center_x, center_y, 0.0], parallel_scale

def generate_movie_3D(AllPlotFiles):
    paraview.simple._DisableFirstRenderCameraReset()

    plt00 = AMReXBoxLibGridReader(FileNames=AllPlotFiles)
    plt00.CellArrayStatus = SWE_VARIABLES

    animationScene1 = GetAnimationScene()
    animationScene1.UpdateAnimationUsingDataTimeSteps()

    renderView1 = GetActiveViewOrCreate('RenderView')
    renderView1.ViewSize = [args.resolution, args.resolution]

    plt00Display = Show(plt00, renderView1, 'AMRRepresentation')
    plt00Display.Representation = 'Outline'
    plt00Display.ScaleFactor = 0.1
    plt00Display.GlyphType = 'Arrow'
    plt00Display.GaussianRadius = 0.005

    renderView1.ResetCamera()
    renderView1.Update()

    plt00Display.AmbientColor = [0.0, 1.0, 0.0]
    plt00Display.DiffuseColor = [0.0, 1.0, 0.0]

    # Slice Generation
    slice1 = Slice(Input=plt00)
    slice1.SliceType = 'Plane'
    slice1.HyperTreeGridSlicer = 'Plane'
    slice1.SliceOffsetValues = [0.0]
    
    # Read geometry to adjust 3D cut center placement positions smoothly
    domain_center, _ = parse_amrex_geometry(args.inputs_file)
    slice1.SliceType.Origin = [domain_center[0], domain_center[1], 0.5]
    slice1.HyperTreeGridSlicer.Origin = [domain_center[0], domain_center[1], 0.5]

    Hide3DWidgets(proxy=slice1.SliceType)
    slice1.SliceType.Normal = [0.0, 0.0, 1.0]

    slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')
    slice1Display.Representation = 'Surface'
    
    renderView1.Update()

    ColorBy(slice1Display, ('CELLS', args.variable))
    slice1Display.RescaleTransferFunctionToDataRange(True, False)

    varLUT = GetColorTransferFunction(args.variable)
    slice1Display.SetScalarBarVisibility(renderView1, True)

    varLUTColorBar = GetScalarBar(varLUT, renderView1)
    varLUTColorBar.WindowLocation = 'Any Location'
    varLUTColorBar.Position = [0, 0.75]
    varLUTColorBar.ScalarBarLength = 0.2

    output_movie_base = f"amr101_3D_{args.variable}"
    output_png_pattern = f"{output_movie_base}.png"
    
    print(f"Dumping structural PNG stack frames natively: {output_movie_base}.*.png")
    SaveAnimation(output_png_pattern,
                  renderView1,
                  ImageResolution=[args.resolution, args.resolution],
                  FrameRate=args.frame_rate,
                  FrameWindow=[0, len(AllPlotFiles)-1])

    return output_movie_base

def generate_movie_2D(AllPlotFiles):
    paraview.simple._DisableFirstRenderCameraReset()

    plt00 = AMReXBoxLibGridReader(FileNames=AllPlotFiles)
    plt00.CellArrayStatus = SWE_VARIABLES

    animationScene1 = GetAnimationScene()
    animationScene1.UpdateAnimationUsingDataTimeSteps()

    renderView1 = GetActiveViewOrCreate('RenderView')
    renderView1.ViewSize = [args.resolution, args.resolution]

    plt00Display = Show(plt00, renderView1, 'AMRRepresentation')
    plt00Display.Representation = 'Outline'

    # ─── CALCULATE DYNAMIC CAMERA POSITION AND ZOOM ───
    domain_center, computed_scale = parse_amrex_geometry(args.inputs_file)
    
    print(f"Adjusting 2D Camera View Window:")
    print(f" -> Focal Center Location Target: {domain_center[:2]}")
    print(f" -> Calculated Parallel Zoom Scale Factor: {computed_scale}")

    renderView1.InteractionMode = '2D'
    # Position the camera far away on the Z axis pointing directly down at the custom domain center coordinates
    renderView1.CameraPosition = [domain_center[0], domain_center[1], 10000.0]
    renderView1.CameraFocalPoint = [domain_center[0], domain_center[1], 0.0]
    renderView1.CameraParallelScale = computed_scale

    renderView1.Update()
    plt00Display.SetRepresentationType('Surface')

    ColorBy(plt00Display, ('CELLS', args.variable))
    plt00Display.RescaleTransferFunctionToDataRange(True, False)

    varLUT = GetColorTransferFunction(args.variable)
    plt00Display.SetScalarBarVisibility(renderView1, True)

    # Grid line outline overlay reconstruction
    plt00_1 = AMReXBoxLibGridReader(FileNames=AllPlotFiles)
    plt00_1.CellArrayStatus = SWE_VARIABLES
    plt00_1Display = Show(plt00_1, renderView1, 'AMRRepresentation')
    plt00_1Display.Representation = 'Outline'

    renderView1.Update()
    plt00_1Display.AmbientColor = [0.0, 1.0, 0.0]
    plt00_1Display.DiffuseColor = [0.0, 1.0, 0.0]

    varLUTColorBar = GetScalarBar(varLUT, renderView1)
    varLUTColorBar.WindowLocation = 'Any Location'
    varLUTColorBar.Position = [0.05, 0.75]  # Shifted inwards slightly to protect view margins
    varLUTColorBar.ScalarBarLength = 0.2

    output_movie_base = f"amr101_2D_{args.variable}"
    output_png_pattern = f"{output_movie_base}.png"
    
    print(f"Dumping structural PNG stack frames natively: {output_movie_base}.*.png")
    SaveAnimation(output_png_pattern,
                  renderView1,
                  ImageResolution=[args.resolution, args.resolution],
                  FrameRate=args.frame_rate,
                  FrameWindow=[0, len(AllPlotFiles)-1])

    return output_movie_base

def convert_png_stack_to_gif(output_movie_base):
    input_pattern = f"{output_movie_base}.%04d.png"
    generated_frames = glob.glob(f"{output_movie_base}.*.png")
    if not generated_frames:
        print(f"[FATAL] No frame files discovered for pattern: {output_movie_base}.*.png")
        return

    print(f"Compiling {len(generated_frames)} frames into animated GIF using static FFmpeg...")
    
    ffmpeg_cmd = (
        f'{ffmpeg_path} -y -framerate {args.frame_rate} -i "{input_pattern}" '
        f'-vf "scale={args.resolution}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" '
        f'-loop 0 "{output_movie_base}.gif"'
    )
    
    result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Success! Generated file: {output_movie_base}.gif")
    else:
        print("[FFMPEG ERROR DIRECT LOG]")
        print(result.stderr)

if __name__ == "__main__":
    if not (args.spacedim == 2 or args.spacedim == 3):
        print("Please specify --spacedim D (with D=2 or D=3)")
        exit()

    if args.frame_rate <= 0:
        print("Please specify --frame_rate F (with F > 0)")
        exit()

    if args.resolution <= 0:
        print("Please specify --resolution R (with R > 0)")
        exit()

    # Find the plotfiles
    PlotFiles = sorted(glob.glob("plt" + "[0-9]"*5))
    
    if not PlotFiles:
        print("No plotfiles found matching 'plt?????' template.")
        exit()

    output_movie_base = None

    if args.spacedim == 3:
        output_movie_base = generate_movie_3D(PlotFiles)
    elif args.spacedim == 2:
        output_movie_base = generate_movie_2D(PlotFiles)

    if output_movie_base:
        convert_png_stack_to_gif(output_movie_base)