import struct
import zipfile
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import json
import logging
import tempfile
import shutil
import imagej
from scipy.ndimage import binary_dilation, binary_erosion
import math
import os


# remove image size limit that is a checker for decompression bomb DOS attacks
Image.MAX_IMAGE_PIXELS = None


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_roi_data(roi_data):
    """
    Minimal parser for an ImageJ .roi file.
    Returns either:
       {'type': 'polygon', 'points': [...]}      or
       {'type': 'composite', 'paths': [ [...], ... ]}
       {'type': 'points', 'points': [...]}
    """
    roi_type = struct.unpack('>b', roi_data[6:7])[0]
    shape_roi_size = struct.unpack('>i', roi_data[36:40])[0]
    is_composite = shape_roi_size > 0

    # Handle Point ROI (type 10)
    if roi_type == 10:
        left, top, right, bottom = struct.unpack('>hhhh', roi_data[8:16])
        num_points = (len(roi_data) - 64) // 4
        x_coords = []
        y_coords = []
        for i in range(num_points):
            x = struct.unpack('>h', roi_data[64 + 2 * i: 66 + 2 * i])[0] + left
            y = struct.unpack('>h', roi_data[64 + 2 * num_points + 2 * i: 66 + 2 * num_points + 2 * i])[0] + top
            x_coords.append(x)
            y_coords.append(y)
        return {'type': 'points', 'points': list(zip(x_coords, y_coords))}

    # Check for supported ROI types
    if not is_composite and roi_type not in (0, 8):
        raise ValueError("Unsupported ROI type (need polygon, composite, or point).")

    top, left, bottom, right = struct.unpack('>hhhh', roi_data[8:16])

    if is_composite:
        # Composite ROI logic
        shape_array = []
        for i in range(shape_roi_size):
            val = struct.unpack('>f', roi_data[64 + i * 4: 68 + i * 4])[0]
            shape_array.append(val)

        paths = []
        current_path = None
        i = 0
        while i < len(shape_array):
            segment_type = int(shape_array[i])
            if segment_type == 0:  # SEG_MOVETO
                if current_path is not None:
                    paths.append(current_path)
                current_path = []
                x = shape_array[i + 1]
                y = shape_array[i + 2]
                current_path.append((x, y))
                i += 3
            elif segment_type == 1:  # SEG_LINETO
                x = shape_array[i + 1]
                y = shape_array[i + 2]
                current_path.append((x, y))
                i += 3
            elif segment_type == 4:  # SEG_CLOSE
                if current_path is not None:
                    if current_path and current_path[0] != current_path[-1]:
                        current_path.append(current_path[0])
                    paths.append(current_path)
                    current_path = None
                i += 1
            else:
                # Skip unrecognized segments
                i += 1

        if current_path is not None:
            paths.append(current_path)

        return {'type': 'composite', 'paths': paths}
    else:
        # Polygon ROI logic
        num_points = struct.unpack('>h', roi_data[16:18])[0]
        x_coords = [
            struct.unpack('>h', roi_data[64 + i * 2: 66 + i * 2])[0] + left
            for i in range(num_points)
        ]
        y_coords = [
            struct.unpack('>h', roi_data[64 + num_points * 2 + i * 2: 66 + num_points * 2 + i * 2])[0] + top
            for i in range(num_points)
        ]
        polygon_coords = list(zip(x_coords, y_coords))
        return {'type': 'polygon', 'points': polygon_coords}


def get_combined_roi_bounds(all_parsed_rois, dilation_pixels=0):
    """
    Get the bounding box that encompasses all ROIs, with symmetric padding.
    Returns (width, height, offset_x, offset_y) needed to contain all ROIs without clipping.
    """
    max_x = 0
    max_y = 0
    min_x = float('inf')
    min_y = float('inf')

    for parsed_roi in all_parsed_rois:
        if parsed_roi['type'] in ['points', 'polygon']:
            for x, y in parsed_roi['points']:
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                min_y = min(min_y, y)

        elif parsed_roi['type'] == 'composite':
            for path in parsed_roi['paths']:
                for x, y in path:
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)

    # Handle empty case
    if min_x == float('inf'):
        return 1024, 1024, 0, 0

    # Symmetric padding logic
    padding = dilation_pixels + 200
    half_padding = padding // 2

    # Final image size with padding
    width = int(max_x + padding)
    height = int(max_y + padding)

    # Offsets to apply to ROI coordinates
    offset_x = half_padding - min_x
    offset_y = half_padding - min_y

    print(f"    Combined ROI bounds - X: {min_x:.1f} to {max_x:.1f}, Y: {min_y:.1f} to {max_y:.1f}")
    print(f"    Final image dimensions (with symmetric padding): {width} x {height}")
    print(f"    ROI offset: x + {offset_x:.1f}, y + {offset_y:.1f}")

    return width, height, offset_x, offset_y


def offset_all_rois(all_parsed_rois, offset_x, offset_y):
    for roi in all_parsed_rois:
        if roi['type'] in ['points', 'polygon']:
            roi['points'] = [
                (round(x + offset_x), round(y + offset_y)) for x, y in roi['points']
            ]
        elif roi['type'] == 'composite':
            roi['paths'] = [
                [(round(x + offset_x), round(y + offset_y)) for x, y in path] for path in roi['paths']
            ]


def is_point_inside_path(point, path):
    """
    Check if a point is inside a path using ray casting algorithm.
    Used to determine path relationships.
    """
    x, y = point
    inside = False

    # Ensure path is closed
    if path[0] != path[-1]:
        path = path + [path[0]]

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside

    return inside


def find_path_relationships(paths):
    """
    Determine which paths are boundaries and which are holes.
    Returns a list of (path, holes) tuples where:
    - path is the boundary path
    - holes is a list of paths that are holes inside this boundary (may be empty)
    """
    if not paths:
        return []

    n_paths = len(paths)
    # For each path, check if it's inside any other path
    is_inside = {i: set() for i in range(n_paths)}

    for i in range(n_paths):
        # Use a point from path i to test against other paths
        test_point = paths[i][0]  # Use first point of the path

        for j in range(n_paths):
            if i != j and is_point_inside_path(test_point, paths[j]):
                is_inside[i].add(j)

    # Group paths into boundaries and their holes
    result = []
    processed = set()

    for i in range(n_paths):
        if i in processed:
            continue

        # A path is a boundary if it's not inside any other path OR
        # all paths containing it are already processed
        if len(is_inside[i]) == 0 or all(j in processed for j in is_inside[i]):
            # Check for holes for this boundary
            holes = []
            for j in range(n_paths):
                if j != i and j not in processed:
                    # A path is a hole for boundary i if:
                    # 1. It's directly inside path i
                    # 2. It's not inside any other unprocessed path
                    inside_i = i in is_inside[j]
                    inside_others = any(
                        k != i and k not in processed and k in is_inside[j]
                        for k in range(n_paths)
                    )

                    if inside_i and not inside_others:
                        holes.append(paths[j])
                        processed.add(j)

            # Add the boundary and its holes
            result.append((paths[i], holes))
            processed.add(i)

    return result


def draw_roi_on_image(image, parsed_roi):
    """
    Draw ROI on PIL Image with white fill.
    Ensures coordinates are properly handled for PIL's coordinate system.
    """
    draw = ImageDraw.Draw(image)

    if parsed_roi['type'] == 'points':
        # Draw points as small circles
        for x, y in parsed_roi['points']:
            # Ensure coordinates are integers and within bounds
            x, y = int(x), int(y)
            if 0 <= x < image.width and 0 <= y < image.height:
                # Draw a small circle for each point
                radius = 2
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill='white')

    elif parsed_roi['type'] == 'polygon':
        # Draw filled polygon
        points = parsed_roi['points']
        if len(points) >= 3:
            # Convert to integers and ensure they're within bounds
            valid_points = []
            for x, y in points:
                x, y = int(x), int(y)
                # Clamp coordinates to image bounds
                x = max(0, min(x, image.width - 1))
                y = max(0, min(y, image.height - 1))
                valid_points.append((x, y))

            if len(valid_points) >= 3:
                # Convert to flat list for PIL
                flat_points = [coord for point in valid_points for coord in point]
                draw.polygon(flat_points, fill='white')

    elif parsed_roi['type'] == 'composite':
        # Handle composite ROI with boundaries and holes
        roi_structures = find_path_relationships(parsed_roi['paths'])

        for boundary, holes in roi_structures:
            if len(boundary) >= 3:
                # Process boundary coordinates
                valid_boundary = []
                for x, y in boundary:
                    x, y = int(x), int(y)
                    # Clamp coordinates to image bounds
                    x = max(0, min(x, image.width - 1))
                    y = max(0, min(y, image.height - 1))
                    valid_boundary.append((x, y))

                if len(valid_boundary) >= 3:
                    # Draw the boundary filled with white
                    flat_boundary = [coord for point in valid_boundary for coord in point]
                    draw.polygon(flat_boundary, fill='white')

                    # Draw holes filled with black to "cut out" from white
                    for hole in holes:
                        if len(hole) >= 3:
                            valid_hole = []
                            for x, y in hole:
                                x, y = int(x), int(y)
                                # Clamp coordinates to image bounds
                                x = max(0, min(x, image.width - 1))
                                y = max(0, min(y, image.height - 1))
                                valid_hole.append((x, y))

                            if len(valid_hole) >= 3:
                                flat_hole = [coord for point in valid_hole for coord in point]
                                draw.polygon(flat_hole, fill='black')


def run_ij_macro(ij, temp_folder):
    """
    Run ImageJ macro to convert PNG files to ROI files using PyImageJ.

    Args:
        ij: ImageJ instance
        temp_folder: Path to folder containing PNG files
    """
    logger.info(f"Running ImageJ macro on {temp_folder}")

    # Create the ImageJ macro
    macro = f'''dir = "{temp_folder.as_posix()}/";
list = getFileList(dir);
for (i = 0; i < list.length; i++) {{
    if (endsWith(list[i], ".png")) {{
        open(dir + list[i]);
        run("8-bit");
        call("java.lang.System.gc");
        setThreshold(255, 255);
        run("Create Selection");
        nBins = 256;
        hist = newArray(nBins);
        getStatistics(area, mean, min, max, std, hist);
        if (hist[255] == 0) {{ close(); continue; }}
        name = substring(list[i], 0, lengthOf(list[i]) - 4) + ".roi";
        saveAs("Selection", dir + name);
        close();
    }}
}}'''

    try:
        # Run the macro using PyImageJ
        logger.info("Executing ImageJ macro via PyImageJ")
        result = ij.py.run_macro(macro)
        logger.info("ImageJ macro completed successfully")
        return result

    except Exception as e:
        logger.error(f"ImageJ macro failed: {e}")
        raise


def erode_roi_mask(png_path, output_path, erosion_pixels=1):
    """
    Apply morphological erosion to the binary mask represented by the PNG.
    Args:
        png_path: Input PNG file path (the ROI mask image).
        output_path: Output PNG after erosion.
        erosion_pixels: Number of pixels to erode.
    Returns:
        Path to the eroded PNG.
    """
    image = Image.open(png_path).convert('L')
    binary_mask = np.array(image) > 0
    structure = np.ones((2 * erosion_pixels + 1, 2 * erosion_pixels + 1), dtype=bool)
    eroded_mask = binary_erosion(binary_mask, structure=structure)
    eroded_img = Image.fromarray(eroded_mask.astype(np.uint8) * 255)
    eroded_img.save(output_path)
    logger.info(f"Eroded mask saved to {output_path}")
    return output_path


def process_roi_files_flattened(roi_files, output_file, dilation_pixels=0):
    """
    Process multiple ROI files and create a single flattened PNG image.
    Args:
        roi_files: List of ROI file paths or single zip file path
        output_file: Output PNG file path
        image_width: Override image width (optional)
        image_height: Override image height (optional)
        dilation_pixels: Number of pixels for morphological dilation (default=0 - no dilation)
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_parsed_rois = []
    processed_roi_names = []
    if isinstance(roi_files, (str, Path)):
        roi_files = [roi_files]
    for roi_input in roi_files:
        roi_path = Path(roi_input)
        if roi_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(roi_path, 'r') as zf:
                roi_names = [f for f in zf.namelist() if f.endswith('.roi')]
                print(f"    Found {len(roi_names)} ROI files in {roi_path}")
                for roi_name in roi_names:
                    try:
                        with zf.open(roi_name) as roi_file:
                            roi_data = roi_file.read()
                        parsed_roi = parse_roi_data(roi_data)
                        all_parsed_rois.append(parsed_roi)
                        processed_roi_names.append(roi_name)
                        print(f"        Parsed {roi_name} ({parsed_roi['type']})")
                    except Exception as e:
                        print(f"Error processing {roi_name}: {e}")
        elif roi_path.suffix.lower() == '.roi':
            try:
                with open(roi_path, 'rb') as f:
                    roi_data = f.read()
                parsed_roi = parse_roi_data(roi_data)
                all_parsed_rois.append(parsed_roi)
                processed_roi_names.append(str(roi_path))
                print(f"Parsed {roi_path} ({parsed_roi['type']})")
            except Exception as e:
                print(f"Error processing {roi_path}: {e}")
        else:
            print(f"Unsupported file type: {roi_path}")
    if not all_parsed_rois:
        print("No ROI files were successfully parsed!")
        return None
    width, height, offset_x, offset_y = get_combined_roi_bounds(all_parsed_rois, dilation_pixels=dilation_pixels)
    offset_all_rois(all_parsed_rois, offset_x, offset_y)

    image = Image.new('RGB', (width, height), 'black')
    for i, parsed_roi in enumerate(all_parsed_rois):
        draw_roi_on_image(image, parsed_roi)

    # Apply dilation if requested
    if dilation_pixels > 0:
        gray_image = image.convert('L')
        binary_mask = np.array(gray_image) > 0
        structure = np.ones((2 * dilation_pixels + 1, 2 * dilation_pixels + 1), dtype=bool)
        dilated_mask = binary_dilation(binary_mask, structure=structure)
        dilated_img = Image.fromarray(dilated_mask.astype(np.uint8) * 255)
        image = dilated_img.convert('RGB')

    image.save(output_path)
    print(f"\nFlattened image created: {output_path}")

    summary = {
        'total_rois': len(all_parsed_rois),
        'output_file': str(output_path),
        'dimensions': (width, height),
        'roi_files': processed_roi_names,
        'roi_types': [roi['type'] for roi in all_parsed_rois],
        'dilation_pixels': dilation_pixels,
        'offset_x': offset_x,
        'offset_y': offset_y
    }

    summary_file = output_path.parent / (output_path.stem + '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    return summary


def save_roi(parsed_roi, filepath):
    """
    Basic ROI writer for polygon ROI type.
    Supports only polygon for demonstration.
    """
    if parsed_roi['type'] != 'polygon':
        raise NotImplementedError("save_roi: Only polygon ROI writing implemented")

    points = parsed_roi['points']
    n_points = len(points)

    if n_points == 0:
        raise ValueError("save_roi: No points to write")

    left = int(math.floor(min(x for x, y in points)))
    top = int(math.floor(min(y for x, y in points)))
    right = int(math.ceil(max(x for x, y in points)))
    bottom = int(math.ceil(max(y for x, y in points)))

    # Header length is 64 bytes as per ImageJ specification
    # ROI type = 0 (polygon)

    header = bytearray(64)

    # Magic 'Iout'
    header[0:4] = b'Iout'

    # Version, 218 for ImageJ 1.48u (arbitrary but standard)
    struct.pack_into('>H', header, 4, 218)

    # ROI type = 0 (polygon)
    header[6] = 0

    # Top, Left, Bottom, Right - 2 bytes each short ints
    struct.pack_into('>hhhh', header, 8, left, top, bottom, right)

    # Number of coordinates
    struct.pack_into('>H', header, 16, n_points)

    # Write x coordinates relative to left and y to top
    # Coordinates start at byte 64

    x_coords_bytes = bytearray()
    y_coords_bytes = bytearray()

    for x, y in points:
        rel_x = int(round(float(x) - left))
        x_coords_bytes += struct.pack('>h', rel_x)
    for x, y in points:
        rel_y = int(round(float(y) - top))
        y_coords_bytes += struct.pack('>h', rel_y)

    with open(filepath, 'wb') as f:
        f.write(header)
        f.write(x_coords_bytes)
        f.write(y_coords_bytes)


def get_bbox(parsed_roi):
    xs, ys = [], []
    if parsed_roi['type'] in ['points', 'polygon']:
        for x, y in parsed_roi['points']:
            xs.append(x)
            ys.append(y)
    elif parsed_roi['type'] == 'composite':
        for path in parsed_roi['paths']:
            for x, y in path:
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def reverse_offset_on_png(input_png, output_png, offset_x, offset_y):
    """
    Reverse the coordinate offset directly on a binary PNG mask by translating pixels.
    This avoids re-vectorization errors before coordinate reversal.
    """
    img = Image.open(input_png).convert('L')
    arr = np.array(img)

    # Create empty mask in a canvas large enough to hold shifted content
    shifted_arr = np.zeros_like(arr, dtype=np.uint8)

    # Positive offset_x means ROI was shifted right originally -> now move left
    src_y, src_x = np.nonzero(arr > 0)
    dest_x = (src_x - offset_x).astype(int)
    dest_y = (src_y - offset_y).astype(int)

    # Only keep pixels that remain in bounds
    valid_mask = (
        (dest_x >= 0) & (dest_x < arr.shape[1]) &
        (dest_y >= 0) & (dest_y < arr.shape[0])
    )

    shifted_arr[dest_y[valid_mask], dest_x[valid_mask]] = 255

    Image.fromarray(shifted_arr).save(output_png)
    logger.info(f"Reversed offset PNG saved to {output_png}")


def process_and_convert_to_roi_with_erosion(roi_files, output_png=None, output_roi=None, dilation_pixels=0):
    """
    Complete pipeline: ROI files -> flattened PNG (with dilation) -> erode mask -> new ROI file.
    Args:
        roi_files: List of ROI file paths or single zip file path
        output_png: Output PNG file path (optional)
        output_roi: Output ROI file path (optional)
        image_width: Override image width (optional)
        image_height: Override image height (optional)
        dilation_pixels: Number of pixels for dilation and erosion
    Returns:
        Dictionary with paths to created files
    """

    logger.info("Initializing ImageJ...")
    ij = imagej.init()
    try:
        # Step 1: Create flattened PNG with dilation
        dilated_offset_png_path = Path(output_png).with_name(Path(output_png).stem + '_dilated_offset.png')
        print(f"Step 1: Creating flattened PNG from ROI files with dilation of {dilation_pixels} pixels...")
        summary = process_roi_files_flattened(
            roi_files=roi_files,
            output_file=dilated_offset_png_path,
            dilation_pixels=dilation_pixels
        )
        if summary is None:
            raise RuntimeError("Failed to create flattened PNG")

        offset_x = summary['offset_x']
        offset_y = summary['offset_y']

        # Step 2: Erode the dilated PNG to approximate original boundaries
        if dilation_pixels > 0:
            eroded_png_path = Path(output_png).with_name(Path(output_png).stem + '_eroded.png')
            print(f"Step 2: Applying erosion of {dilation_pixels} pixels to undo dilation...")
            eroded_png = erode_roi_mask(
                png_path=dilated_offset_png_path,
                output_path=eroded_png_path,
                erosion_pixels=dilation_pixels
            )
        else:
            eroded_png = output_png

        # Step 3: Reverse offset **on the PNG mask** before vectorization
        print(f"Step 3: Reversing offset ({-offset_x}, {-offset_y}) directly on raster mask...")
        reversed_mask_png = Path(output_png).with_name(Path(output_png).stem + '_rev_offset.png')
        reverse_offset_on_png(eroded_png, reversed_mask_png, offset_x, offset_y)

        # Step 4: Convert the reversed-offset PNG directly to ROI
        print("Step 4: Converting reversed-offset PNG to ROI using ImageJ...")
        final_roi = png_to_roi(
            png_file=reversed_mask_png,
            ij=ij,
            output_roi=output_roi
        )

        result = {
            'input_rois': summary['roi_files'],
            'flattened_png': output_png,
            'eroded_png': str(eroded_png) if dilation_pixels > 0 else None,
            'rev_offset_png': str(reversed_mask_png),
            'final_roi': str(final_roi),
            'summary': summary
        }
        print("\nPipeline complete!")
        return result
    finally:
        if ij is not None:
            logger.info("Cleaning up ImageJ instance...")
            try:
                ij.dispose()
            except Exception:
                pass


def png_to_roi(png_file, ij=None, output_roi=None):
    """
    Convert a PNG file back to ROI format using PyImageJ.

    Args:
        png_file: Path to PNG file
        ij: ImageJ instance (will create new one if None)
        output_roi: Output ROI file path (optional, defaults to same name as PNG)

    Returns:
        Path to created ROI file
    """
    png_path = Path(png_file)

    if not png_path.exists():
        raise FileNotFoundError(f"PNG file not found: {png_path}")

    # Set default output path
    if output_roi is None:
        output_roi = png_path.with_suffix('.roi')
    else:
        output_roi = Path(output_roi)

    # Initialize ImageJ if not provided
    if ij is None:
        logger.info("Initializing ImageJ...")
        ij = imagej.init()

    # Create temporary directory for ImageJ processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_folder = Path(temp_dir)

        # Copy PNG file to temp directory
        temp_png = temp_folder / png_path.name
        shutil.copy2(png_path, temp_png)

        # Run ImageJ macro to convert PNG to ROI
        run_ij_macro(ij, temp_folder)

        # Find the generated ROI file
        roi_files = list(temp_folder.glob("*.roi"))

        if not roi_files:
            raise RuntimeError("No ROI file was generated by ImageJ. The PNG might not contain valid white regions.")

        if len(roi_files) > 1:
            logger.warning(f"Multiple ROI files generated: {roi_files}. Using the first one.")

        generated_roi = roi_files[0]

        # Copy ROI file to output location
        shutil.copy2(generated_roi, output_roi)

        logger.info(f"ROI file created: {output_roi}")
        return output_roi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder_path")
    parser.add_argument("--output_folder_path")
    parser.add_argument("--dilation", type=int, default=0, help="Number of pixels for dilation during flattening (and erosion correction)")
    args = parser.parse_args()

    # create an output directory OR overwrite and recreate
    base_output_dir = Path(args.output_folder_path)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    input_zip_files = sorted(Path(args.input_folder_path).glob("*.zip"))
    for zip_file in input_zip_files:
        input_roizipfile_base_name = os.path.splitext(os.path.basename(zip_file))[0]
        output_dir = base_output_dir / input_roizipfile_base_name
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir()

        try:
            result = process_and_convert_to_roi_with_erosion(
                roi_files=zip_file,
                output_png=output_dir / "flattened_rois.png",
                output_roi=output_dir / "merged.roi",
                dilation_pixels=args.dilation
            )
        except Exception as e:
            print(f"❌ Error processing {zip_file}: {e}")
            continue

        # Zip all .roi outputs for this run
        zip_path = output_dir / f"{input_roizipfile_base_name}_merged.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in output_dir.iterdir():
                if file_path.suffix == ".roi":
                    zipf.write(file_path, arcname=file_path.name)
                    os.remove(file_path)