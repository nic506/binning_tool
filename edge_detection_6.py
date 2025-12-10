import numpy
from roifile import ImagejRoi
import argparse
from shapely.geometry import Polygon, LinearRing, LineString, mapping
import json
import os
import zipfile
import tempfile
import glob
import shutil
from contextlib import redirect_stdout
from scipy.spatial import KDTree
import pandas


# reads a .roi file, extracts its anchor point coordinates, and converts to a Shapely polygon to
# enable geometry operations
### NOTE THAT COMPOSITE ROIS ARE REDUCED, ONLY RETAINING THE LARGEST POLYGON (I think by area) ###
def roi_to_poly(path):
    roi = ImagejRoi.fromfile(path)
    xs, ys = roi.coordinates().T
    return Polygon(zip(xs, ys))

# forces all polygon anchor points to be listed clockwise, as the ROI may have been drawn clockwise
# or counter-clockwise in Fiji, it also removes the duplicate closing point
def clockwise_coords(poly):
    pts = list(poly.exterior.coords)[:-1]
    if LinearRing(pts).is_ccw:
        pts.reverse()
    return pts

# computes internal angle between 2 inputted vectors
def internal_angle_between_vectors(v1, v2):
    # dot product - calculates angle
    cosine_angle = numpy.dot(v1, v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))
    angle = numpy.arccos(numpy.clip(cosine_angle, -1.0, 1.0))

    # cross product - cross negative if >180 degrees, positive if <180
    cross = numpy.cross(v1, v2)[2]
    if cross < 0:
        angle = 2 * numpy.pi - angle

    return numpy.degrees(angle)

# walks a fixed distance along a polygons perimeter clockwise or counter-clockwise from an inputted
# anchor point, and returns the exact coordinate at that distance using interpolation
def walk_along_polygon(poly_coords, index, distance, direction):
    n_polygon_coords = len(poly_coords)
    accumulated_distance = 0
    steps = 1 if direction == 'cw' else -1
    i = index
    start = numpy.array(poly_coords[i])

    # infinite loop that stops when the 'if' statement is satisfied
    while True:
        j = (i + steps) % n_polygon_coords
        end = numpy.array(poly_coords[j])
        segment_vec = end - start
        segment_len = numpy.linalg.norm(segment_vec)

        # distance lies within edge segment between current start and end points
        if accumulated_distance + segment_len >= distance:
            remaining_len = distance - accumulated_distance
            # linear interpolation formula - finds the exact point along the polygons edge where the
            # accumulated_distance is equal to the inputted distance
            interp_point = start + (segment_vec / segment_len) * remaining_len
            return interp_point

        accumulated_distance += segment_len
        start = end
        i = j

        if i == index:
            raise ValueError(f"The specified multiscale distances exceeds the perimeter of the polygon.")

# compute the shortest distance along the polygon perimeter between 2 inputted anchor points (tries
# both clockwise and counter-clockwise and returns the shorter)
def shortest_polygonal_distance(poly_coords, idx1, idx2):
    n_polygon_coords = len(poly_coords)

    distance_idx1_to_idx2 = 0
    i = idx1
    while i != idx2:
        distance_idx1_to_idx2 += numpy.linalg.norm(
            numpy.array(poly_coords[i]) - numpy.array(poly_coords[(i + 1) % n_polygon_coords])
        )
        i = (i + 1) % n_polygon_coords

    distance_idx2_to_idx1 = 0
    i = idx2
    while i != idx1:
        distance_idx2_to_idx1 += numpy.linalg.norm(
            numpy.array(poly_coords[i]) - numpy.array(poly_coords[(i + 1) % n_polygon_coords])
        )
        i = (i + 1) % n_polygon_coords

    return min(distance_idx1_to_idx2, distance_idx2_to_idx1)

# detects 4 rectangle-like vertices in the polygon
def detect_rectangle_vertices(
        polygon_coords,
        multiscale_distances,
        multiscale_angle_thresh,
        proximity_thresh
):

# Step 1: Multiscale (at multiple distances) angle filtering for internal angle candidates (uses
# walk_along_polygon and internal_angle_between_vectors self-defined functions)
    # for each internal angle candidate, records: (1) how many multiscale angles pass the
    # multiscale_angle_thresh, (2) the mean angle, (3) the standard deviation of the multiscale
    # angles (ie: how consistent the corner-ness is)
    multiscale_angle_candidate_list = []
    for i in range(len(polygon_coords)):
        multiscale_angles = []
        angle_thresh_check = []
        for dist in multiscale_distances:
            point_ccw = walk_along_polygon(polygon_coords, i, dist, 'ccw')
            point_cw = walk_along_polygon(polygon_coords, i, dist, 'cw')
            vector_from_point_ccw = numpy.append(numpy.array(point_ccw) - polygon_coords[i], 0)
            vector_to_point_cw = numpy.append(numpy.array(point_cw) - polygon_coords[i], 0)
            angle = internal_angle_between_vectors(vector_from_point_ccw, vector_to_point_cw)
            angle_thresh_check.append(int(multiscale_angle_thresh[0] <= angle <=
                                          multiscale_angle_thresh[1]))
            multiscale_angles.append(angle)
        score = sum(angle_thresh_check)
        multiscale_angles_mean = numpy.mean(multiscale_angles)
        multiscale_angles_sd = numpy.std(multiscale_angles)
        multiscale_angle_candidate_list.append({
            'index': i,
            'coord': polygon_coords[i],
            'score': score,
            'mean_angle': multiscale_angles_mean,
            'angle_consistency': multiscale_angles_sd
        })

# Step 2: Rank mutliscale angle candidates according to their score (priority) and angle
# consistency, this ensures the order which candidates will be processed in Step 4
    multiscale_angle_candidate_list.sort(key=lambda x: (-x['score'], x['angle_consistency']))

# Step 3: Cluster multiscale angle candidates that are close to each other along the polygon
# perimeter (within proximity_thresh) (uses shortest_polygonal_distance self-defined function), then
# keep only the cluster member with the highest score (priority) and best angle consistency (lowest
# sd means more consistent)
    ranked_proxfilter_candidate_list = []
    taken = set()
    for i in range(len(multiscale_angle_candidate_list)):
        a = multiscale_angle_candidate_list[i]
        if a['index'] in taken:
            continue
        cluster = [a]
        taken.add(a['index'])
        for j in range(i + 1, len(multiscale_angle_candidate_list)):
            b = multiscale_angle_candidate_list[j]
            a_to_b_distance = shortest_polygonal_distance(polygon_coords, a['index'], b['index'])
            if a_to_b_distance <= proximity_thresh:
                cluster.append(b)
                taken.add(b['index'])
        cluster.sort(key=lambda x: (-x['score'], x['angle_consistency']))
        ranked_proxfilter_candidate_list.append(cluster[0])

# Step 4: Select the top four candidates, which are ranked (done in Step 3) according to their score
# (priority) and angle consistency - this is after having accounted for candidates that are too
# proximal to eachother
    chosen_4_vertices_list = ranked_proxfilter_candidate_list[:4]


# Step 5: orders the 4 chosen vertices according to index (this is clockwise) and output their index
    chosen_4_vertices_list.sort(key=lambda x: (x['index']))
    chosen_4_vertices_indices = [i['index'] for i in chosen_4_vertices_list]

    return ranked_proxfilter_candidate_list, chosen_4_vertices_indices


# # finds the indices of the closest anchor points (ie: the vertices) on the polygon to each manual click
# def find_indices_closest_anchor(csv_path, polygon_coords):
#
#     # reads .csv file containing the manually defined coordinates of the polygon vertices
#     manual_click_coords = pandas.read_csv(csv_path, header=None)
#
#     vertices_indices_on_poly = []
#     for _, row in manual_click_coords.iterrows():
#         click = (row[0], row[1])
#         min_dist = float("inf")
#         for i, anchor in enumerate(polygon_coords):
#             dist = numpy.linalg.norm(numpy.array(anchor) - numpy.array(click))
#             if dist < min_dist:
#                 min_dist = dist
#                 idx = i
#         vertices_indices_on_poly.append(idx)
#
#     if len(vertices_indices_on_poly) != 4:
#         print(f"4 closest anchor points not found for {csv_path}")
#
#     # orders the 4 chosen indices in ascending order
#     sorted_vertices_indices_on_poly = sorted(vertices_indices_on_poly)
#
#     return sorted_vertices_indices_on_poly


# finds the indices of the closest anchor points (ie: the vertices) on the polygon to each manual click
def find_indices_closest_anchor(csv_path, polygon_coords):

    # reads .csv file containing the manually defined coordinates of the polygon vertices
    manual_click_coords = pandas.read_csv(csv_path, header=None).to_numpy()

    tree = KDTree(numpy.array(polygon_coords))
    _, vertices_indices_on_poly = tree.query(manual_click_coords, k=1)

    if len(vertices_indices_on_poly) != 4:
        print(f"4 closest anchor points not found for {csv_path}")

    # orders the 4 chosen indices in ascending order
    sorted_vertices_indices_on_poly = sorted(vertices_indices_on_poly)

    return sorted_vertices_indices_on_poly


# identifies the clockwise adjacent vertex for a list of inputted vertice indices
def find_clockwise_adjacent_vertices(vertice_indices):
    n_vertices = len(vertice_indices)
    clockwise_adjacent_vertice_indices = {}

    for i in range(n_vertices):
        current_vertex = vertice_indices[i]
        adjacent_cw_vertex = vertice_indices[(i + 1) % n_vertices]
        clockwise_adjacent_vertice_indices[current_vertex] = adjacent_cw_vertex

    return clockwise_adjacent_vertice_indices

# calculate clockwise path length along polygon perimeter between two inputted points
def polygonal_path_length_cw(poly_coords, start_idx, end_idx):
    n_polygon_coords = len(poly_coords)
    path_length = 0
    idx = start_idx

    while idx != end_idx:
        next_idx = (idx + 1) % n_polygon_coords
        current_point_coords = numpy.array(poly_coords[idx])
        next_point_coords = numpy.array(poly_coords[next_idx])
        path_length += numpy.linalg.norm(next_point_coords - current_point_coords)
        idx = next_idx

    return path_length

# computes perimeter distances between each of the 4 vertices and their clockwise adjacent vertex
def assign_corners_and_edges(chosen_4_vertices_indices, poly_coords):

    # uses find_clockwise_adjacent_vertices self-defined function
    chosen_4_vertices_adjacent_indices = find_clockwise_adjacent_vertices(chosen_4_vertices_indices)

    # uses polygonal_path_length_cw self-defined function
    clockwise_edge_lengths = []
    for i in chosen_4_vertices_indices:
        adjacent_index = chosen_4_vertices_adjacent_indices[i]
        path_length = polygonal_path_length_cw(poly_coords, i, adjacent_index)
        clockwise_edge_lengths.append({'start': i, 'end': adjacent_index,'length': path_length})

    return clockwise_edge_lengths


# returns coordinates of anchor points between two given vertices along a polygon ordered in the
# clockwise direction
def get_anchor_points_between(poly_coords, start_idx, end_idx):
    n_polygon_coords = len(poly_coords)
    coords_between = []
    idx = start_idx
    coords_between.append(poly_coords[idx])

    while True:
        idx = (idx + 1) % n_polygon_coords
        coords_between.append(poly_coords[idx])
        if idx == end_idx:
            break

    return coords_between

# returns LineStrings, that are oriented in the same direction, of the longest edge in the polygon
# and the edge opposite that one
def select_opposite_long_edges(edge_lengths, poly_coords):

    # sort and pick the longest edge
    edge_lengths.sort(key=lambda x: (-x['length']))
    longest_edge_start, longest_edge_end = edge_lengths[0]['start'], edge_lengths[0]['end']

    # pick the edge opposite the longest edge
    for i in edge_lengths:
        if (i['start'] not in (longest_edge_start, longest_edge_end) and i['end'] not in
                (longest_edge_start, longest_edge_end)):
            opposite_edge_start, opposite_edge_end = i['start'], i['end']
            break

    # uses get_anchor_points_between self-defined function
    longest_edge_coords = get_anchor_points_between(poly_coords, longest_edge_start,
                                                    longest_edge_end)
    opposite_edge_coords = get_anchor_points_between(poly_coords, opposite_edge_start,
                                                     opposite_edge_end)

    # build the 2 edge LineStrings, reverse the order of one of the edge coord lists to get
    # counter-clockwise orientation so that both edges point in the same direction
    longest_edge_linestring  = LineString(longest_edge_coords)
    opposite_edge_linestring = LineString(reversed(opposite_edge_coords))

    return longest_edge_linestring, opposite_edge_linestring


# takes 2 LineStrings and saves them into a GeoJSON file
def save_linestrings_to_geojson(linestring1, linestring2, filename):
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"line_id": "longest_edge"},
                "geometry": mapping(linestring1)
            },
            {
                "type": "Feature",
                "properties": {"line_id": "opposite_edge"},
                "geometry": mapping(linestring2)
            }
        ]
    }
    with open(filename, "w") as f:
        json.dump(geojson_data, f, indent=2)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", help="File path to the merging output")
    parser.add_argument("--animal", choices=["mouse", "marmoset", "human"], help="Type name of animal - mouse or marmoset or human")
    args = parser.parse_args()

    ### NOTE THAT THE INPUT ROIS HAVE TO BE OUTPUT FROM ROI MERGING OR AUTOMATED DELINEATOR (because
    # manually drawn anchor points may not be exact coordinates, ie: 2.567...) ###

    # loop over each subfolder
    for subfolder in os.listdir(args.directory):
        subfolder_path = os.path.join(args.directory, subfolder)

        # look for .zip file inside subfolder
        zip_path = next(glob.iglob(os.path.join(subfolder_path, "*.zip")), None)
        if not zip_path:
            print(f"No zip file found in {subfolder_path}")
            continue

        # create an output directory OR overwrite and recreate
        roizipfile_base_name = os.path.splitext(os.path.basename(zip_path))[0]
        output_dir = os.path.join(subfolder_path, f"edge_detect_{roizipfile_base_name}")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # sets arguments for detect_rectangle_vertices function depending on the animal
        if args.animal == "mouse":
            args_detect_rectangle_vertices = {
                'multiscale_distances': list(range(0, 0, 0)),
                'multiscale_angle_thresh': (0, 0),
                'proximity_thresh': 0
            }
        elif args.animal == "marmoset":
            args_detect_rectangle_vertices = {
                'multiscale_distances': list(range(5, 900, 5)),
                'multiscale_angle_thresh': (60, 120),
                'proximity_thresh': 150
            }
        else:
            args_detect_rectangle_vertices = {
                'multiscale_distances': list(range(10, 1501, 10)),
                'multiscale_angle_thresh': (30, 150),
                'proximity_thresh': 400
            }

        # create a temporary directory to extract individual ROI files
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            roi_files = glob.glob(os.path.join(tmpdir, '*.roi'))

            # redirect all print() output to a text file
            output_textfile = os.path.join(output_dir, "edge_detection_output_log.txt")
            with open(output_textfile, "w") as f:
                with redirect_stdout(f):

                    for i in roi_files:

                        ### SEE NOTE WHERE FUNCTION IS DEFINED ###
                        poly = roi_to_poly(i)

                        ### NOTE THIS IS ESSENTIAL FOR DOWNSTREAM PROCESSING ###
                        poly_simplified = poly.simplify(tolerance=3.0, preserve_topology=True)

                        polygon_coords = clockwise_coords(poly_simplified)

                        # proceed with automated vertex detection if no manual_vertex_coords folder
                        if not glob.glob(os.path.join(subfolder_path, "manual_vertex_coords")):
                            ranked_proxfilter_candidate_list, chosen_4_vertices_indices = detect_rectangle_vertices(polygon_coords, **args_detect_rectangle_vertices)

                        # proceed with manual vertex detection if manual_vertex_coords folder present
                        else:
                            csv_path = os.path.join(subfolder_path, "manual_vertex_coords", f"{os.path.splitext(os.path.basename(i))[0]}.csv")
                            chosen_4_vertices_indices = find_indices_closest_anchor(csv_path, polygon_coords)

                        if len(chosen_4_vertices_indices) != 4:
                            print(f"ROI {i}, expected 4 final vertex candidates, but found {len(chosen_4_vertices_indices)}.")
                            continue

                        clockwise_edge_lengths = assign_corners_and_edges(chosen_4_vertices_indices, polygon_coords)

                        longest_edge, opposite_edge = select_opposite_long_edges(clockwise_edge_lengths, polygon_coords)

                        roifile_base_name = os.path.splitext(os.path.basename(i))[0]
                        geojson_filename = os.path.join(output_dir, f"{roifile_base_name}.geojson")
                        save_linestrings_to_geojson(longest_edge, opposite_edge, geojson_filename)


                        # edge detection output log information
                        decoration = '~' * ((80 - len(roifile_base_name) - 2) // 2)
                        print('\n' + decoration + ' ' + roifile_base_name + ' ' + decoration)

                        if not glob.glob(os.path.join(subfolder_path, "manual_vertex_coords")):
                            print("\nVertex candidates:")
                            for candidate in ranked_proxfilter_candidate_list:
                                coord = candidate['coord']
                                score = candidate['score']
                                mean_angle = candidate['mean_angle']
                                angle_consistency = candidate['angle_consistency']
                                print(f"  Coord: {coord}, Score: {score}, Mean angle: {mean_angle:.2f}°, Angle consistency (std dev): {angle_consistency:.2f}°")

                        print("\nLongest edge:")
                        print(f"  Start coords: {longest_edge.coords[0]}")
                        print(f"  End coords:   {longest_edge.coords[-1]}")
                        print(f"  Length (px):  {round(longest_edge.length, 2)}")

                        print("\nOpposite edge:")
                        print(f"  Start coords: {opposite_edge.coords[0]}")
                        print(f"  End coords:   {opposite_edge.coords[-1]}")
                        print(f"  Length (px):  {round(opposite_edge.length, 2)}")

                        print('\n' + '~' * 80)

    print(f"\n  COMPLETED SUCCESSFULLY\n")