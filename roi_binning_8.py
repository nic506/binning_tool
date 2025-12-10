import os
import shutil
import argparse
import numpy
from shapely.geometry import shape, Polygon, LineString, Point
from shapely.ops import substring, unary_union
from shapely.affinity import scale
from roifile import ImagejRoi
import json
from scipy.spatial import Voronoi
import networkx as nx
import tempfile
import math
from shapelysmooth import chaikin_smooth, taubin_smooth
from itertools import chain
import glob
import zipfile
import pandas


# reads a .geojson file
def load_linestrings_from_geojson(filename):
    with open(filename) as f:
        data = json.load(f)

    lines = {}
    for feature in data['features']:
        line_id = feature['properties']['line_id']
        geom = shape(feature['geometry'])
        if not isinstance(geom, LineString):
            raise ValueError(f"Geometry for {line_id} is not a LineString")
        lines[line_id] = geom

    return lines['longest_edge'], lines['opposite_edge']


# splits a LineString into equal segments by dividing the Linestring length by n and interpolates
# sample point coordinates (outputs a list of shapely Point objects)
def fixednumber_segment_linestring(linestring, number_segments):
    trunc_number_segments = int(number_segments)
    segment_size = numpy.linspace(0, linestring.length, trunc_number_segments + 1)
    segment_points = [linestring.interpolate(i) for i in segment_size]
    return segment_points


# scale units in pixels to units in micrometers of shapely geometries
def scale_px_to_um(shapely_geometry):

    # calculate the resolution of montage images by dividing the original pixel size (0.068 um/px)
    # by the x- and y-downsampling factor used in the montage stitching process (note they are
    # different for x and y)
    x_resolution_um_per_px = 0.068 / (60/942)  # 60 is downsampled tile width, 942 is original tile width
    y_resolution_um_per_px = 0.068 / (59/920)  # 59 is downsampled tile height, 920 is original tile height

    # scale the input geometry accordingly
    scaled_geometry = scale(shapely_geometry, xfact=x_resolution_um_per_px, yfact=y_resolution_um_per_px, origin=(0, 0))

    return scaled_geometry


# splits a LineString into equal segments of a specified length and interpolates the sample point
# coordinates (outputs a list of shapely Point objects)
def fixedlength_segment_linestring(linestring, length_in_um):

    # scale the input linestring to um, then interpolate sample points along it (also in um)
    segment_points_in_um = []
    distance = 0
    linestring_in_um = scale_px_to_um(linestring)
    while distance < linestring_in_um.length:
        segment_points_in_um.append(linestring_in_um.interpolate(distance))
        distance += length_in_um
        if distance > linestring_in_um.length:
            break

    # create a numpy array of sample point coordinates for efficient scaling
    sample_point_coords_in_um = numpy.array([p.coords[0] for p in segment_points_in_um])

    # scale sample point coordinates from um to pixels (inverse of x and y resolution in um/px gives
    # resolution in px/um)
    ### SEE FUNCTION 'scale_px_to_um' FOR INFO ON WHAT 'x_resolution' AND 'y_resolution' MEAN ###
    x_resolution_um_per_px = 0.068 / (60/942)
    y_resolution_um_per_px = 0.068 / (59/920)
    scaled_x = sample_point_coords_in_um[:, 0] / x_resolution_um_per_px
    scaled_y = sample_point_coords_in_um[:, 1] / y_resolution_um_per_px

    # create a list of shapely Point objects from the scaled sample point coordinates (units are px)
    segment_points = [Point(x, y) for x, y in zip(scaled_x, scaled_y)]

    return segment_points


# cuts a LineString between two inputted points, preserving anchor points in between
def cut_line_between(linestring, pt_start, pt_end):

    # finds the geometric distance along LineString to the inputted points
    distance_start = linestring.project(pt_start)
    distance_end = linestring.project(pt_end)

    # cuts the LineString from 'd_start' to 'd_end', whilst preserving all original vertices on the
    # Linestring
    linestring_between = substring(linestring, distance_start, distance_end)

    return linestring_between


# calculates the medial axis of a polygon using Voronoi diagram
def create_medial_axis(line1, line2):

# Step 1: Increase the density of vertices on the two inputted LineStrings and make the vertices
# equidistant
    # dividing line length by 2 interpolates a point approximately every 2 pixels (approx because
    # fixednumber_segment_linestring truncates if not an integer)
    line1_points = fixednumber_segment_linestring(line1, line1.length / 2)
    line2_points = fixednumber_segment_linestring(line2, line2.length / 2)
    all_points = list(line1_points) + list(line2_points)
    all_points_coords = numpy.array([i.coords[0] for i in all_points])

# Step 2: Compute the Voronoi diagram
    voronoi_object = Voronoi(all_points_coords)

# Step 3: Filter and extract Voronoi ridges
    internal_ridges = []
    # to enable checking if Voronoi ridges are contained within the two inputted LineString, create
    # a polygon from the LineStrings and then add a small negative buffer to avoid floating point
    # issues
    polygon_from_linestrings = Polygon(list(line1.coords) + list(reversed(line2.coords)))
    buffered_polygon = polygon_from_linestrings.buffer(-1e-5)
    for v1_idx, v2_idx in voronoi_object.ridge_vertices:
        # exclude infinite ridges (those at edge of Voronoi diagram, they have an invalid index)
        if v1_idx == -1 or v2_idx == -1:
            continue
        # extract the coordinates of the ridge
        v1 = voronoi_object.vertices[v1_idx]
        v2 = voronoi_object.vertices[v2_idx]
        # create a LineString for the current ridge
        ridge_line = LineString([Point(v1), Point(v2)])
        # keep only ridges that are inside the polygon
        if buffered_polygon.contains(ridge_line):
            internal_ridges.append(ridge_line)

# Step 4: Find the longest continuous path of Voronoi ridges (uses double-sweep Dijkstra's shortest
# path algorithm)
    # build a graph with Voronoi vertices as nodes and ridge length as weights
    G = nx.Graph()
    for i in internal_ridges:
        coords = list(i.coords)
        p1, p2 = tuple(coords[0]), tuple(coords[1])
        length = i.length
        G.add_edge(p1, p2, weight=length, line=i)
    # find an arbitrary start node
    start_node = next(iter(G.nodes), None)
    # first Dijkstra to find farthest node from start_node (will find node at on of the edges)
    lengths, _ = nx.single_source_dijkstra(G, start_node, weight='weight')
    farthest_node1 = max(lengths, key=lengths.get)
    # second Dijkstra from farthest_node1 to find longest path (will find path from one edge to the
    # other)
    lengths, paths = nx.single_source_dijkstra(G, farthest_node1, weight='weight')
    farthest_node_2 = max(lengths, key=lengths.get)
    longest_path = paths[farthest_node_2]

    # Step 5: Build a medial axis LineString from the longest path
    medial_axis = LineString(longest_path)

    return medial_axis


# finds intersection points by extending perpendiculars from segment_points on the medial axis line
### NOTE THAT tangent_sample_dist HAS BEEN SET ARBITRARILY ###
def perpendicular_intersections(reference_line, longest_edge, opposite_edge, vertical_bin_width, horizontal_n_bins,
                                tangent_sample_dist=400, grow_normal_vec=2000):

    # splits the reference_line into segments of width vertical_bin_width (units in um), if
    # horizontal bins have been specified it splits into segments of width vertical_bin_width / 6
    if horizontal_n_bins != 0:
        segment_points = fixedlength_segment_linestring(reference_line, vertical_bin_width / 6)
    else:
        segment_points = fixedlength_segment_linestring(reference_line, vertical_bin_width)

    # make shapely polygon from the two LineStrings to ensure the tangent-normal grows in the correct
    # direction (ie: into the polygon shape)
    polygon_from_linestrings = Polygon(list(longest_edge.coords) + list(reversed(opposite_edge.coords)))

    paired_bin_boundary_coords = []
    for i in range(len(segment_points)):

        # store the first point on each LineString as the first set of boundary points
        if i == 0:
            paired_bin_boundary_coords.append([Point(longest_edge.coords[0]), Point(opposite_edge.coords[0])])

        # store the last point on each LineString as the last set of boundary points
        elif i == len(segment_points) - 1:
            paired_bin_boundary_coords.append([Point(longest_edge.coords[-1]), Point(opposite_edge.coords[-1])])

        # find perpendicular intersections for all other points on LineStrings
        else:
            point = segment_points[i]

            # find the distance of the point along the reference_line
            point_dist = reference_line.project(point)

            # find the distances along the reference_line forwards and backwards tangent_sample_dist
            # from point_dist and clamp distances to valid range of the reference_line
            dist_fwd = min(reference_line.length, point_dist + tangent_sample_dist)
            dist_back = max(0, point_dist - tangent_sample_dist)

            # sample multiple points in the point_back -> point_fwd window
            num_samples = 50
            window_sampled_distances = numpy.linspace(dist_back, dist_fwd, num_samples)
            window_sampled_points = [reference_line.interpolate(d) for d in window_sampled_distances]

            # calculate: (1) a tangent vector from each window sampled point to the next, (2) the
            # summed distance each tangent-pair window_sampled_points is to the current point that
            # an intersection line is being created from
            tangent_vectors = []
            tangent_dists = []
            for j in range(len(window_sampled_points) // 2):
                p_start = window_sampled_points[j]
                p_end = window_sampled_points[len(window_sampled_points) - j - 1]

                # (1)
                vec = numpy.array([p_end.x - p_start.x, p_end.y - p_start.y])
                tangent_vectors.append(vec)

                # (2)
                p_start_dist = window_sampled_distances[j]
                p_end_dist = window_sampled_distances[len(window_sampled_points) - j - 1]
                sum_dist_to_point = (point_dist - p_start_dist) + (p_end_dist - point_dist)
                tangent_dists.append(sum_dist_to_point)

            # extract the indices of tangent_dists sorted in ascending order
            sorted_indices = numpy.argsort(tangent_dists)

            # assign weights: largest weight (when tangent_dists is smallest) = len(tangent_dists),
            # smallest weight (when tangent_dists is largest) = 1
            weights = numpy.zeros(len(tangent_dists), dtype=int)
            for sorted_indices_idx, tangent_dists_idx in enumerate(sorted_indices):
                weights[tangent_dists_idx] = len(tangent_dists) - sorted_indices_idx

            # calculate the weighted average of all tangent vectors
            tangent_vec = numpy.average(tangent_vectors, axis=0, weights=weights)

            # (1) normalize the tangent vector to get a unit vector; (2) calculate normal unit
            # vector to tangent_unit_vec by rotating by 90 degrees
            tangent_unit_vec = tangent_vec / numpy.linalg.norm(tangent_vec)
            normal_vec = numpy.array([-tangent_unit_vec[1], tangent_unit_vec[0]])

            # compute test points by scaling the tangent-normal vector in each direction
            test_point_left = Point(point.x + normal_vec[0] * 2, point.y + normal_vec[1] * 2)
            test_point_right = Point(point.x - normal_vec[0] * 2, point.y - normal_vec[1] * 2)

            # scale the tangent-normal vector in the direction that extends into the polygon_from_linestrings
            if polygon_from_linestrings.contains(test_point_left) and not polygon_from_linestrings.contains(test_point_right):
                normal_point_left = Point(point.x + normal_vec[0] * grow_normal_vec, point.y + normal_vec[1] * grow_normal_vec)
                normal_line = LineString([point, normal_point_left])
            elif polygon_from_linestrings.contains(test_point_right) and not polygon_from_linestrings.contains(test_point_left):
                normal_point_right = Point(point.x - normal_vec[0] * grow_normal_vec, point.y - normal_vec[1] * grow_normal_vec)
                normal_line = LineString([point, normal_point_right])

            # check for intersections from the normal vector in both directions
            intersection_longest_edge = normal_line.intersection(longest_edge)
            #intersection_opposite_edge = normal_line.intersection(opposite_edge) # because taking opposite_edge as reference_line

            # helper function to find the closest point if multiple intersections occur
            def get_closest_point(intersection_geom, ref_point):
                if not intersection_geom.is_empty:
                    if intersection_geom.geom_type == 'Point':
                        return intersection_geom
                    elif intersection_geom.geom_type in ['MultiPoint', 'GeometryCollection']:
                        points = [p for p in intersection_geom.geoms if p.geom_type == 'Point']
                        if not points: return None
                        return min(points, key=lambda p: p.distance(ref_point))
                return None

            # output the closest intersection points to the reference_line
            point_on_longest_edge = get_closest_point(intersection_longest_edge, point)
            #point_on_opposite_edge = get_closest_point(intersection_opposite_edge, point) # because taking opposite_edge as reference_line

            # because taking opposite_edge as reference_line
            point_on_opposite_edge = point

            if point_on_longest_edge is not None and point_on_opposite_edge is not None:
                # successfully found a boundary on both lines
                paired_bin_boundary_coords.append([point_on_longest_edge, point_on_opposite_edge])
            else:
                # mark as NA if either intersection failed
                paired_bin_boundary_coords.append(['NA', point_on_opposite_edge])

    return paired_bin_boundary_coords


# combines the above functions and constructs vertical bins and prepares horizontal bins for
# construction
def make_bins(horizontal_n_bins, paired_bin_boundary_coords, longest_edge, opposite_edge, start_vert_bin_number=1, reprocessing="no"):

    n_vert_bins = len(paired_bin_boundary_coords) - 1

    # if horizontal bins have been specified: group vert bins accounting for making 6x as many in
    # perpendicular_intersections
    if horizontal_n_bins != 0:
        n_vert_groups = math.ceil(n_vert_bins / 6)
    else:
        n_vert_groups = n_vert_bins

    bin_poly_dict = []
    failed_bin_boundaries = [False] * n_vert_bins
    vert_group_fail_flags = {group: False for group in range(1, n_vert_groups + 1)}

    # loop over the number of vertical bins have got boundary coordinates for
    for i in range(n_vert_bins):
        vertical_number = i + start_vert_bin_number

        # if horizontal bins have been specified: group vert bins accounting for making 6x as many in
        # perpendicular_intersections
        if horizontal_n_bins != 0:
            current_group = math.floor((vertical_number - 1) / 6) + 1
        else:
            current_group = vertical_number

        # record and skip over when the intersection has failed, but not if reprocessing failed bins
        if reprocessing == "no" and (paired_bin_boundary_coords[i][0] == 'NA' or paired_bin_boundary_coords[i + 1][0] == 'NA'):
            vert_group_fail_flags[current_group] = True
            continue

        # define longest_edge and opposite_edge bin-boundary coordinates
        v_longest_edge_left_point = paired_bin_boundary_coords[i][0]
        v_longest_edge_right_point = paired_bin_boundary_coords[i + 1][0]
        v_opposite_edge_left_point = paired_bin_boundary_coords[i][1]
        v_opposite_edge_right_point = paired_bin_boundary_coords[i + 1][1]

        # cut longest_edge and opposite_edge at bin-boundary coordinates (obtains LineStrings of the segments)
        v_longest_edge_bin_linestring = cut_line_between(longest_edge, v_longest_edge_left_point, v_longest_edge_right_point)
        v_opposite_edge_bin_linestring = cut_line_between(opposite_edge, v_opposite_edge_left_point, v_opposite_edge_right_point)

        # extracts the coordinates of the vertical bin polygon
        vertical_bin_polygon_coords = list(v_longest_edge_bin_linestring.coords) + list(reversed(v_opposite_edge_bin_linestring.coords))

        # creates a shapely polygon from the vertical bin polygon coordinates
        vertical_bin_poly = Polygon(vertical_bin_polygon_coords)

        # record and skip over when the polygon made intersects itself, but not if reprocessing
        # failed bins (shapely documentation for is_simple is 'returns True if the feature does not
        # cross itself')
        if reprocessing == "no" and not vertical_bin_poly.is_simple:
            vert_group_fail_flags[current_group] = True
            continue

        # if horizontal binning is requested
        if horizontal_n_bins != 0:
            # define left/right edges of vertical bin
            left_edge = LineString([v_longest_edge_left_point.coords[0], v_opposite_edge_left_point.coords[0]])
            right_edge = LineString([v_longest_edge_right_point.coords[0], v_opposite_edge_right_point.coords[0]])

            # uses fixednumber_segment_linestring self-defined function
            h_left_edge_bin_points = fixednumber_segment_linestring(left_edge, horizontal_n_bins)
            h_right_edge_bin_points = fixednumber_segment_linestring(right_edge, horizontal_n_bins)

            for j in range(horizontal_n_bins):
                horizontal_number = j + 1

                # define left_edge and right_edge bin-boundary coordinates
                h_left_edge_bin_point_1 = h_left_edge_bin_points[j]
                h_left_edge_bin_point_2 = h_left_edge_bin_points[j + 1]
                h_right_edge_bin_point_1 = h_right_edge_bin_points[j]
                h_right_edge_bin_point_2 = h_right_edge_bin_points[j + 1]

                line_coords_parallel_LE = []
                if j == 0:
                    # include curved longest_edge edge
                    line_coords_parallel_LE.extend(v_longest_edge_bin_linestring.coords)
                else:
                    # straight edge
                    line_coords_parallel_LE.append(tuple(h_left_edge_bin_point_1.coords[0]))
                    line_coords_parallel_LE.append(tuple(h_right_edge_bin_point_1.coords[0]))

                line_coords_parallel_OE = []
                if j == horizontal_n_bins - 1:
                    # include curved opposite_edge edge
                    line_coords_parallel_OE.extend(v_opposite_edge_bin_linestring.coords)
                else:
                    # straight edge
                    line_coords_parallel_OE.append(tuple(h_left_edge_bin_point_2.coords[0]))
                    line_coords_parallel_OE.append(tuple(h_right_edge_bin_point_2.coords[0]))

                # stores a list of the preparation information to downstream create horizontal bin polygons
                bin_poly_dict.append({
                    "numbering": {"vertical": vertical_number, "horizontal": horizontal_number},
                    "group": current_group,
                    "line_coords_parallel_to_LE": line_coords_parallel_LE,
                    "line_coords_parallel_to_OE": line_coords_parallel_OE
                })

        else:
            # stores each vertical_bin_poly in a dict alongside its number (in order of creation)
            bin_poly_dict.append({
                "id": f"v{vertical_number}",
                "numbering": {"vertical": vertical_number, "horizontal": None},
                "group": current_group,
                "poly": vertical_bin_poly,
                "area": scale_px_to_um(vertical_bin_poly).area,
                "LE_vert_width": scale_px_to_um(v_longest_edge_bin_linestring).length,
                "OE_vert_width": scale_px_to_um(v_opposite_edge_bin_linestring).length,
                "horiz_height": numpy.nan
            })

    # record which bin boundaries are in a vert group that has failed, ensuring that if horizontal
    # bins specified that vert bins are grouped accounting for making 6x as many in
    # perpendicular_intersections
    if reprocessing == "no":
        for i in range(n_vert_bins):
            vertical_number = i + start_vert_bin_number
            if horizontal_n_bins != 0:
                current_group = math.floor((vertical_number - 1) / 6) + 1
            else:
                current_group = vertical_number
            if vert_group_fail_flags[current_group]:
                failed_bin_boundaries[i] = True

    # remove bin_poly_dict entries that are in a vert group that has failed
    failed_groups = {g for g, failed in vert_group_fail_flags.items() if failed}
    bin_poly_dict = [b for b in bin_poly_dict if b["group"] not in failed_groups]

    # delete the "group" key from bin_poly_dict
    for b in bin_poly_dict: del b["group"]

    return bin_poly_dict, failed_bin_boundaries


# outputs the LineStrings of contiguous patches of bin failure
def identify_LineStrings_failed_patches(failed_bin_boundaries, paired_bin_boundary_coords, longest_edge, opposite_edge):
    # only enter the main block if at least one bin failure exists
    if any(failed_bin_boundaries):

        failed_patches_LineString_info = []
        i = 0
        while i < len(failed_bin_boundaries):

            # identify the index of the start of a failed patch
            if failed_bin_boundaries[i]:
                start_failure_index = i

                # capture the starting bin number for this patch
                start_bin_num = start_failure_index + 1

                # identify the index of the end of the current failed patch
                j = i
                while j < len(failed_bin_boundaries) and failed_bin_boundaries[j]:
                    j += 1
                end_failure_index = j

                # the number of failed bins in the current failed patch
                num_failed_bins = end_failure_index - start_failure_index

                # the current patch starts at the last good boundary points before the failure patch
                # (ie: at start_failure_index), and ends at the first good boundary points after the
                # failure patch (ie: at end_failure_index)
                # this relies on successfully defining the very FIRST and LAST set of boundary
                # points on each input LineString (ie: idx 0 and max(idx) of
                # paired_bin_boundary_coords have coordinates) - this has been ensured in the
                # perpendicular_intersections function
                start_point_LE = paired_bin_boundary_coords[start_failure_index][0]
                start_point_OE = paired_bin_boundary_coords[start_failure_index][1]
                end_point_LE = paired_bin_boundary_coords[end_failure_index][0]
                end_point_OE = paired_bin_boundary_coords[end_failure_index][1]

                # extract the LineString segments for the current patch
                longest_edge_patch_linestring = cut_line_between(longest_edge, start_point_LE, end_point_LE)
                opposite_edge_patch_linestring = cut_line_between(opposite_edge, start_point_OE, end_point_OE)

                # if longest_edge_patch_linestring and opposite_edge_patch_linestring contain
                # LineStrings store them
                if longest_edge_patch_linestring and not longest_edge_patch_linestring.is_empty and opposite_edge_patch_linestring and not opposite_edge_patch_linestring.is_empty:
                    failed_patches_LineString_info.append({
                        "longest_edge_LineString": longest_edge_patch_linestring,
                        "opposite_edge_LineString": opposite_edge_patch_linestring,
                        "failed_bin_count": num_failed_bins,
                        "start_bin_number": start_bin_num
                    })

                # move the counter past this failure patch
                i = j

            else:
                i += 1

    else:
        # set to None if no bin failures exist
        failed_patches_LineString_info = None

    return failed_patches_LineString_info


# segments the LineStrings for each failed patch and returns a list of lists where each inner list
# contains a set of paired_bin_boundary_coords, with one per failed patch
def reprocess_failed_patches(failed_patches_LineString_info):
    list_patches_paired_bin_boundary_coords = []

    for i in failed_patches_LineString_info:
        # extract the two LineStrings for the current patch
        patch_longest_edge_LineString = i["longest_edge_LineString"]
        patch_opposite_edge_LineString = i["opposite_edge_LineString"]

        # the number of new bins should equal the number of failed bins
        num_new_segments = i["failed_bin_count"]

        # segment each LineString in the patch using fixednumber_segment_linestring self-defined
        # function
        segment_points_patch_longest_edge = fixednumber_segment_linestring(patch_longest_edge_LineString, num_new_segments)
        segment_points_patch_opposite_edge = fixednumber_segment_linestring(patch_opposite_edge_LineString, num_new_segments)

        # zip the corresponding points together from each LineString into desired [p1, p2] format
        patch_paired_bin_boundary_coords = [[p1, p2] for p1, p2 in zip(segment_points_patch_longest_edge, segment_points_patch_opposite_edge)]

        list_patches_paired_bin_boundary_coords.append(patch_paired_bin_boundary_coords)

    return list_patches_paired_bin_boundary_coords


# finishes the construction of vert+horiz bins (prepared from make_bins) by combining vert bins in
# groups of 6 (within horizontal layers) and smoothing their edges
def finish_making_VertHoriz_bins(bin_poly_dict, horizontal_n_bins):

    # group horizontal line coords by their horizontal bin number and into vertical groups of 6
    from collections import defaultdict
    combined_groups = defaultdict(list)
    for i in bin_poly_dict:
        h_group = i["numbering"]["horizontal"]
        vertical_bin_number = i["numbering"]["vertical"]
        v_group = math.floor((vertical_bin_number - 1) / 6) + 1
        combined_groups[(h_group, v_group)].append((i["line_coords_parallel_to_LE"], i["line_coords_parallel_to_OE"]))

    final_bin_poly_dict = []
    for (h_group, v_group), bin_group in sorted(combined_groups.items()):

        # flatten the lists of coordinates for horizontal line coords in the current bin group
        flattened_line_coords_parallel_to_LE = list(chain.from_iterable([LE_line for LE_line, _ in bin_group]))
        flattened_line_coords_parallel_to_OE = list(chain.from_iterable([OE_line for _, OE_line in bin_group]))

        # remove duplicates while preserving order of coordinates
        cleaned_line_coords_parallel_to_LE = []
        for coord in flattened_line_coords_parallel_to_LE:
            if coord not in cleaned_line_coords_parallel_to_LE:
                cleaned_line_coords_parallel_to_LE.append(coord)

        cleaned_line_coords_parallel_to_OE = []
        for coord in flattened_line_coords_parallel_to_OE:
            if coord not in cleaned_line_coords_parallel_to_OE:
                cleaned_line_coords_parallel_to_OE.append(coord)

        # make LineStrings
        linestring_parallel_to_LE = LineString(cleaned_line_coords_parallel_to_LE)
        linestring_parallel_to_OE = LineString(cleaned_line_coords_parallel_to_OE)

        # don't smooth bin edges on longest_edge and opposite_edge
        if h_group == 1:
            smoothed_linestring_LEdirection = linestring_parallel_to_LE
            smoothed_linestring_OEdirection = taubin_smooth(linestring_parallel_to_OE, steps=40)
        elif h_group == horizontal_n_bins:
            smoothed_linestring_LEdirection = taubin_smooth(linestring_parallel_to_LE, steps=40)
            smoothed_linestring_OEdirection = linestring_parallel_to_OE
        else:
            smoothed_linestring_LEdirection = taubin_smooth(linestring_parallel_to_LE, steps=40)
            smoothed_linestring_OEdirection = taubin_smooth(linestring_parallel_to_OE, steps=40)

        # extract coordinates from the LineStrings to make the bin polygon
        bin_polygon_coords = list(smoothed_linestring_LEdirection.coords) + list(reversed(smoothed_linestring_OEdirection.coords))

        # creates a shapely polygon
        bin_poly = Polygon(bin_polygon_coords)

        # calculate the horizontal bin height by averaging the heights at the two bin ends
        scaled_left_edge_length = Point(scale_px_to_um(smoothed_linestring_LEdirection).coords[0]).distance(Point(scale_px_to_um(smoothed_linestring_OEdirection).coords[0]))
        scaled_right_edge_length = Point(scale_px_to_um(smoothed_linestring_LEdirection).coords[-1]).distance(Point(scale_px_to_um(smoothed_linestring_OEdirection).coords[-1]))
        scaled_horiz_height = (scaled_left_edge_length + scaled_right_edge_length) / 2

        final_bin_poly_dict.append({
            "id": f"v{v_group}_h{h_group}",
            "numbering": {"horizontal": h_group, "vertical": v_group},
            "poly": bin_poly,
            "area": scale_px_to_um(bin_poly).area,
            "LE_vert_width": scale_px_to_um(smoothed_linestring_LEdirection).length,
            "OE_vert_width": scale_px_to_um(smoothed_linestring_OEdirection).length,
            "horiz_height": scaled_horiz_height
        })

    return final_bin_poly_dict


# combines vertical bins into horizontal layers
def combine_vertical_bins(bin_poly_dict):

    # group bin polygons by their horizontal bin number
    from collections import defaultdict
    horizontal_bin_groups = defaultdict(list)
    for i in bin_poly_dict:
        horizontal_bin_groups[i["numbering"]["horizontal"]].append((i["poly"], i["horiz_height"]))

    # combine vertical bins using unary_union (combines all rois at least touching)
    horizontal_combined_bin_poly_dict = []
    for j in sorted(horizontal_bin_groups.keys()):
        cleaned_group = [poly.buffer(1e-9) for poly, _ in horizontal_bin_groups[j]]
        horizontal_combined_poly = unary_union(cleaned_group).buffer(-1e-9)
        av_horizontal_bin_heights = numpy.mean([height for _, height in horizontal_bin_groups[j]])
        horizontal_combined_bin_poly_dict.append({
            "id": f"h{j}", "numbering": {"vertical": None, "horizontal": j},
            "poly": horizontal_combined_poly,
            "area": scale_px_to_um(horizontal_combined_poly).area,
            "LE_vert_width": numpy.nan,
            "OE_vert_width": numpy.nan,
            "horiz_height": av_horizontal_bin_heights
        })

    return horizontal_combined_bin_poly_dict


# saves each bin as a separate .roi file
def convert_bin_polys_to_rois(bin_poly_dict):

    bin_rois_dict = []
    for i in bin_poly_dict:

        poly = i["poly"]
        if poly.geom_type == "Polygon":
            rois = [ImagejRoi.frompoints(list(poly.exterior.coords))]
        elif poly.geom_type == "MultiPolygon":
            rois = [ImagejRoi.frompoints(list(p.exterior.coords)) for p in poly.geoms]
        else:
            print(f"Unexpected bin geometry type: {poly.geom_type}")
            continue

        # for some reason I don't know, you have to leave the for loop
        id = i["id"]
        for roi in rois:
            bin_rois_dict.append({
                "id": id,
                "roi": roi
            })

    return bin_rois_dict


# opens a zip file, extracts its .roi files, converts each ROI to a shapely polygon and stores its
# name as an ID
### NOTE THAT COMPOSITE ROIS ARE REDUCED, ONLY RETAINING THE LARGEST POLYGON (I think by area) ###
def rois_to_polys(zip_path):

    # create a temporary directory to extract individual .roi files
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        roi_files = glob.glob(os.path.join(tmpdir, '*.roi'))

        region_poly_dict = []
        for i in roi_files:
            roi = ImagejRoi.fromfile(i)
            xs, ys = roi.coordinates().T
            poly = Polygon(zip(xs, ys)).buffer(1e-9)
            region_poly_dict.append({
                "id": roi.name,
                "poly": poly
            })

    return region_poly_dict


# determines if each bin is located within an area of interest - this can be: delineated regions,
# areas of tissue damage, or cortical folds (gyri and sulci)
def assign_bins_to_areas_of_interest(bins_dict, areas_interest_dict):
    assignments = {}
    bins_assigned_in_pass_one = set()

    # Pass 1: assign a delineated region to bins that are completely contained within that given
    # region (uses shapely contain function)
    for current_bin in bins_dict:
        for current_area in areas_interest_dict:
            if current_area["poly"].contains(current_bin["poly"]):
                assignments[current_bin["id"]] = current_area["id"]
                bins_assigned_in_pass_one.add(current_bin["id"])
                break

    # Pass 2: for bins located at the boundaries between regions, assign the region that the bin has
    # the greatest area inside
    unassigned_bins = [i for i in bins_dict if i["id"] not in bins_assigned_in_pass_one]

    for current_bin in unassigned_bins:
        max_overlapped_area = 0
        largest_overlapped_region = None
        current_bin_poly = current_bin["poly"].buffer(0)

        for current_area in areas_interest_dict:
            current_area_poly = current_area["poly"].buffer(0)
            if not current_bin_poly.intersects(current_area_poly):
                continue
            overlapped_area = current_bin_poly.intersection(current_area_poly).area

            if overlapped_area > max_overlapped_area:
                max_overlapped_area = overlapped_area
                largest_overlapped_region = current_area["id"]

        assignments[current_bin["id"]] = largest_overlapped_region

    return assignments


# returns true if the inputted directory is not empty, returns false otherwise
def directory_not_empty(folder_path):
    return os.path.isdir(folder_path) and len(os.listdir(folder_path)) > 0






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", help="File path to the edge detection output")
    parser.add_argument("--vertical_bin_width", type=int, help="Number of vertical bins to create")
    parser.add_argument("--horizontal_n_bins", type=int, help="Number of horizontal bins to create")
    args = parser.parse_args()

    # for horizontal only layer bins, combine vertical bins constructed from width of 100 µm
    if args.vertical_bin_width == 0:
        FOR_BINNING_vertical_bin_width = 100
    else:
        FOR_BINNING_vertical_bin_width = args.vertical_bin_width

    # loop over each subfolder
    for subfolder in sorted(os.listdir(args.directory)):
        subfolder_path = os.path.join(args.directory, subfolder)

        # look for edge detection folder inside subfolder
        edge_detection_folder = next(glob.iglob(os.path.join(subfolder_path, "edge_detect_*")), None)
        if not edge_detection_folder:
            print(f"No edge detection folder found in: {subfolder_path}")
            continue

        # defining output directory name
        if args.horizontal_n_bins != 0:
            if args.vertical_bin_width == 0:
                binning_info = f"h{args.horizontal_n_bins}n"
            else:
                binning_info = f"v{args.vertical_bin_width}µm,h{args.horizontal_n_bins}n"
        else:
            binning_info = f"v{args.vertical_bin_width}µm"

        output_dir = os.path.join(subfolder_path, f"binning({binning_info})_{subfolder}")

        # create or overwrite output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        for i in os.listdir(edge_detection_folder):
            if i.endswith('.geojson'):

                geojson_path = os.path.join(edge_detection_folder, i)
                longest_edge, opposite_edge = load_linestrings_from_geojson(geojson_path)

                ### **NOT IN USE CURRENTLY** MEDIAL AXIS PERPENDICULARS CODE ###
                #reference_line = reate_medial_axis(longest_edge, opposite_edge)

                ### **NOT IN USE CURRENTLY (because merged anchor points are too close together - could use taubin smoothing instead)** SMOOTHED REFERENCE LINE CODE ###
                # Chaikin interpolation was chosen because it cuts sharp, artificial corners
                # to create a fluid curve - although it can cause shrinkage of the smoothed
                # boundary, I deemed it ok because of the low degree of smoothing used and
                # that very little shrinkage was seen at this degree of smoothing (actually
                # less so than Taubin smoothing which is meant to resist shrinkage)
                #reference_line = chaikin_smooth(opposite_edge, iters=3)

                reference_line = opposite_edge

                paired_bin_boundary_coords = perpendicular_intersections(reference_line, longest_edge, opposite_edge, FOR_BINNING_vertical_bin_width, args.horizontal_n_bins)

                bin_poly_dict, failed_bin_boundaries = make_bins(args.horizontal_n_bins, paired_bin_boundary_coords, longest_edge, opposite_edge)

                failed_patches_LineString_info = identify_LineStrings_failed_patches(failed_bin_boundaries, paired_bin_boundary_coords, longest_edge, opposite_edge)

                if failed_patches_LineString_info is not None:
                    list_patches_paired_bin_boundary_coords = reprocess_failed_patches(failed_patches_LineString_info)
                    for idx, patch_info in enumerate(failed_patches_LineString_info):

                        # get paired bin boundary coordinates for the current patch
                        patch_paired_bin_boundary_coords = list_patches_paired_bin_boundary_coords[idx]

                        # get original start_vert_bin_number for the current patch
                        original_vert_bin_number = patch_info["start_bin_number"]

                        # rerun make_bins on the current patch, providing start_vert_bin_number
                        reprocessed_bin_poly_dict, _ = make_bins(args.horizontal_n_bins, patch_paired_bin_boundary_coords, longest_edge, opposite_edge, start_vert_bin_number=original_vert_bin_number, reprocessing="yes")

                        # add the reprocessed data to bin_poly_dict and then order, first, based on
                        # vertical bin number and, second, based on horizontal bin number
                        bin_poly_dict.extend(reprocessed_bin_poly_dict)
                        bin_poly_dict.sort(key=lambda x: (x["numbering"]["vertical"], x["numbering"]["horizontal"]))

                if args.horizontal_n_bins != 0:
                    bin_poly_dict = finish_making_VertHoriz_bins(bin_poly_dict, args.horizontal_n_bins)

                if args.vertical_bin_width == 0:
                    bin_poly_dict = combine_vertical_bins(bin_poly_dict)

                bin_rois_dict = convert_bin_polys_to_rois(bin_poly_dict)


                # determining: (1) which delineated region each bin is located within; (2) which
                # bins are located within areas of tissue damage (eg: tissue folding); (3) which
                # bins are located within cortical folds (gyri or sulci)
                region_roi_dir_path = os.path.join(subfolder_path, "region_rois")
                damage_roi_dir_path = os.path.join(subfolder_path, "damage_rois")
                cortical_curve_roi_dir_path = os.path.join(subfolder_path, "cortical_curve_rois")
                if directory_not_empty(region_roi_dir_path) or directory_not_empty(damage_roi_dir_path) or directory_not_empty(cortical_curve_roi_dir_path) and args.vertical_bin_width > 0:
                    bin_poly_dict_within = []
                    for j in bin_poly_dict:
                        bin_poly_dict_within.append({
                            "id": j["id"],
                            "poly": j["poly"]
                        })

                    # (1)
                    if directory_not_empty(region_roi_dir_path):
                        region_roi_zip_path = next((os.path.join(region_roi_dir_path, f) for f in os.listdir(region_roi_dir_path) if f.endswith('.zip')), None)
                        region_poly_dict = rois_to_polys(region_roi_zip_path)
                        region_assigned_bins_dict = assign_bins_to_areas_of_interest(bin_poly_dict_within, region_poly_dict)

                        # check that all bins have been assigned a delineated region
                        for bin_id, region in region_assigned_bins_dict.items():
                            if region is None:
                                print(f"For {subfolder_path} - {os.path.splitext(i)[0]} bin {bin_id} was not assigned a delineated region")

                    # (2)
                    if directory_not_empty(damage_roi_dir_path):
                        damage_roi_zip_path = next((os.path.join(damage_roi_dir_path, f) for f in os.listdir(damage_roi_dir_path) if f.endswith('.zip')), None)
                        damage_poly_dict = rois_to_polys(damage_roi_zip_path)
                        damage_assigned_bins_dict = assign_bins_to_areas_of_interest(bin_poly_dict_within, damage_poly_dict)

                        # reassign damage_assigned_bins_dict["id"] to True
                        for bin_id, damage in damage_assigned_bins_dict.items():
                            if damage is not None:
                                damage_assigned_bins_dict[bin_id] = True

                    # (3)
                    if directory_not_empty(cortical_curve_roi_dir_path):
                        cortical_curve_roi_zip_path = next((os.path.join(cortical_curve_roi_dir_path, f) for f in os.listdir(cortical_curve_roi_dir_path) if f.endswith('.zip')), None)
                        cortical_curve_poly_dict = rois_to_polys(cortical_curve_roi_zip_path)
                        cortical_curve_assigned_bins_dict = assign_bins_to_areas_of_interest(bin_poly_dict_within, cortical_curve_poly_dict)

                        # reassign cortical_curve_assigned_bins_dict["id"] to either 'convex' or
                        # 'concave' (defining as concave or convex is with respect to the
                        # opposite_edge or the edge at the white-grey matter boundary)
                        for bin_id, curve in cortical_curve_assigned_bins_dict.items():
                            if curve is not None:
                                if "convex" in curve:
                                    cortical_curve_assigned_bins_dict[bin_id] = "convex"
                                if "concave" in curve:
                                    cortical_curve_assigned_bins_dict[bin_id] = "concave"


                # .roi file naming and saving
                for j in bin_rois_dict:
                    roi_obj = j["roi"]
                    bin_id = j["id"]
                    roi_name = f"{bin_id}.roi"
                    roi_output_path = os.path.join(output_dir, roi_name)
                    with open(roi_output_path, 'wb') as f:
                        f.write(roi_obj.tobytes())


                # defining .zip file name
                geojsonfile_base_name = os.path.splitext(i)[0]
                roi_zip_name = f"bins({binning_info})_{geojsonfile_base_name}_{subfolder}.zip"

                ### DEBUGGER - OUTPUTTING ROI LINE OF 'reference_line' ###
                x, y = reference_line.xy
                reference_line_coords = list(zip(map(int, x), map(int, y)))
                reference_line_roi = ImagejRoi.frompoints(reference_line_coords)
                reference_line_roi.tofile(os.path.join(output_dir, f"reference_line_{geojsonfile_base_name}.roi"))

                # create or overwrite .zip file inside output_dir
                zip_path = os.path.join(output_dir, roi_zip_name)
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    # move each .roi into .zip file then delete .roi
                    for filename in os.listdir(output_dir):
                        if filename.endswith(".roi"):
                            if "reference_line_" in filename:
                                continue
                            else:
                                roi_file_path = os.path.join(output_dir, filename)
                                zipf.write(roi_file_path, arcname=filename)
                                os.remove(roi_file_path)


                # check that the expected number of bins has been generated
                if args.horizontal_n_bins != 0:
                    if args.vertical_bin_width == 0:
                        expected_n_bins = args.horizontal_n_bins
                    else:
                        expected_n_bins = math.ceil((len(paired_bin_boundary_coords) - 1) / 6) * args.horizontal_n_bins
                else:
                    expected_n_bins = len(paired_bin_boundary_coords) - 1

                if len(bin_poly_dict) != expected_n_bins:
                    print(f"\nExpected number of bins NOT generated for: {geojsonfile_base_name}_{subfolder}")
                    print(f"Generated / Expected: {len(bin_poly_dict)} / {expected_n_bins}\n")


                # outputting roi binning information as a .tsv file
                bin_measures_df = pandas.DataFrame({
                    "bin_id": [entry["id"] for entry in bin_poly_dict],
                    "area.um2": [entry["area"] for entry in bin_poly_dict],
                    "vert.width.longest.edge.um": [entry["LE_vert_width"] for entry in bin_poly_dict],
                    "vert.width.opposite.edge.um": [entry["OE_vert_width"] for entry in bin_poly_dict],
                    "horiz.height.midline.um": [entry["horiz_height"] for entry in bin_poly_dict],
                })

                if "region_assigned_bins_dict" in globals():
                    bin_measures_df["delineated_region"] = bin_measures_df["bin_id"].map(region_assigned_bins_dict)

                if "damage_assigned_bins_dict" in globals():
                    bin_measures_df["damaged_area"] = bin_measures_df["bin_id"].map(damage_assigned_bins_dict)

                if "cortical_curve_assigned_bins_dict" in globals():
                    bin_measures_df["cortical_fold"] = bin_measures_df["bin_id"].map(cortical_curve_assigned_bins_dict)

                # round only the columns that are not full of NAs to 2 decimal places
                numeric_cols = ["area.um2", "vert.width.longest.edge.um", "vert.width.opposite.edge.um", "horiz.height.midline.um"]
                for col in numeric_cols:
                    if not bin_measures_df[col].isna().all():
                        bin_measures_df[col] = bin_measures_df[col].round(2)
                bin_measures_df[numeric_cols] = bin_measures_df[numeric_cols].round(2)

                # open a text file and write the bin_measures dataframe
                bin_measures_df.to_csv(os.path.join(output_dir,f"bin({binning_info})_measures_{geojsonfile_base_name}_{subfolder}.txt"), sep='\t', index=False)

    print(f"\n  COMPLETED SUCCESSFULLY\n")