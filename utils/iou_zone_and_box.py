from shapely.geometry import box,Polygon
import ast

def calculate_iou(box_coords, polygon_coords):
    """Calculate the Intersection over Union (IoU) between a bounding box and a polygon."""
    # Create a bounding box polygon
    bbox_polygon = box(*box_coords)  # Unpack box_coords directly
    zone_polygon = Polygon(polygon_coords)

    # Calculate intersection and union
    intersection_area = bbox_polygon.intersection(zone_polygon).area
    union_area = bbox_polygon.union(zone_polygon).area

    # Calculate IoU
    return intersection_area / union_area if union_area > 0 else 0.0


def parse_string(zone_string):
    zone_string = zone_string.replace("np.array", "")
    zone_list = ast.literal_eval(zone_string)

    return zone_list