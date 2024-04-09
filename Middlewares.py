import yaml
import math
import numpy as np
import supervision as sv
from Output import DetectionOutput
import csv
from datetime import datetime
import os

def load_polygone_config(yml_path):
    # Load YAML zones file
    with open(yml_path, 'r') as file:
        zones = yaml.safe_load(file)
    
    return zones

def draw_zones(annotated_frame, zones):
    Colors = [sv.Color.RED, sv.Color.BLUE, sv.Color.GREEN, sv.Color.YELLOW]
    i=0
    for zone in zones:
        polygon = np.array(zone['polygon'])
        annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=polygon, color=Colors[i], thickness=4)
        i+=1

    return annotated_frame

def get_polygones_list(zones, video_info):
    
    polygon_list = []

    for zone in zones:
        polygon = np.array(zone['polygon'])
        polygon_zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)
        polygon_list.append(polygon_zone)
    
    return polygon_list

def get_targets_width(zones):
    targets_width = []
    for zone in zones:
        targets_width.append(zone['target_width'])
    return targets_width

def get_targets_height(zones):
    targets_height = []
    for zone in zones:
        targets_height.append(zone['target_height'])
    return targets_height

def calculate_proximity_threshold(speed_kmh):
    """
    Calculate the proximity threshold based on the tens digit of the speed in kilometers per hour.
    The proximity threshold is calculated by taking the tens digit of the speed and multiplying it by 6.
    """
    # Convert speed to integer to extract the tens digit
    speed_integer = int(speed_kmh)

    # Extract the tens digit of the speed
    tens_digit = (speed_integer // 10) % 10

    # Calculate proximity threshold
    proximity_threshold = tens_digit * 6

    return proximity_threshold

def get_detection_bbox(detections, tracker_id):
    """
    Get the bounding box (bbox) for a specific tracker_id from the Detections object.
    """
    # Ensure tracker_id exists in the tracker_id array
    if tracker_id in detections.tracker_id:
        # Find the index of the tracker_id in the tracker_id array
        index = np.where(detections.tracker_id == tracker_id)[0][0]

        # Extract bounding box coordinates (xyxy format)
        bbox = detections.xyxy[index]

        return bbox
    else:
        return None  # Tracker ID not found in detections
    
def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def check_proximity(coordinates, detections_zone, current_tracker_id, proximity_threshold):
    """
    Check if the current tracked vehicle is too close (danger condition) to any other vehicle.
    """
    for tracker_id in detections_zone.tracker_id:
            if tracker_id != current_tracker_id:
                distance = calculate_distance(x1=coordinates[current_tracker_id][-1][0], y1=coordinates[current_tracker_id][-1][1], x2=coordinates[tracker_id][-1][0], y2=coordinates[tracker_id][-1][1])
                #abs(coordinates[current_tracker_id][-1] - coordinates[tracker_id][-1])
            
                # Check if the distance is less than the proximity threshold
                if distance < proximity_threshold:
                    return True  # Danger condition (too close)

    return False

def write_object_to_csv(obj):
    # Get current date in YYYY-MM-DD format
    current_date = datetime.now().strftime('%Y-%m-%d')
    reslutDirectory = './datas/detections_datas/'

    # Define CSV file path with date in the name
    csv_file = f'{reslutDirectory}objects_data_{current_date}.csv'

    # Define field names based on object attributes
    field_names = ['tracker_id', 'class_id', 'pixel_coords', 'meter_coords', 'speed(km/h)', 'speed(pixel/s)', 'distance_warning', 'frame_number']

    # Check if the CSV file already exists or not
    file_exists = os.path.isfile(csv_file)

    # Write object data to CSV file
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)

        # Write header if the file is newly created
        if not file_exists:
            writer.writeheader()

        # Write object data as a new row
        writer.writerow({'tracker_id': obj.tracker_id, 'class_id': obj.class_id, 'pixel_coords': obj.pixel_coords, 'meter_coords': obj.meter_coords, 'speed(km/h)': obj.speed, 'speed(pixel/s)': obj.speedpxl, 'distance_warning': obj.isDanger, 'frame_number': obj.frame_number})

def process_frame(
        annotated_frame,
        polygon,
        detections, 
        IOU_THRESHOLD, 
        byte_track, 
        view_transformer, 
        coordinates, 
        video_info,
        trace_annotator,
        bounding_box_annotator,
        label_annotator,
        zone,
        coordinatespxl,
        frameNbr
        ):
    #print(f"Zone : {zone}")
    # format labels
    labels = []
    # refine detections using non-max suppression
    detections = detections.with_nms(IOU_THRESHOLD)
    # pass detection through the tracker
    detections = byte_track.update_with_detections(detections=detections)
    # filter out detections outside the zone
    detections_zone = detections[polygon.trigger(detections)]
    points = detections_zone.get_anchors_coordinates(
        anchor=sv.Position.BOTTOM_CENTER
    )

    # calculate the detections position inside the target RoI
    points = view_transformer.transform_points(points=points).astype(int)
    # store detections position
    for tracker_id, [x, y] in zip(detections_zone.tracker_id, points):
        coordinates[tracker_id].append((x,y))
    
    for tracker_id, [x, y] in zip(detections_zone.tracker_id, detections_zone.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)):
        coordinatespxl[tracker_id].append((x,y))

    p = 0
    for tracker_id in detections_zone.tracker_id:
        
        if len(coordinates[tracker_id]) < video_info.fps / 2:
            labels.append(f"#{tracker_id}")
        else:
            # calculate speed
            
            coordinate_start = coordinates[tracker_id][-1]
            coordinate_startpxl = coordinatespxl[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            coordinate_endpxl = coordinatespxl[tracker_id][0]
            distance = calculate_distance(x1=coordinate_start[0], y1=coordinate_start[1], x2=coordinate_end[0], y2=coordinate_end[1])
            distancepxl = calculate_distance(x1=coordinate_startpxl[0], y1=coordinate_startpxl[1], x2=coordinate_endpxl[0], y2=coordinate_endpxl[1])
            #distance = abs(coordinate_start - coordinate_end)
            time = len(coordinates[tracker_id]) / video_info.fps
            speedKmh = distance / time * 3.6
            speedpxlsec = distancepxl / time

            
            # Calculate proximity threshold based on speed
            proximity_threshold = calculate_proximity_threshold(speedKmh)
            #print(f"Proximity_threshold : {proximity_threshold}")
            # Check proximity to other vehicles (danger condition)
            is_danger = check_proximity(coordinates, detections_zone, tracker_id, proximity_threshold)
            
            DetectOutputObject = DetectionOutput(tracker_id, detections_zone.class_id[p], coordinatespxl[tracker_id][-1], coordinates[tracker_id][-1], speedKmh, speedpxlsec, is_danger, frameNbr)
            write_object_to_csv(DetectOutputObject)
            p+=1

            if is_danger:
                labels.append(f"#{tracker_id} {int(speedKmh)} km/h {int(speedpxlsec)} px/s DANGER")
            else:
                labels.append(f"#{tracker_id} {int(speedKmh)} km/h {int(speedpxlsec)} px/s")

    # annotate frame
    
    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame, detections=detections_zone
    )
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame, detections=detections_zone
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections_zone, labels=labels
    )
    #print("DETECTION ZONE : ")
    #print(detections_zone)


    return annotated_frame