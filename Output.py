

class DetectionOutput:
    def __init__(self, tracker_id, class_id, pixel_coords, meter_coords, speed, speedpxl, frame_number):
        self.tracker_id = tracker_id
        self.class_id = class_id
        self.pixel_coords = pixel_coords
        self.meter_coords = meter_coords
        self.speed = speed
        self.speedpxl = speedpxl
        self.frame_number = frame_number