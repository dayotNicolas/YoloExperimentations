from Viewtransformer import ViewTransformer
from ultralytics import YOLO
from collections import defaultdict, deque
from tqdm import tqdm
from Middlewares import *
import os
from dotenv import load_dotenv
load_dotenv()

def main():
    # get polygones config
    zones = load_polygone_config('./zones.yml')["zones"]
    targets_width = get_targets_width(zones)
    targets_height = get_targets_height(zones)

    MODEL_NAME = os.getenv("MODEL_NAME")
    SOURCE_VIDEO_PATH = os.getenv("SOURCE_VIDEO_PATH")
    TARGET_VIDEO_PATH = os.getenv("TARGET_VIDEO_PATH")
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD"))
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD"))
    MODEL_RESOLUTION = os.getenv("MODEL_RESOLUTION")

    model = YOLO(MODEL_NAME)

    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

    # tracer initiation
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_thresh=CONFIDENCE_THRESHOLD
    )

    # annotators configuration
    thickness = sv.calculate_dynamic_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_dynamic_text_scale(
        resolution_wh=video_info.resolution_wh
    )
    bounding_box_annotator = sv.BoundingBoxAnnotator(
        thickness=thickness
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER
    )

    polygons = get_polygones_list(zones, video_info)
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    coordinatespxl = defaultdict(lambda: deque(maxlen=video_info.fps))

    # open target video
    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        nbframe = video_info.total_frames
        # loop over source video frame
        k=0
        for frame in tqdm(frame_generator, total=nbframe):
            k+=1
            result = model(frame, imgsz=int(MODEL_RESOLUTION), verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            # filter out detections by class and confidence
            detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
            detections = detections[detections.class_id == 2] or detections[detections.class_id == 3] or detections[detections.class_id == 7] 
            annotated_frame = frame.copy()
            i=0
            for polygon in polygons:
                TARGET = np.array([
                    [0, 0],
                    [targets_width[i] - 1, 0],
                    [targets_width[i] - 1, targets_height[i] - 1],
                    [0, targets_height[i] - 1],
                ])
                view_transformer = ViewTransformer(source=np.array(zones[i]['polygon']), target=TARGET)
                annotated_frame = process_frame(annotated_frame, polygon, detections, IOU_THRESHOLD, byte_track, view_transformer, coordinates, video_info, trace_annotator, bounding_box_annotator, label_annotator, i, coordinatespxl, k)
                i+=1
                

            # add frame to target video
            sink.write_frame(annotated_frame)


if __name__ == "__main__":
    main()