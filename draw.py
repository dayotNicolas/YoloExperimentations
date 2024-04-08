import os
from dotenv import load_dotenv
from Middlewares import load_polygone_config, draw_zones
import supervision as sv
load_dotenv()

SOURCE_VIDEO_PATH = os.getenv("SOURCE_VIDEO_PATH")

zones = load_polygone_config("./zones.yml")["zones"]

frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
frame_iterator = iter(frame_generator)
frame = next(frame_iterator)
annotated_frame = frame.copy()

draw_zones(annotated_frame, zones)
sv.plot_image(annotated_frame)
