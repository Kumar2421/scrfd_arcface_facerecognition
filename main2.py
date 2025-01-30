import os
import cv2
import random
import warnings
import argparse
import logging
import numpy as np

import onnxruntime
from typing import Union, List, Tuple
from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection-and-Recognition")
    parser.add_argument("--det-weight", type=str, default="./weights/det_10g.onnx", help="Path to detection model")
    parser.add_argument("--rec-weight", type=str, default="./weights/w600k_r50.onnx", help="Path to recognition model")
    parser.add_argument("--similarity-thresh", type=float, default=0.4, help="Similarity threshold between faces")
    parser.add_argument("--confidence-thresh", type=float, default=0.5, help="Confidence threshold for face detection")
    parser.add_argument("--faces-dir", type=str, default="./faces", help="Path to faces stored dir")
    parser.add_argument("--max-num", type=int, default=0, help="Maximum number of face detections from a frame")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

def connect_to_camera(ip, username, password):
    url = f"rtsp://{username}:{password}@{ip}/cam/realmonitor?channel=1&subtype=0"
    vs = cv2.VideoCapture(url)
    return vs

def build_targets(detector, recognizer, params: argparse.Namespace) -> List[Tuple[np.ndarray, str]]:
    targets = []
    for filename in os.listdir(params.faces_dir):
        name = filename[:-4]
        image_path = os.path.join(params.faces_dir, filename)

        image = cv2.imread(image_path)
        bboxes, kpss = detector.detect(image, max_num=1)

        if len(kpss) == 0:
            logging.warning(f"No face detected in {image_path}. Skipping...")
            continue

        embedding = recognizer(image, kpss[0])
        targets.append((embedding, name))
    return targets

def frame_processor(frame: np.ndarray, detector: SCRFD, recognizer: ArcFace, targets: List[Tuple[np.ndarray, str]], colors: dict, params: argparse.Namespace) -> np.ndarray:
    bboxes, kpss = detector.detect(frame, params.max_num)

    for bbox, kps in zip(bboxes, kpss):
        *bbox, conf_score = bbox.astype(np.int32)
        embedding = recognizer(frame, kps)

        max_similarity = 0
        best_match_name = "Unknown"
        for target, name in targets:
            similarity = compute_similarity(target, embedding)
            if similarity > max_similarity and similarity > params.similarity_thresh:
                max_similarity = similarity
                best_match_name = name

        if best_match_name != "Unknown":
            color = colors[best_match_name]
            draw_bbox_info(frame, bbox, similarity=max_similarity, name=best_match_name, color=color)
        else:
            draw_bbox(frame, bbox, (255, 0, 0))

    return frame

def main(params):
    setup_logging(params.log_level)
    detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
    recognizer = ArcFace(params.rec_weight)
    targets = build_targets(detector, recognizer, params)
    colors = {name: (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) for _, name in targets}
    
    source_1 = connect_to_camera('192.168.29.208', 'admin', 'Libs2000@')
    source_2 = connect_to_camera('192.168.29.209', 'admin', 'Libs2000@')
    
    if not source_1.isOpened() or not source_2.isOpened():
        raise Exception("Could not open one or more IP cameras")
    
    while True:
        ret1, frame1 = source_1.read()
        ret2, frame2 = source_2.read()
        
        if not ret1 or not ret2:
            break
        
        frame1 = frame_processor(frame1, detector, recognizer, targets, colors, params)
        frame2 = frame_processor(frame2, detector, recognizer, targets, colors, params)
        
        combined_frame = np.hstack((frame1, frame2))
        cv2.imshow("IP Camera Feed", combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    source_1.release()
    source_2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    main(args)
