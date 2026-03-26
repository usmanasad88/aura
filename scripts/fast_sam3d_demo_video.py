import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together

def get_args():
    parser = argparse.ArgumentParser(description="SAM 3D Body Demo on Video")
    parser.add_argument("--video_path", type=str, required=True, help="Input video path")
    parser.add_argument("--output_path", type=str, default="./output_video.mp4", help="Output video path")
    parser.add_argument("--model", type=str, default="facebook/sam-3d-body-dinov3")
    parser.add_argument("--detector", type=str, default="yolo_pose")
    parser.add_argument("--detector_model", type=str, default="./checkpoints/yolo/yolo11m-pose.engine")
    parser.add_argument("--local_checkpoint", type=str, default="./checkpoints/sam-3d-body-dinov3")
    return parser.parse_args()

def main():
    args = get_args()
    
    print(f"Loading model... from {args.local_checkpoint}")
    
    estimator = setup_sam_3d_body(
        hf_repo_id=args.model,
        detector_name=args.detector,
        detector_model=args.detector_model,
        local_checkpoint_path=args.local_checkpoint,
    )
    
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error opening video {args.video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Let's read first frame to configure VideoWriter
    ret, frame_bgr = cap.read()
    if not ret:
        print("Empty video")
        return
        
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # The new original Fast-SAM-3D-Body script passes use_yolo_pose_for_hands via kwargs if hand_box_source is yolo_pose, maybe we don't need it.
    
    outputs = estimator.process_one_image(frame_rgb)
    
    rend_img = visualize_sample_together(
        frame_bgr,
        outputs,
        estimator.faces,
         
         
         
    )
    
    out_height, out_width = rend_img.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (out_width, out_height))
    out.write(rend_img)
    
    print(f"Processing video {args.video_path} into {args.output_path} ({total_frames} frames @ {fps} fps)")
    for i in tqdm(range(1, total_frames)):
        ret, frame_bgr = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            outputs = estimator.process_one_image(frame_rgb)
            rend_img = visualize_sample_together(
                frame_bgr,
                outputs,
                estimator.faces,
                 
                 
                 
            )
            
            if rend_img.shape[:2] != (out_height, out_width):
                rend_img = cv2.resize(rend_img, (out_width, out_height))
                
            out.write(rend_img)
        except Exception as e:
            print(f"Frame {i} error: {e}")
            out.write(cv2.resize(frame_bgr, (out_width, out_height)))
            
    cap.release()
    out.release()
    print("Done!")

if __name__ == "__main__":
    main()
