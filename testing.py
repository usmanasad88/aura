import torch
#################################### For Image ####################################
# from PIL import Image
# from sam3.model_builder import build_sam3_image_model
# from sam3.model.sam3_image_processor import Sam3Processor
# # Load the model
# model = build_sam3_image_model()
# processor = Sam3Processor(model)
# # Load an image
# image = Image.open("/home/mani/Repos/sam3/exp1.png").convert("RGB")
# inference_state = processor.set_image(image)
# # Prompt the model with text
# output = processor.set_text_prompt(state=inference_state, prompt="chair")

# # Get the masks, bounding boxes, and scores
# masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
# print("Image inference successful!")
# print(f"Masks shape: {masks.shape}")

# # Visualize the results
# import matplotlib.pyplot as plt
# from sam3.visualization_utils import plot_results
# plot_results(image, output)
# plt.savefig("output.png")
# print("Visualization saved to output.png")

#################################### For Video ####################################

# from sam3.model_builder import build_sam3_video_predictor

# video_predictor = build_sam3_video_predictor()
# video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# # Start a session
# response = video_predictor.handle_request(
#     request=dict(
#         type="start_session",
#         resource_path=video_path,
#     )
# )
# response = video_predictor.handle_request(
#     request=dict(
#         type="add_prompt",
#         session_id=response["session_id"],
#         frame_index=0, # Arbitrary frame index
#         text="<YOUR_TEXT_PROMPT>",
#     )
# )
# output = response["outputs"]

#################################### For Realtime Inference ####################################

import cv2
import numpy as np
from PIL import Image
import time
import sys
import base64
import json
import re
from io import BytesIO

# Set environment before importing torch
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

print("Python:", sys.executable)
print("OpenCV version:", cv2.__version__)

# Import Gemini API
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    print("Warning: google-genai not installed. Install with: pip install google-genai")
    GEMINI_AVAILABLE = False

# Import torch and model after environment setup
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model
print("Loading SAM3 model...")
model = build_sam3_image_model()
processor = Sam3Processor(model)
print("Model loaded successfully!")


def detect_objects_with_gemini(frame_rgb: np.ndarray, max_objects: int = 5) -> list:
    """
    Use Gemini to detect important objects in a frame.
    
    Args:
        frame_rgb: RGB numpy array of the frame
        max_objects: Maximum number of objects to detect
        
    Returns:
        List of object names to track
    """
    if not GEMINI_AVAILABLE:
        print("Gemini not available, using default objects")
        return ["person", "hand"]
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not set, using default objects")
        return ["person", "hand"]
    
    # Convert frame to base64 JPEG
    pil_image = Image.fromarray(frame_rgb)
    
    # Resize if too large
    max_dim = 640
    if max(pil_image.size) > max_dim:
        ratio = max_dim / max(pil_image.size)
        new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85)
    image_bytes = buffer.getvalue()
    
    # Query Gemini
    client = genai.Client(api_key=api_key)
    
    prompt = f"""Analyze this image and list the {max_objects} most important/prominent objects that would be good to track in real-time.

Focus on:
- Objects that are clearly visible
- Objects that might move or be interacted with
- Focus on rigid objects which can be 6DOF pose tracked.

Respond with ONLY a JSON array of object names, like:
["laptop", "phone", "coffee mug", "chair"]

Keep object names simple and singular (e.g., "hand" not "hands", "person" not "people").
"""

#- People, body parts (hands, face), furniture, devices, etc.
    
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(mime_type="image/jpeg", data=image_bytes)
    ]
    
    contents = [types.Content(role="user", parts=parts)]
    
    generate_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        temperature=0.3,
    )
    
    print("🔍 Querying Gemini to detect objects in first frame...")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=generate_config,
        )
        
        response_text = response.text
        print(f"Gemini response: {response_text}")
        
        # Parse JSON array from response
        # Try to find JSON array in the response
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            objects = json.loads(json_match.group())
            print(f"✅ Detected objects: {objects}")
            return objects[:max_objects]
        else:
            print("Failed to parse Gemini response, using defaults")
            return ["person", "hand"]
            
    except Exception as e:
        print(f"Gemini query failed: {e}, using defaults")
        return ["person", "hand"]


# Text prompts will be detected from first frame
TEXT_PROMPTS = []  # Will be populated after first frame

# Colors for different object classes (BGR format for OpenCV)
COLORS = [
    (0, 255, 0),    # Green for person
    (255, 0, 0),    # Blue for headphones
    (0, 0, 255),    # Red for chair
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
]

# Initialize webcam (0 = default camera, or use video file path)
print("Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Test read a frame first
ret, test_frame = cap.read()
print(f"Test frame read: {ret}, shape: {test_frame.shape if ret else 'None'}")

# Detect objects from the first frame using Gemini
if ret and test_frame is not None:
    first_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
    TEXT_PROMPTS = detect_objects_with_gemini(first_frame_rgb, max_objects=5)
else:
    print("Could not read first frame, using default objects")
    TEXT_PROMPTS = ["phone"]

# Use headless mode by default, set USE_DISPLAY=True to try cv2.imshow
USE_DISPLAY = os.environ.get("USE_DISPLAY", "0") == "1"
MAX_FRAMES = 50  # Limit frames in headless mode

if not USE_DISPLAY:
    print(f"Running in headless mode (saving frames). Set USE_DISPLAY=1 env var to enable display.")
    print(f"Will process {MAX_FRAMES} frames and save output video.")
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('realtime_output.mp4', fourcc, 10.0, (640, 480))

print("Starting realtime inference...")
print(f"Detecting: {', '.join(TEXT_PROMPTS)}")
if USE_DISPLAY:
    print("Press 'q' to quit.")

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        height, width = frame.shape[:2]

        # Run inference for each text prompt
        all_detections = {}  # Store detections per class
        total_detections = 0
        
        with torch.no_grad():
            inference_state = processor.set_image(image)
            
            for prompt_idx, prompt in enumerate(TEXT_PROMPTS):
                # Reset prompts before each new text query
                processor.reset_all_prompts(inference_state)
                output = processor.set_text_prompt(state=inference_state, prompt=prompt)
                
                color = COLORS[prompt_idx % len(COLORS)]
                
                if output["boxes"] is not None and len(output["boxes"]) > 0:
                    boxes = output["boxes"].cpu().numpy()
                    scores = output["scores"].cpu().numpy()
                    masks = output["masks"].cpu().numpy() if output["masks"] is not None else None
                    
                    num_detections = len(boxes)
                    total_detections += num_detections
                    
                    # Store detection info
                    all_detections[prompt] = {
                        "count": num_detections,
                        "boxes": boxes,
                        "scores": scores,
                        "masks": masks
                    }
                    
                    for i, (box, score) in enumerate(zip(boxes, scores)):
                        # Box coordinates are already in pixel coordinates from the processor
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Clamp to image bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)

                        # Draw bounding box
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label with score
                        label = f"{prompt}: {score:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        # cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                        # cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # Overlay mask if available
                        if masks is not None and i < len(masks):
                            mask = masks[i]
                            if mask.ndim == 3:
                                mask = mask[0]  # Take first channel if multi-channel
                            mask_binary = (mask > 0.5).astype(np.uint8)
                            colored_mask = np.zeros_like(frame)
                            colored_mask[:, :, 0] = mask_binary * (color[0] // 2)
                            colored_mask[:, :, 1] = mask_binary * (color[1] // 2)
                            colored_mask[:, :, 2] = mask_binary * (color[2] // 2)
                            frame = cv2.addWeighted(frame, 1, colored_mask, 0.4, 0)
                else:
                    all_detections[prompt] = {"count": 0, "boxes": None, "scores": None, "masks": None}

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Display info on frame
        info_text = f"FPS: {fps:.1f} | Total: {total_detections}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display per-class counts
        y_offset = 60
        for prompt_idx, prompt in enumerate(TEXT_PROMPTS):
            color = COLORS[prompt_idx % len(COLORS)]
            count = all_detections.get(prompt, {}).get("count", 0)
            cv2.putText(frame, f"{prompt}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 25
        
        # Print detailed progress every 10 frames
        if frame_count % 10 == 0:
            print(f"\n=== Frame {frame_count} | {fps:.1f} FPS ===")
            for prompt in TEXT_PROMPTS:
                det = all_detections.get(prompt, {})
                count = det.get("count", 0)
                print(f"  {prompt}: {count} detection(s)")
                if count > 0 and det.get("boxes") is not None:
                    for i, (box, score) in enumerate(zip(det["boxes"], det["scores"])):
                        x1, y1, x2, y2 = box.astype(int)
                        print(f"    [{i+1}] bbox: ({x1}, {y1}, {x2}, {y2}), score: {score:.3f}")
                        if det["masks"] is not None and i < len(det["masks"]):
                            mask = det["masks"][i]
                            if mask.ndim == 3:
                                mask = mask[0]
                            mask_area = (mask > 0.5).sum()
                            print(f"        mask area: {mask_area} pixels")

        if USE_DISPLAY:
            # Show the frame
            cv2.imshow("SAM3 Realtime Inference", frame)
            # Break loop on 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            # Write frame to video
            out.write(frame)
            if frame_count >= MAX_FRAMES:
                print(f"\nReached {MAX_FRAMES} frames limit.")
                break

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
cap.release()
if USE_DISPLAY:
    cv2.destroyAllWindows()
else:
    out.release()
    print(f"Output saved to realtime_output.mp4")
    
print(f"Realtime inference stopped. Processed {frame_count} frames in {elapsed_time:.1f}s ({fps:.1f} FPS average)")