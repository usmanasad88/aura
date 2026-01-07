#!/usr/bin/env python
"""Test script for sound monitor with Gemini Live."""

import asyncio
import argparse
import logging

from aura.monitors.sound_monitor import SoundMonitor, SoundConfig

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_basic_conversation(input_device=None, output_device=None, sample_rate=16000):
    """Test basic speech conversation."""
    print("=" * 50)
    print("Sound Monitor Test - Basic Conversation")
    print("=" * 50)
    print(f"Input: {input_device or 'Default'}, Output: {output_device or 'Default'}, Rate: {sample_rate}")
    print("\nSpeak into your microphone. Type 'q' to quit.")
    print("Type a message to send text to Gemini.\n")
    
    config = SoundConfig(
        enable_speech_output=True,
        keywords_of_interest=["robot", "move", "pick", "place", "stop", "help"],
        input_device_name=input_device,
        output_device_name=output_device,
        input_sample_rate=sample_rate
    )
    
    received_texts = []
    
    def on_response(text: str):
        print(f"\n[Gemini]: {text}")
        received_texts.append(text)
    
    monitor = SoundMonitor(config=config, on_response=on_response)
    
    try:
        await monitor.start_listening()
        
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == 'q':
                break
            await monitor.send_text(text)
            
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        await monitor.stop_listening()
        
    print(f"\nSession ended. Received {len(received_texts)} responses.")


async def test_command_detection():
    """Test command/keyword detection."""
    print("=" * 50)
    print("Sound Monitor Test - Command Detection")
    print("=" * 50)
    print("\nSay commands with keywords: robot, move, pick, place, stop")
    print("Press Ctrl+C to stop.\n")
    
    config = SoundConfig(
        enable_speech_output=False,  # Text only
        keywords_of_interest=["robot", "move", "pick", "place", "stop"],
        system_instruction="""You are a robot command interpreter. 
When the user speaks, extract any robot commands and respond with:
- COMMAND: <action> <object> <location> if you detect a command
- UNCLEAR: <what's unclear> if the command is ambiguous
- NONE: if no command was given

Examples:
User: "Robot, pick up the red cup"
Response: COMMAND: pick red_cup none

User: "Move to the table"  
Response: COMMAND: move none table

User: "Hello, how are you?"
Response: NONE: greeting only
"""
    )
    
    commands_detected = []
    
    def on_response(text: str):
        print(f"[Interpreter]: {text}")
    
    monitor = SoundMonitor(config=config, on_response=on_response)
    
    try:
        await monitor.start_listening()
        
        # Monitor commands for 60 seconds
        print("Listening for 60 seconds...")
        for i in range(60):
            await asyncio.sleep(1)
            new_commands = monitor.get_recent_commands()
            # This is a bit simplistic, just looking at what's in buffer
            # Ideally we'd consume them
            pass
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await monitor.stop_listening()
    
    # Check what we found
    commands_detected = monitor.get_recent_commands()
    print(f"\nDetected {len(commands_detected)} commands:")
    for cmd in commands_detected:
        print(f"  - {cmd.text}")


async def test_multimodal():
    """Test multimodal (audio + image) input."""
    import cv2
    from io import BytesIO
    from PIL import Image
    
    print("=" * 50)
    print("Sound Monitor Test - Multimodal (Audio + Vision)")
    print("=" * 50)
    print("\nWill capture webcam frames and send to Gemini.")
    print("Speak to describe what you see or give commands.")
    print("Press 'q' in the video window to quit.\n")
    
    config = SoundConfig(
        enable_speech_output=True,
        system_instruction="""You are assisting a user with a robotic task.
You can see images from their workspace. When they speak:
1. Describe what you see if asked
2. Identify objects they mention
3. Suggest actions the robot could take
Keep responses brief and helpful."""
    )
    
    monitor = SoundMonitor(config=config)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    try:
        await monitor.start_listening()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            cv2.imshow('Camera', frame)
            
            # Send frame to Gemini (every ~1 sec)
            # In a real app we'd pace this better
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail([512, 512]) # Resize for bandwidth
            
            image_io = BytesIO()
            img.save(image_io, format="jpeg")
            image_bytes = image_io.getvalue()
            
            await monitor.send_image(image_bytes)
            
            # Check for quit
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await monitor.stop_listening()
        cap.release()
        cv2.destroyAllWindows()


async def test_tools_brain(input_device=None, output_device=None, sample_rate=16000):
    """Test interaction with a mock 'Brain' using tools."""
    print("=" * 50)
    print("Sound Monitor Test - Brain Tools")
    print("=" * 50)
    print(f"Input: {input_device or 'Default'}, Output: {output_device or 'Default'}, Rate: {sample_rate}")
    
    # Mock brain functions
    async def stop_robot():
        print("\n\n!! BRAIN: EXECUTING EMERGENCY STOP !!\n")
        return {"status": "success", "message": "Emergency stop triggered. Robot is halted."}
    
    async def get_robot_status():
        print("\n!! BRAIN: Checking status !!\n")
        return {"status": "idle", "position": [0.5, 0.2, 0.1]}
    
    # Tool definitions
    from google.genai import types
    tools = [
        types.FunctionDeclaration(
            name="stop_robot",
            description="Emergency stop the robot immediately.",
        ),
        types.FunctionDeclaration(
            name="get_robot_status",
            description="Get the current status and position of the robot."
        )
    ]
    
    tool_handlers = {
        "stop_robot": stop_robot,
        "get_robot_status": get_robot_status
    }

    config = SoundConfig(
        enable_speech_output=True,
        input_device_name=input_device,
        output_device_name=output_device,
        input_sample_rate=sample_rate,
        system_instruction="""You are the voice interface for a robotic assistant. 
You are connected to a 'Brain' that controls the robot. 
If the user asks to stop, halt, or kill the robot, you MUST call the stop_robot tool.
If the user asks for status, call get_robot_status.
Always wait for the tool outcome before confirming to the user.""",
        tools=tools
    )
    
    def on_response(text: str):
        print(f"\n[Gemini]: {text}")
        
    monitor = SoundMonitor(config=config, on_response=on_response, tool_handlers=tool_handlers)
    
    try:
        await monitor.start_listening()
        
        print("\nSay 'Stop the robot' or 'What is your status?'")
        print("Type 'q' to quit (or wait 60s).")
        
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == 'q':
                break
            await monitor.send_text(text)
            
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        await monitor.stop_listening()

def main():
    parser = argparse.ArgumentParser(description="Test sound monitor")
    parser.add_argument("--mode", type=str, default="conversation",
                       choices=["conversation", "commands", "multimodal", "tools"],
                       help="Test mode")
    parser.add_argument("--input-device", type=str, default=None, help="Substring of input device name")
    parser.add_argument("--output-device", type=str, default=None, help="Substring of output device name")
    parser.add_argument("--rate", type=int, default=16000, help="Input sample rate")
    
    args = parser.parse_args()
    
    if args.mode == "conversation":
        asyncio.run(test_basic_conversation(args.input_device, args.output_device, args.rate))
    elif args.mode == "commands":
        asyncio.run(test_command_detection())
    elif args.mode == "multimodal":
        asyncio.run(test_multimodal())
    elif args.mode == "tools":
        asyncio.run(test_tools_brain(args.input_device, args.output_device, args.rate))



if __name__ == "__main__":
    main()
