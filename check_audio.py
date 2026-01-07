import pyaudio

p = pyaudio.PyAudio()
print("Available Audio Devices:")
for i in range(p.get_device_count()):
    try:
        dev = p.get_device_info_by_index(i)
        print(f"Index {i}: {dev['name']}")
        print(f"  Max Inputs: {dev['maxInputChannels']}, Max Outputs: {dev['maxOutputChannels']}")
        print(f"  Default Sample Rate: {dev['defaultSampleRate']}")
        
        # Test 16k support if it's an input device
        if dev['maxInputChannels'] > 0:
            try:
                stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, input_device_index=i)
                print(f"  -> Supports 16000Hz capture")
                stream.close()
            except Exception:
                print(f"  -> Does NOT support 16000Hz capture")
    except Exception as e:
        print(f"Error checking device {i}: {e}")

p.terminate()
