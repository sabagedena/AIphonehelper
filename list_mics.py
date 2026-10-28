import pyaudio

p = pyaudio.PyAudio()
default_info = p.get_default_input_device_info()
print("DEFAULT INPUT DEVICE:")
print(f"Index: {default_info['index']}, Name: {default_info['name']}")
print("\nALL AVAILABLE INPUT DEVICES:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('maxInputChannels') > 0:
        print(f"Index {dev['index']}: {dev['name']} (Channels: {dev['maxInputChannels']}, Rate: {dev['defaultSampleRate']})")
p.terminate()
