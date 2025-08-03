import Picamera2, Preview

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.set_controls({"Rotation": 180})
picam2.start()
input("Press Enter to quit...")
picam2.stop()