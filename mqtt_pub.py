# publisher.py
import paho.mqtt.client as mqtt

BROKER_IP = "172.20.10.13"  # IP of Pi #2 (broker)
TOPIC = "servo/control"

client = mqtt.Client()
client.connect(BROKER_IP, 1883, 60)

# Send command
client.publish(TOPIC, "MOVE")
print("Command sent: MOVE")

client.disconnect()
