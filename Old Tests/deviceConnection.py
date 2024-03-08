# Ethan Owen, 2/22/2024
# This program is a test bed for using the huesdk package to control a hue bridge

from huesdk import Discover
from huesdk import Hue
import time

# These are the values I found for my bridge
bridge_ip = "192.168.1.148"
bridge_username = "ysFHipKKPahizAwVKB8zYJlpPbVc4tyFBLF6MJDg"

hue = Hue(bridge_ip=bridge_ip, username=bridge_username)

# Get light data:
lights = hue.get_lights()

# Print light properties
for light in lights:
    print(light.id_)
    print(light.name)
    print(light.is_on)
    print(light.bri)
    print(light.hue)
    print(light.sat)
    
er_cornerLamp = hue.get_light(id_=1) # gets the first light
er_bedLamp = hue.get_light(id_=2) # gets the second light

# testing:
er_cornerLamp.off()
er_bedLamp.off()
time.sleep(3)
er_cornerLamp.on()
er_bedLamp.on()
time.sleep(3)