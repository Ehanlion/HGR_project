# Ethan Owen, 3/17/2024
# Test program to demo simple functions of huesdk library functions

import huesdk
from huesdk import Discover
from huesdk import Hue

# define the bridge information, local just to my project
bridge_ip = "192.168.1.148"
bridge_username = "ysFHipKKPahizAwVKB8zYJlpPbVc4tyFBLF6MJDg"

try:
    hue = Hue(bridge_ip=bridge_ip, username=bridge_username) # create the hue bridge object
    print(f"Connected to Hue Bridge at IP={bridge_ip} with Username={bridge_username}")
    
    lights = hue.get_lights()
    for light in lights:
        print(f"Object ID: {light.id_}")
        print(f"    Name: {light.name}")
        print(f"    Brightness: {light.bri}")
        print(f"    Hue: {light.hue}")
        print(f"    Saturation: {light.sat}")
    
    # Create all of the lights
    light_corner = hue.get_light(id_=1)
    light_bed = hue.get_light(id_=2)
    light_computer = hue.get_light(id_=3)
    light_bookcase = hue.get_light(id_=4)
    
except Exception as e:
    print(f"Encountered Exception loading Hue Object: {e}")
    exit()