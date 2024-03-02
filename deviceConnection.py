# Ethan Owen, 2/22/2024
# This program is a test bed for using the huesdk package to control a hue bridge

from huesdk import Discover
from huesdk import Hue

discover = Discover()
# bridge_ip = discover.find_hue_bridge() 
bridge_ip = discover.find_hue_bridge_mdns(timeout=5) # specifically local
username = Hue.connect(bridge_ip=bridge_ip) # get username using ip address
hue = Hue(bridge_ip=bridge_ip, username=username)

print(f'Bridge IP = {bridge_ip}')
print(f'Bridge Username = {username}')

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