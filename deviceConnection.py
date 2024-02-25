# Ethan Owen, 2/22/2024
# This program is a test bed for using the huesdk package to control a hue bridge

from huesdk import Discover
from huesdk import Hue

disco = Discover()
print(disco.find_hue_bridge())