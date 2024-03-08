from huesdk import Discover
from huesdk import Hue

# Discovery code, slotted into a try-except clause:
try:
    discover = Discover()
    print(discover.find_hue_bridge() )
    print(Hue.connect(bridge_ip="192.168.1.148"))
except Exception as e:
    print(f'Exception encountered: {e}')

# These are the values I found for my bridge
bridge_ip = "192.168.1.148"
bridge_username = "ysFHipKKPahizAwVKB8zYJlpPbVc4tyFBLF6MJDg"