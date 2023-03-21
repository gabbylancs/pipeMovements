#  @author: Gabby
#  Created: 23/03/23
#  Last updated: 23/02/2023

# The purpose of this document is to demonstrate the use of different algorithms on pipe images and or
# videos to compare their effectiveness and help decide which will be best used in the project.

import matplotlib.pyplot as plt
import functions
import simulation
import solarSystem

# functions.optical_flow_mod()
solar_system = solarSystem.SolarSystem(200)

body = solarSystem.SolarSystemBody(solar_system, 100, velocity=(1, 1, 1))
for _ in range(100):
    solar_system.update_all()
    solar_system.draw_all()

