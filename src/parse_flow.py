import json
import argparse
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random

path = "data/6.0.real_1x1_straight/train/flow.json"
# update the config file with arguments
with open(path) as json_file:
    flow_file = json.load(json_file)

print(len(flow_file))
new_flow_file = []

for car in flow_file:
    if car["route"] == ['road_0_1_0', 'road_1_1_0'] or car["route"] == ["road_2_1_2", "road_1_1_2"] \
            or car["route"] == ["road_1_0_1", "road_1_1_1"] or car["route"] == ["road_1_2_3", "road_1_1_3"]:
        new_flow_file.append(car)

print(len(new_flow_file))

# write to file so the engine can open it.
with open("data/6.0.real_1x1_straight/train/flow.json", 'w') as outfile:
    json.dump(new_flow_file, outfile)



