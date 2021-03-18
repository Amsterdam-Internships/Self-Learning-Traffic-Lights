import json
import argparse
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random

SCENARIO = "7.2.hangzou3_1x1_turns"
SCENARIO = "hangzhou_1x1_1h_C2"

PATH = "/Users/sierkkanis/Documents/MscAI/Thesis/Code/Eigen/data/{}/flow.json".format(SCENARIO)
WINDOW_SIZE = 600


def get_inflow_per_lane(path):
    inflow_per_lane = {'road_0_1_0': [], 'road_1_2_3': [], 'road_2_1_2': [], 'road_1_0_1': []}
    # update the config file with arguments
    with open(path) as json_file:
        flow_file = json.load(json_file)
    print("amount of cars: " + str(len(flow_file)))

    for car in flow_file:
        inflow_per_lane[car['route'][0]].append(car['startTime'])

    return inflow_per_lane


def get_inflow_smoothed(inflow):
    smoothed_flow = []
    for t in range(int(WINDOW_SIZE/2), 3600-int(WINDOW_SIZE/2)):
        smoothed_flow.append(len([i for i in inflow if t - WINDOW_SIZE / 2 < i < t + WINDOW_SIZE / 2]))
    return smoothed_flow


def get_hist(inflow):
    histogram = []
    for t in range(0, 3600, WINDOW_SIZE):
        histogram.append(len([i for i in inflow if t < i < t + WINDOW_SIZE]))
    return histogram


inflow_per_lane = get_inflow_per_lane(PATH)
smoothed_flow_per_lane = []
hists = []
inflow_per_lane_list = []

# fig = plt.figure()
# for i, (k, v) in enumerate(inflow_per_lane.items()):
#     fig.add_subplot(2, 2, i + 1)
#     plt.xlabel('Spawn second')
#     plt.ylabel('Car id')
#     plt.title('Road %i' %i)
#     plt.plot(v, range(len(v)), ',')
# fig.tight_layout()

for i, (k, v) in enumerate(inflow_per_lane.items()):
    smoothed_flow_per_lane.append(get_inflow_smoothed(v))
    hists.append(get_hist(v))
    inflow_per_lane_list.append(v)

with open(PATH) as json_file:
    flow_file = json.load(json_file)

fig2 = plt.figure()
fig2.suptitle(f"{SCENARIO}, amount of cars: {len(flow_file)}")
for i in range(len(smoothed_flow_per_lane)):
    fig2.add_subplot(2, 2, i + 1)
    plt.ylabel('Cars per %i minutes' %int(WINDOW_SIZE/60))
    plt.xlabel('Seconds')
    plt.title('Road %i' %i)
    plt.plot(range(int(WINDOW_SIZE/2), 3600-int(WINDOW_SIZE/2)), smoothed_flow_per_lane[i])
fig2.tight_layout()

path = "analysed_data"
if not os.path.exists(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)

path = "{}/{}_moving_window.png".format(path, SCENARIO)
plt.savefig(path)

fig3 = plt.figure()
fig3.suptitle(f"{SCENARIO}, amount of cars: {len(flow_file)}")
for i in range(len(hists)):
    fig3.add_subplot(2, 2, i + 1)
    plt.ylabel('Cumulated Cars')
    plt.xlabel(f"Every {int(WINDOW_SIZE/60)} minutes")
    plt.title('Road %i' %i)
    plt.bar(range(len(hists[i])), hists[i])
fig3.tight_layout()

path = "analysed_data"
path = "{}/{}_barplot.png".format(path, SCENARIO)
plt.savefig(path)
# plt.show()

SCENARIO = "hangzhou_1x1_1h_A1"
PATH = "/Users/sierkkanis/Documents/MscAI/Thesis/Code/Eigen/data/{}/roadnet.json".format(SCENARIO)

with open(PATH) as json_file:
    flow_file = json.load(json_file)

SCENARIO = "hangzhou_1x1_1h_B1"
PATH = "/Users/sierkkanis/Documents/MscAI/Thesis/Code/Eigen/data/{}/roadnet.json".format(SCENARIO)

with open(PATH) as json_file:
    flow_file2 = json.load(json_file)

print(flow_file2 == flow_file)
