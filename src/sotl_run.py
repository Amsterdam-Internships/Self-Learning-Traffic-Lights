from src.cityflow_env import CityFlowEnv
from src.evaluate import *
from src.utility import *

"""
This file contains the implementation of the rule-based Self-organizing Traffic Lights platoon method, 
as described in Gershenson (2005), adjusted for multi-phase intersections.

Source:
https://github.com/tianrang-intelligence/TSCC2019
Self-organizing traffic lights: A realistic simulation (Cools, S et al., 2013)
"""

PHI = 20
THETA = 200
MU = 20
# values have been raised, because the cost of switch is even higher I think, because it takes longer to return to the ligh
# because other lights will get in between. Also, just tested a bit. It also changes because mu is a little different.

args = parse_arguments()


def run_sotl(config):
    """ SOTL
    """
    env = CityFlowEnv(config)

    lane_phase_info = config['lane_phase_info']
    intersection_id = list(lane_phase_info.keys())[0]
    # CHANGE when straight
    phase_list = lane_phase_info[intersection_id]["phase"]
    # phase_list = phase_list[:2]  # when straight only
    lane_list = lane_phase_info[intersection_id]['start_lane']
    phase_startLane_mapping = lane_phase_info[intersection_id]["phase_startLane_mapping"]  # from 1 to len(phase_list)
    intersection_id = list(lane_phase_info.keys())[0]
    start_lane = lane_phase_info[intersection_id]['start_lane']

    t = 0
    env.reset()
    kappa = np.zeros(len(lane_list))
    state = env.get_state_sotl()
    action = choose_action(state, kappa, phase_list, phase_startLane_mapping)
    last_action = action

    while t < config['num_step']:
        state = env.get_state_sotl()

        # # Calculate kappa
        # all_green_lanes_per_phase = [phase_startLane_mapping[j] for j in phase_list]
        # for i, phase in enumerate(all_green_lanes_per_phase):
        #     current_phase_lanes = phase_startLane_mapping[state["current_phase"] + 1]
        #     kappa[i] += sum([state["lane_vehicle_count"][k] for k in phase if k not in current_phase_lanes])
        # kappa[state["current_phase"]] = 0

        # Update kappa.
        lane_vehicle_count = [state['lane_vehicle_count'][lane] for lane in state["start_lane"]]
        kappa += lane_vehicle_count

        current_phase_lanes = phase_startLane_mapping[state["current_phase"] + 1]
        for i, lane in enumerate(state["start_lane"]):
            for lane2 in current_phase_lanes:
                if lane == lane2:
                    kappa[i] = 0

        action = choose_action(state, kappa, phase_list, phase_startLane_mapping)

        if action == last_action:
            env.step(action)
        else:
            for _ in range(env.yellow_time):
                # Continue calculating all kappa's during red light.
                lane_vehicle_count = [state['lane_vehicle_count'][lane] for lane in state["start_lane"]]
                kappa += lane_vehicle_count

                env.step(-1)  # required yellow time
                t += 1
                flag = (t >= config['num_step'])
                if flag:
                    break
            if flag:
                break
            env.step(action)

            # Set kappa of green light to zero.
            current_phase_lanes = phase_startLane_mapping[state["current_phase"] + 1]
            for i, lane in enumerate(state["start_lane"]):
                for lane2 in current_phase_lanes:
                    if lane == lane2:
                        kappa[i] = 0

        last_action = action
        t += 1
    env.log()
    evaluate_one_traffic(config, args.scenario, 'sotl', 'print')


def choose_action(state, kappa, phase_list, phase_startLane_mapping):
    """
    Choose action based on maximum rho, if condition is met.

    Params
    ======
    state (array_like): current state
    kappa (array_like): waiting time per start lane
    """
    action = state["current_phase"]  # if no action is selected, the current phase remains active.

    rho = np.zeros(len(phase_list))  # cumulative waiting time per phase.
    for i, phase_lanes in enumerate([phase_startLane_mapping[j] for j in phase_list]):
        for j, start_lanes in enumerate(state["start_lane"]):
            for lane in phase_lanes:
                if lane == start_lanes:
                    rho[i] += kappa[j]

    # lane_waiting_vehicle_count is used as approximation for lane_vehicle_count < omega,
    # because vehicles with speed less than 0.1m/s are considered as waiting in CityFlow.
    cars_approaching_green = sum([state["lane_waiting_vehicle_count"][i]
                                  for i in phase_startLane_mapping[state["current_phase"] + 1]])

    if state["current_phase_time"] >= PHI:
        if not 0 < cars_approaching_green < MU:
            if rho.max() > THETA:
                action = rho.argmax()
                # # if state["current_phase"] == 1:  # use if only 2 actions (because it goes through full action cycle)
                # # if cur_phase == len(phase_list) - 1:
                #     action = 0
                # else:
                #     action = 1
    return action
