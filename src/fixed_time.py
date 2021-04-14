from src.cityflow_env import CityFlowEnv
from src.evaluate import *
from src.utility import *

args = parse_arguments()


def run_fixed_time(config):
    config['acyclic'] = 0  # You can set variables not dependent on the engine like this, but otherwise use # utility.
    env = CityFlowEnv(config)

    # for t in range(config['num_step']):
    #     env.eng.next_step()
    # print(env.get_average_travel_time())

    phi = 20
    yellow_light_counters = [0 for i in config['intersection_indices']]
    for t in range(config['num_step']):
        for i, intersection_index in enumerate(env.intersection_indices):
            if yellow_light_counters[i] == 0:
                if env.current_phase_times[intersection_index] < phi:
                    env.step(0, intersection_index)  # stay on current light
                else:
                    env.step(-1, intersection_index)
                    yellow_light_counters[i] += 1
            elif 0 < yellow_light_counters[i] < env.yellow_time:
                env.step(-1, intersection_index)
                yellow_light_counters[i] += 1
            elif yellow_light_counters[i] == env.yellow_time:
                # Switch to next light.
                env.step(1, intersection_index)
                yellow_light_counters[i] = 0

    # for t in range(config['num_step']):
    #     # Keep light for phi seconds.
    #     for _ in range(phi):
    #         env.step(0)  # stay on current light
    #         t += 1
    #         flag = (t >= config['num_step'])
    #         if flag:
    #             break
    #     if flag:
    #         break
    #     # Add required yellow time.
    #     for _ in range(env.yellow_time):
    #         env.step(-1)
    #         t += 1
    #         flag = (t >= config['num_step'])
    #         if flag:
    #             break
    #     if flag:
    #         break
    #     # Switch to next light.
    #     env.step(1)

    env.log()
    evaluate_one_traffic(config, 'print')
