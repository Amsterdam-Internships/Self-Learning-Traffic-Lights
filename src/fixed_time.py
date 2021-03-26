from src.cityflow_env import CityFlowEnv
from src.evaluate import *
from src.utility import *

args = parse_arguments()


def run_fixed_time(config):
    config['acyclic'] = 0
    env = CityFlowEnv(config)
    phi = 20

    for t in range(config['num_step']):
        # Keep light for phi seconds.
        for _ in range(phi):
            env.step(0)  # stay on current light
            t += 1
            flag = (t >= config['num_step'])
            if flag:
                break
        if flag:
            break
        # Add required yellow time.
        for _ in range(env.yellow_time):
            env.step(-1)
            t += 1
            flag = (t >= config['num_step'])
            if flag:
                break
        if flag:
            break
        # Switch to next light.
        env.step(1)

    env.log()
    evaluate_one_traffic(config, 'print')
