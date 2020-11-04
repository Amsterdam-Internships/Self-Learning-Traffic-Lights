from code.cityflow_env import CityFlowEnv
from code.dqn_agent_tscc import DQNAgent
from code.utility import parse_roadnet
import numpy as np
import os
from code.utility import parse_arguments
import json
from code.constants import ROOT_DIR

if __name__ == "__main__":

    os.chdir(ROOT_DIR)

    args = parse_arguments()
    with open('code/config.json') as json_file:
        config = json.load(json_file)

    config['lane_phase_info'] = parse_roadnet("data/{}/roadnettest.json".format(args.scenario))
    intersection_id = list(config['lane_phase_info'].keys())[0]
    phase_list = config['lane_phase_info'][intersection_id]['phase']
    config['state_size'] = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1
    config['action_size'] = len(phase_list)
    config['num_step'] = 300

    # # add visible gpu if necessary
    # os.environ["CUDA_VISIBLE_DEVICES"] = ''

    env = CityFlowEnv(config)
    agent = DQNAgent(config)

    # some parameters in dqn
    batch_size = 32
    EPISODES = 100
    learning_start = 300
    update_model_freq = 300
    update_target_model_freq = 1500
    num_step = config['num_step']
    state_size = config['state_size']

    for e in range(EPISODES):
        # reset initially at each episode
        env.reset()
        t = 0
        state = env.get_state()
        state = np.array(list(state['start_lane_vehicle_count'].values()) + [state['current_phase']]) # a sample state definition
        state = np.reshape(state, [1, state_size])
        last_action = phase_list[agent.choose_action(state)]
        while t < num_step:
            action = phase_list[agent.choose_action(state)]
            if action == last_action:
                env.step(action)
            else:
                for _ in range(env.yellow_time):
                    env.step(0)  # required yellow time
                    t += 1
                    flag = (t >= num_step)
                    if flag:
                        break
                if flag:
                    break
                env.step(action)
            last_action = action
            t += 1
            next_state = env.get_state()
            reward = env.get_reward()
            next_state = np.array(list(next_state['start_lane_vehicle_count'].values()) + [next_state['current_phase']])
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state)
            state = next_state

            total_time = t + e * num_step
            if total_time > learning_start and total_time % update_model_freq == update_model_freq - 1:
                agent.replay()
            if total_time > learning_start and total_time % update_target_model_freq == update_target_model_freq -1:
                agent.update_target_network()

            print("episode: {}/{}, time: {}, acton: {}, reward: {}"
                  .format(e, EPISODES, t-1, action, reward))

        # if e % 10 == 0:
        #     if not os.path.exists("model"):
        #         os.makedirs("model")
        #     agent.model.save("model/trafficLight-dqn-{}.h5".format(e))

    # log environment files
    env.log()