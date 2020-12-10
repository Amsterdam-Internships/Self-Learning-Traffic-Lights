from src.cityflow_env import CityFlowEnv
from src.dqn_agent import *
from src.utility import *
from src.evaluate import evaluate_one_traffic

"""
action 0 is WE on green.
[W, S, N, E]
positive is NS on green, negative WE.
"""

TRAJECTORIES = 3000
NUM_STEPS = 300
TRAINING_RUNS = 1
NORM_TAU = 1e-3
NORM_INPUTS = 0  # Set to 1 to normalize inputs
NORM_REWARDS = 0  # Set to 1 to normalize rewards
LOAD = 0  # Set to 1 to load checkpoint
RANDOM_RUN = 0
TENSORBOARD = 1
# LRS = [1e-2, 1e-3, 1e-4, 1e-5]
LRS = [1e-3]

config = setup_config(NUM_STEPS, 'train', NORM_INPUTS, NORM_REWARDS, NORM_TAU)
intersection_id = list(config['lane_phase_info'].keys())[0]
phase_list = config['lane_phase_info'][intersection_id]['phase']
action_size = 2
# action_size = len(phase_list)
state_size = len(CityFlowEnv(config).reset())
best_travel_time = 100000

args = parse_arguments()


agent = Agent(state_size, action_size, 0, 0.001)
checkpoint = torch.load("trained_models/{}/checkpoint.tar".format(args.exp_name))
agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])

q_val_size = 6

q_value_list = np.zeros((4, q_val_size, q_val_size))
# q_value_list_list = []
counter = 0
fig = plt.figure()
for q_val in range(action_size):
    for light in range(action_size):
        for i in range(q_val_size):
            for j in range(q_val_size):
                phases = np.zeros(action_size)
                phases[light] = 1
                state = [0, 0, j, i]
                combined = np.array(list(state) + list(phases))

                action, q_values = agent.act(combined, 0)
                q_value_list[counter][i][j] = q_values[0][q_val]

        # q_value_list_list.append(q_value_list)
        counter += 1
        fig.add_subplot(2, 3, counter)
        plt.ylabel('Cars WE')
        plt.xlabel('Cars NS')
        plt.title("Phase " + str(light) + ", Q value " + str(q_val))
        plt.imshow(q_value_list[counter-1], cmap='hot', interpolation='nearest')

q_val_dif_1 = np.subtract(q_value_list[2], q_value_list[0])

# for i in range(q_val_size):
#     for j in range(q_val_size):
#         if q_val_dif_1[i][j] < 0:
#             q_val_dif_1[i][j] = 0
#         if q_val_dif_1[i][j] > 0:
#             q_val_dif_1[i][j] = 1

fig.add_subplot(2, 3, 5)
plt.ylabel('Cars NS')
plt.xlabel('Cars WE')
plt.title("Phase 0 which Q val")
plt.imshow(q_val_dif_1, cmap='hot', interpolation='nearest')

q_val_dif_2 = np.subtract(q_value_list[3], q_value_list[1])

# for i in range(q_val_size):
#     for j in range(q_val_size):
#         if q_val_dif_2[i][j] < 0:
#             q_val_dif_2[i][j] = 0
#         if q_val_dif_2[i][j] > 0:
#             q_val_dif_2[i][j] = 1

fig.add_subplot(2, 3, 6)
plt.ylabel('Cars NS')
plt.xlabel('Cars WE')
plt.title("Phase 1 which Q val")
plt.imshow(q_val_dif_2, cmap='hot', interpolation='nearest')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(q_val_dif_1)

# We want to show all ticks...
ax.set_xticks(np.arange(q_val_size))
ax.set_yticks(np.arange(q_val_size))

# Loop over data dimensions and create text annotations.
for i in range(q_val_size):
    for j in range(q_val_size):
        text = ax.text(j, i, round(q_val_dif_1[i, j]),
                       ha="center", va="center", color="w")

ax.set_title("Phase 0: Difference of Q values")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(q_val_dif_2)

# We want to show all ticks...
ax.set_xticks(np.arange(q_val_size))
ax.set_yticks(np.arange(q_val_size))

# Loop over data dimensions and create text annotations.
for i in range(q_val_size):
    for j in range(q_val_size):
        text = ax.text(j, i, round(q_val_dif_2[i, j]),
                       ha="center", va="center", color="w")

ax.set_title("Phase 1: Difference of Q values")
fig.tight_layout()
plt.show()

evaluate_one_traffic(config, args.scenario, 'train', 'print')

