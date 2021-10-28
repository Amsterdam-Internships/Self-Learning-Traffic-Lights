# Self-Learning-Traffic-Lights
Sierk Kanis' master thesis Artificial Intelligence at the University of Amsterdam as intern at the municipality of Amsterdam. 

![](media/traffic.png)

The aim of this project is to improve the traffic flow of Amsterdam to increase its mobility and decrease its CO2 emission. One way to realise this is by decreasing the waiting time of traffic in front of traffic lights. 
Sensor data can be used to simulate the traffic within a virtual simulation environment,
by which the configuration of traffic lights can be optimised by Reinforcement Learning techniques. 
If the performance of the intelligently trained traffic lights surpasses a certain baseline, they can be applied and evaluated in real-life, 
to subsequently be trained with the newly created data.

---

## In Short

Relevance: with the recent advances in reinforcement learning and the increasement of traffic data, it is natural to wonder whether traffic networks can be optimized by machine learning to improve mobility and CO2 emission.

Approach: Deep Reinforcement Learning with one agent controling one intersection.

Data: real-world traffic flow data virtually simulated in the CityFlow Simulator.

Contributions: revisiting the fundamental framework of Adaptive Traffic Signal Control for reinforcement learning approaches.
- State: what traffic input is necessary, without being abundant, to maximize rewards?
- Actions: what actions does the agent have at its proposal?
- Markov Decision Process: at what timesteps does the agent perform an action?
- Reward: what constitutes a bad or good action?


---

## Project Folder Structure

1) [`src`](./src): python code.
1) [`data`](./data): traffic flow data sets.
1) [`trained_models`](./trained_models): trained models by Q-learning.
1) [`media`](./media): visualisation of the simulation environment.

---

## Results

![alt text](https://github.com/Amsterdam-Internships/Self-Learning-Traffic-Lights/blob/workinprogress/media/videos/compare.gif)

---

## How it works

The traffic lights of an intersection get controlled by a reinforcement learning agent.
The Deep Q-learning Network (DQN) of the agent gets trained by trial-and-error within a virtual environment, 
which simulates the traffic flow of a given data set of traffic trajectories.

[`dqn_train.py`](./src/dqn_train.py) creates and constantly refreshes the training set by running the simulation engine under the current policy of the trained agent.  
[`dqn_agent.py`](./src/dqn_agent.py) learns from the experiences by updating the parameters of its neural network.  
[`dqn_model.py`](./src/dqn_model.py) declares the neural network of dqn_agent.py.  
[`cityflow_env.py`](./src/cityflow_env.py) declares the state-action-reward representation of the environment the agent is interacting with.  
[`tuning.py`](./src/tuning.py) loops over the given hyperparameters and evaluates the performance with baseline methods.

---

## Installation Guide

### MacOS

Follow these instructions to set up the environment to run the code.

1) Clone this repository:
    ```bash
    git clone https://github.com/Amsterdam-Internships/Self-Learning-Traffic-Lights
    ```

2) Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3) Make sure XCode is installed. (For Linux: install cpp dependencies with sudo apt update && sudo apt install -y build-essential cmake)

4) Clone simulation software CityFlow from github.

```bash
git clone https://github.com/cityflow-project/CityFlow.git
```
5) Go to CityFlow project’s root directory and run
```bash
pip install .
```

---

## Usage

To train the model with, for example, a state representation containing the number of waiting vehicles, approaching vehicles, and their distance, using an acyclic phase order, run:
```bash
python src/tuning.py --waiting --distance --acyclic
```

If you want to try your own state representation, you can update [`cityflow_env.py`](./src/cityflow_env.py).  

If you want to compare your signal plan with ours, you can run evaluate.py on a specific dataset with the path to your signal_plan_template.txt, [`evaluate.py`](./src/evaluate.py). 

---

## Acknowledgements

Sample code has been used of the Traffic Signal Control Competition [TSCC2019](https://github.com/tianrang-intelligence/TSCC2019) and https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda.
