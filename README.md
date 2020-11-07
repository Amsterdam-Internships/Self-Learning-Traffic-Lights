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

Relevance: Improve traffic flow to improve mobility and CO2 emission.

Idea: Multi agent DRL approach with 1 agent per intersection.

Data: Learn from simulating virtual traffic.

Literature: focused on increasing amount of agents, but no consensus about:
- Objective: what contains ‘improving traffic flow’?
- Rewards: what to minimize to obtain objective?
- State: what traffic input is necessary, without being abundant, to minimize rewards?
- Actions: How many options can the agent choose from?

Addition to the field: this project will focus on *revising* the current state of Deep Reinforcement Learning (DRL) for smart Traffic Light Control (TLC).


---

## Project Folder Structure

1) [`src`](./src): Folder containing all the Python code.
1) [`data`](./data): Folder containing the different traffic data sets.
1) [`experiments`](./experiments): Folder containing results of different experiment setups.
1) [`media`](./media): Folder concerning visualisation of the simulation environment.

---

## How it works

The traffic lights of an intersection get controlled by a reinforcement learning agent.
The Deep Q-learning Network (DQN) of the agent gets trained by trial-and-error within a virtual environment, 
which simulates the traffic flow of a given data set of traffic trajectories.

[`dqn_train.py`](./src/dqn_train.py) creates and constantly refreshes the learning data set by running the simulation engine under the current policy of the trained agent.  
[`dqn_agent.py`](./src/dqn_agent.py) learns from the experiences by updating the parameters of its neural network.  
[`dqn_model.py`](./src/dqn_model.py) declares the neural network of dqn_agent.py.

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

To train the reinforcement learning algorithm, run:
[comment]: <> (moet uiteindelijk een trained model inladen en testen)

```
$ python main.py
```

---

## Acknowledgements

Sample code has been used of the Traffic Signal Control Competition [TSCC2019](https://github.com/tianrang-intelligence/TSCC2019) and https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda.
(Add links and check whether the authors have explicitly stated citation preference for using the DOI or citing a paper or so.)
