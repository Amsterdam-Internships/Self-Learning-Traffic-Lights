# Self-Learning-Traffic-Lights
Sierk Kanis' master thesis Artificial Intelligence at the University of Amsterdam as intern at the municipality of Amsterdam. 

![](media/traffic.png)

The aim of this project is to improve the traffic flow of Amsterdam to increase its mobility and decrease its CO2 emission. One way to increase mobility is by decreasing the waiting time of traffic in front of traffic lights. 
Sensor data can be used to simulate the traffic within a virtual simulation environment (e.g. Simulation of Urban MObility (https://www.eclipse.org/sumo/index.html)),
by which the configuration of traffic lights can be optimised by Reinforcement Learning techniques. 
If the performance of the intelligently trained traffic lights surpasses a certain baseline, they can be applied and evaluated in real-life, 
to subsequently be trained with the newly created data.

---

## Project Folder Structure

1) [`src`](./src): Folder containing all the Python code.
1) [`data`](./data): Folder containing the different data sets.
1) [`experiments`](./experiments): Folder containing results of different experiment setups.
1) [`media`](./media): Folder concerning visualisation of the simulation environment.

---

## Installation Guide

Follow these instructions to set up the environment to run the code.

1) Clone this repository:
    ```bash
    git clone https://github.com/Amsterdam-Internships/Self-Learning-Traffic-Lights
    ```

2) Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3) For MacOS: make sure XCode is installed. For Linux: install cpp dependencies with sudo apt update && sudo apt install -y build-essential cmake

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

To train the reinforcement learning algorithm:

```
$ python main.py
```

---

## How it works

The training of the reinforcement learning algorithm calls a Deep Q-learning agent, which in turn calls a Deep Q-learning model.

---

## Acknowledgements

Sample code used of the Traffic Signal Control Competition [TSCC2019](https://github.com/tianrang-intelligence/TSCC2019) and code for DQN in pytorch used from https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda.
(Add links and check whether the authors have explicitly stated citation preference for using the DOI or citing a paper or so.)
