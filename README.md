# Self-Learning-Traffic-Lights
Sierk Kanis' master thesis Artificial Intelligence at the University of Amsterdam as intern at the municipality of Amsterdam. The aim of this project is to improve the traffic flow of Amsterdam to increase its mobility and decrease its CO2 emission.

![](media/traffic.png)

## Description
One way to increase mobility is by decreasing the waiting time of traffic in front of traffic lights. 
Sensor data can be used to simulate the traffic within a virtual simulation environment (e.g. Simulation of Urban MObility (https://www.eclipse.org/sumo/index.html)),
by which the configuration of traffic lights can be optimised by Reinforcement Learning techniques. 
If the performance of the intelligently trained traffic lights surpasses a certain baseline, they can be applied and evaluated in real-life, 
to subsequently be trained with the newly created data.

## Project Folder Structure

1) [`src`](./src): Folder containing all the Python code.
1) [`data`](./data): Folder containing the different data sets.
1) [`media`](./media): Folder concerning visualisation of the simulation environment.

## Installation Guide

Explain how to set up everything. 
Let people know if there are weird dependencies - if so feel free to add links to guides and tutorials.

A person should be able to clone this repo, follow your instructions blindly, and still end up with something *fully working*!

1) Clone this repository:
    ```bash
    git clone https://github.com/Amsterdam-Internships/InternshipAmsterdamGeneral
    ```

2) Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
---
