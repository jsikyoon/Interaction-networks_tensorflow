Interaction Networks
====================

Tensorflow Implementation of Interaction Networks from Google Deepmind.

Implementation is on Tensorflow r1.2.

https://arxiv.org/abs/1612.00222

"Reasoning about objects, relations, and physics is central to human intelligence, and
a key goal of artificial intelligence. Here we introduce the interaction network, a
model which can reason about how objects in complex systems interact, supporting
dynamical predictions, as well as inferences about the abstract properties of the
system."
From the paper

![alt tag](https://github.com/jaesik817/Interaction-networks_tensorflow/blob/master/figures/interaction_net.PNG)

![alt tag](https://github.com/jaesik817/Interaction-networks_tensorflow/blob/master/figures/interaction_neti_2.PNG)

N-body Simulation
--------------------

`
python interaction_network.py
`

### Data
Data are gathered from implemented physics engine (physics_engine.py), which by using given the number of objects initializes and processes random weights[0.2-9kg], distance[10-100m] and angle[0-360] same as the paper settings. Currently orbit system is not implemented yet, the experiments on here are gone with not orbit system (no central biggest object).

### Settings
Almost settings are same to the paper, which is clearly written for that except bellowed. 

For state, there no description, and I used 5-D vector [mass,x_position,y_position,x_velocity,y_velocity].
For R_a and X, there are no description for n-body test, thus I used zero-array with D_R=1, D_X=1.
For D_P, descriptions in page 6 are different, (for f_O, D_P=2, and for \phi_A, D_P=10) in my implementation, I used D_P=2 (x and y velocities).

Except above three things, other settings are same to paper as followed.

For \phi_R function, 4 hidden layers MLP with 150 nodes, ReLU activation function for hidden layers, linear ones for output layer and D_E=50 settings are used.
For \phi_O function, 1 hidden layers MLP with 100 nodes, ReLU for hidden layer, linear for output layer and D_P=2 settings are used.
Adam Optimizer is used with 0.001 learning rate.
L2-regularization is used for matrix E and all parameters.

### Results
Currently, in progress of training, validation mse is less than 0.05. That is better than the paper results, for which, I assume state or D_P settings are different to original ones.

After finishing training, I will upload exact results with qualititive ones.
