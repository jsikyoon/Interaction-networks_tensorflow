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

![alt tag](https://github.com/jaesik817/Interaction-networks_tensorflow/blob/master/figures/interaction_net_2.PNG)

N-body Simulation
--------------------

`
python interaction_network.py
`

### Data
Data are gathered from implemented physics engine (physics_engine.py), which by using given the number of objects initializes and processes random weights[0.2-9kg], distance[10-100m] and angle[0-360] same as the paper settings.
Orbit system is implemented 0-th really big object in central (100kg, same to paper), and velocities of other objects are initialized as supporting stable status.
Object state values of non-orbit system are randomly initialzed as above.

Before traing that, the data (not label!) is initialized as bellowed (same to the paper).
median value is going to 0, and 5% and 95% data are going to -1 and 1, respectively.
This initialization is processed differently for mass, positions and velocities.

### Settings
Almost settings are same to the paper, which is clearly written for that except bellowed. 

For state, there no description, and I used 5-D vector [mass,x_position,y_position,x_velocity,y_velocity].
For R_a and X, there are no description for n-body test, thus I used zero-array with D_R=1, D_X=1.
For D_P, descriptions in page 6 are different, (for f_O, D_P=2, and for \phi_A, D_P=10) in my implementation, I used D_P=2 (x and y velocities).
(I checked that with 1st Author of this paper, Peter W. Battaglia, D_P is used as 10 for estimating potiential energy, and 2 for estimating next state.)
For b_k, descriptions in implementation is to concatenate OR_r and OR_s, however in model architecture, that is described to difference vector between them. In my implementation, I used difference vector.
For input of function a, descriptions in implementation is to object matrix O, however in model architecture, that is described to just use velocities as input. In my implementation, I just used velocities as input.

Except above three things, other settings are same to paper as followed.

For \phi_R function, 4 hidden layers MLP with 150 nodes, ReLU activation function for hidden layers, linear ones for output layer and D_E=50 settings are used.
For \phi_O function, 1 hidden layers MLP with 100 nodes, ReLU for hidden layer, linear for output layer and D_P=2 settings are used.
Adam Optimizer is used with 0.001 learning rate.
L2-regularization is used for matrix E and all parameters (lambda values for each regularization are 0.001).

I generated 10 samples, which have 1000 frames, and 90%/10% data are used for training and validation. For qualititive measure, I newly generated 1 sample and made video files for preidiction from model and true ones.

I did not use Random noise when traning and balancing in batch.

### Results

![alt tag](https://github.com/jaesik817/Interaction-networks_tensorflow/blob/master/figures/gravity_object_2.PNG)

The experiments are 2-object and 3-object ones.

Above MSE graph is 2-object experiment one.

The validation velocities MSE has been saturated to about 0.2~0.3 and 10 for 2-object and 3-object, and video generated new data (not training ones!) is quitely good.
