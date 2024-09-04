# Cartpole-V1 consistent 500 score in under 6000 frames

This is a model-based Double-DQN policy. The architecture is show below:


It is commonly thought that reinforcement learning methods (particularly DQN) need millions of training samples to reach optimality.
In the case of cartpole-v1 the number of states `4^n`, where `N` is the amount of segements the number line is broken into into. However, the transitions between states are relatively continuous and therefore should easily be graphed. In-fact cartpole v1 has been solved in 0-shot learning with other algorithms.