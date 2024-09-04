# Cartpole-V1 consistent max-score in about 10,000 frames

This is a model-based Double-DQN policy. The architecture is show below:

![Architecture](images/model_architecture.png)

It is commonly thought that reinforcement learning methods (particularly DQN) need millions of training samples to reach optimality.
In the case of cartpole-v1 the number of states `4^n`, where `n` is the amount of sub-segements 1 unit of the number line is broken into into. However, the transitions between states are relatively continuous and therefore should easily be graphed. In-fact cartpole v1 has been solved in 0-shot learning with other algorithms.

That said, my model is able to solve the problem with only 10000 frames of training, approximately 3 minutes using my home computer running a RTX 2080 Ti.
This coincides with the hard-update to the target model. Further testing is required in this area.

![Graphs](images/full_screenshot.png)

## Setup

1. Download the code.
2. Install the dependencies.
3. Run:
```python
pip install -r requirements.txt
python main.py
```

The model is currently in demonstration mode. Have a look through the config file.

If you want to train a new model you can change some lines:
```python
# Path to the model file to load. Alternatively, "" to start with a fresh model.
model_to_load = "" #"models/actor_model_23500.pth"
step = 0 # This is here because it's appended to the name of the save file, it counts up by 1 each frame. [Default: 0]

TRAIN_INTERVAL = 1 # The number of frames between each training step. [Default: 1]

SNAPSHOT_INTERVAL = 1 # The number of epochs between showing the human visualization. [Default: 25]
SHOW_FIRST = True # Regardless of snapshot interval, epoch 0 won't show a visualization, unless this is TRUE. [Default: False]

DISABLE_RANDOM = False # Disable epsilon_greedy exploration function. [Default: False]
SAVING_ENABLED = False # Enable saving of model files. [Default: True]
LEARNING_ENABLED = True # Enable model training. [Default: True]
```

