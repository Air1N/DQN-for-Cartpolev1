from collections import deque, namedtuple
import random
import torch
import torch.nn as nn
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import gymnasium as gym
from tqdm import tqdm
from utils.atari_wrappers import ImageToPyTorch
from utils.multiplot import Multiplot
from utils.dqn_utils import GreedyEpsilon, ModelAdjuster

torch.autograd.set_detect_anomaly(True)

# Set the environment name. This model is currently tested on CartPole-v1
DISPLAY_MODE = "human"
environment_name = 'ALE/Breakout-v5'
env = gym.make(environment_name, render_mode=DISPLAY_MODE)
env.metadata['render_fps'] = 30
env = ImageToPyTorch(env)

# Choose device automatically
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Seed for consistency in comparisons
torch.manual_seed(123)

print(f"device: {device}")

# Model step to load, this is the number at the end of the file name @ 0 no file is loaded. [Default: 0]
load_step = 0

# This is here because it's appended to the name of the save file, it counts up by 1 each frame. [Default: load_step=0]
step = load_step

# Path to the model file to load. Automatically generated based on step and environment_name values.
if load_step > 0:
    model_to_load = f"models/{environment_name}/actor_model_{step}.pth"
else: model_to_load = ""

BATCH_SIZE = 64 # The number of transitions per mini-batch [Default: 64]
INPUT_N_STATES = 4 # The number of consecutive states to be concatenated for the observation/input. [Default: 4]

TRAIN_INTERVAL = 1 # The number of frames between each training step. [Default: 1]
SAVE_INTERVAL = 5000 # The number of frames between saving the model to a file. [Default: 500]

EPOCHS = 5000 # This determines the maximum length the program will run for, in epochs. [Default: 500]
EPISODES_PER_EPOCH = 1 # This determines how many episodes, playing until termination, there are in each epoch. [Default: 10]

SNAPSHOT_INTERVAL = 1 # The number of epochs between showing the human visualization. [Default: 25]
SHOW_FIRST = True # Regardless of snapshot interval, epoch 0 won't show a visualization, unless this is TRUE. [Default: False]

SOFT_COPY_INTERVAL = 1 # Number of steps before doing a soft-copy. pred_model.params += actor_model.params * TAU. [Default: 1]
HARD_COPY_INTERVAL = 10000 # Number of steps before doing a hard-copy. pred_model = actor_model. [Default: 10000]

GAMMA = 0.99 # Affects how much the model takes into account future Q-values in the current state. target_output = reward + GAMMA * pred_model(next_state)[actor_model(next_state).argmax()] -- Standard DDQN implementation
TAU = 0.0001 # Affects the speed of parameter transfer during soft-copy. pred_model.params += actor_model.params * TAU. High numbers result in instability. [Default: 0.0001]

ACTOR_LR = 0.00015 # Learning rate used in the policy optimizer. [Default: 0.00015]
CODER_LR = 0.001 # Learning rate used in the decoder optimizer. [Default: 0.001]

REWARD_SCALING = 25 # These are for use more complex reward-shape problems. [Default: +25]
MIN_REWARD = -1 # These are for use in more complex reward-shape problems. [Default: -1]
MAX_REWARD = 1 # These are for use in more complex reward-shape problems. [Default: +1]

REWARD_AFFECT_PAST_N = 4 # Affect how many previous reward states, each with diminishing effects. [Default: 4]
REWARD_AFFECT_THRESH = [-0.1, 0.1] # At what thresholds does the reward propogate to the previous samples? [Default: [-0.8, 2]]

MEMORY_REWARD_THRESH = 0.00 # Assume  anything with less abs(reward) isn't useful to learn, and exclude it from memory [Default: 0.04]

DISABLE_RANDOM = False # Disable epsilon_greedy exploration function. [Default: False]
SAVING_ENABLED = True # Enable saving of model files. [Default: True]
LEARNING_ENABLED = True # Enable model training. [Default: True]

eps = 0.5 # Starting epsilon value, used in the epsilon_greedy policy. [Default: 0.5]
EPS_DECAY = 0.0001 # How much epsilon decays each time a random action is chosen. [Default: 0.0001]
MIN_EPS = 0.05 # Minimum epsilon/random action chance. Keep this above 0 to encourage continued learning. [Default: 0.01]

PLOT_DETAIL = 10000 # The maximum number of points to display at once, afterward this amount of points will be uniformly pulled from the set of all points.
MEDIAN_SMOOTHING = 0 # The amount to divide by for median smooth. In this case, 0 = off. 1 should also = off.

# Surprisal is calculated by taking the sum(abs(next_state_batch - next_state_guess)**exponent)
SURPRISAL_EXPONENT = 1 # TODO The exponent applied to individual differences in next state guess. Essentially, how influential are outliers. [Default: 2]
SURPRISAL_BIAS = 0 # Bias the surprisal score before weighting [Default: -1]
SURPRISAL_WEIGHT = 0.0 # The amount that surprisal influences the reward function. [Default: 0.01]

ENCODER_NODES = 64
CODER_SHUTOFF_LOSS = 0.001 # If the encoder loss goes below this threshold, it stops learning to save computation.

last_c_loss = 10

plt.ion()

# torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(3, sci_mode=False)

multiplot = Multiplot(names=("a_loss", "c_loss", "rb", "real_reward", "cumulative_reward", "natural_reward", "cb", "surprisal", "grad_norm", "rb", "output_0", "output_1", "output_2", "output_3"))
greedy_epsilon = GreedyEpsilon(DISABLE_RANDOM, EPS_DECAY, MIN_EPS)
model_adjuster = ModelAdjuster(TAU, HARD_COPY_INTERVAL, SOFT_COPY_INTERVAL)

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(12, 24, kernel_size=5, stride=5)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=2, stride=2)
        
        self.lin1 = nn.Linear(8064, 128)
        self.linO = nn.Linear(128, ENCODER_NODES)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.lin1(x))
        x = self.linO(x)
        return x

encoder_model = Encoder()
encoder_model.to(device)

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linO = nn.Linear(ENCODER_NODES, 128)
        self.lin1 = nn.Linear(128, 8064)
        self.deconv2 = nn.ConvTranspose2d(24, 24, kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(24, 12, kernel_size=5, stride=5)

    def forward(self, x):
        x = self.linO(x)
        x = F.leaky_relu(self.lin1(x))
        x = torch.unflatten(x, 1, (24, 16, 21))
        x = F.leaky_relu(self.deconv2(x))
        x = F.sigmoid(self.deconv1(x))
        return x

decoder_model = Decoder()
decoder_model.to(device)
coder_optimizer = torch.optim.RAdam(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=CODER_LR)

class CustomDQN(torch.nn.Module):
    """
    This class creates a pytorch DQN with a predetermined structure.

    Attributes:
        isPred (boolean): Whether the model is the prediction model or not.

        self.lin_1 (nn.Linear): Shared input layer.

        self.lin_2a (nn.Linear): Hidden layer for Q-value prediction.
        self.lin_oA (nn.Linear): Output layer for Q-value prediction.

        self.lin_2b (nn.Linear): Hidden layer for next state prediction.
        self.lin_oB (nn.Linear): Output layer for next state prediction.
    """
    def __init__(self, isPred):
        """
        The constructor for the CustomDQN class.

        Parameters:
            isPred (boolean): Whether the model is the prediction model or not.
        """
        super(CustomDQN, self).__init__()

        self.isPred = isPred

        self.lin_1 = nn.Linear(ENCODER_NODES, 64)

        self.lin_2a = nn.Linear(64, 64)
        self.lin_oA = nn.Linear(64, env.action_space.n)

        self.lin_2b = nn.Linear(64 + env.action_space.n, 64)
        self.lin_oB = nn.Linear(64, ENCODER_NODES)

    def forward(self, x, real_actions=None, training=False):
        global eps
        """
        The feed-forward/step function of the model.

        Parameters:
            x (torch.tensor): The input state tensor for the model.
            real_actions (torch.tensor): A batch of real actions the model took, only used in training.
            training (boolean): Enable training-specific changes. i.e. Disables greedy-epsilon.
        
        Returns:
            tuple (a, b):
                - a (torch.tensor): The output action Q-values.
                - b (torch.tensor): The predicted next state.
        """
        
        x = F.leaky_relu(self.lin_1(x)) # Take state as input and run through 1 linear layer

        # First head predicts Q values for actions
        a = F.leaky_relu(self.lin_2a(x))
        a = self.lin_oA(a)

        explore, eps = greedy_epsilon.choose(eps)
        if not training and explore:
            a = torch.rand_like(a) * 2 - 1
        elif not training: print(f"{a} {torch.argmax(a)}")
        
        chosen_actions = torch.argmax(a, dim=1)

        # During training, the action is not taken.
        # Fortunately, an action was already taken in that state and saved. Those saved actions can be used here.
        if real_actions != None:
            chosen_actions = real_actions

        one_hot_encoded_action = torch.zeros_like(a).scatter_(1, chosen_actions.unsqueeze(1), 1.)
        
        # Second head predicts next state from state + Q-values
        b = torch.cat((x, one_hot_encoded_action), dim=1)
        b = F.leaky_relu(self.lin_2b(b))
        b = self.lin_oB(b)

        return a, b

actor_model = CustomDQN(isPred=False)
if model_to_load != "":
    actor_model = torch.load(f"models/{environment_name}/actor_model_{load_step}.pth")
    encoder_model = torch.load(f"models/{environment_name}/encoder_model_{load_step}.pth")
    decoder_model = torch.load(f"models/{environment_name}/decoder_model_{load_step}.pth")

pred_model = CustomDQN(isPred=True)
pred_model.load_state_dict(actor_model.state_dict())
pred_model.eval()

actor_optimizer = torch.optim.RAdam(actor_model.parameters(), lr=ACTOR_LR)

actor_model.to(device)
pred_model.to(device)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class MemoryStack(object):
    """
    This class creates a MemoryStack object with size `capacity`,
    upon reaching capacity the oldest objects will be dropped.

    Attributes:
        memory (deque([], maxlen=capacity)): The memory deque which stores the saved objects -- typically transition tensors.
    """
    def __init__(self, capacity):
        """
        The constructor for the MemoryStack class.

        Parameters:
            capacity (int): The number of objects which will be stored before the oldest objects start being dropped from the deque stack.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, x):
        """
        Push an object to the MemoryStack `.memory` deque.

        Parameters:
            x (any): The element to push to the MemoryStack.
        """
        self.memory.append(x)
    
    def sample(self, batch_size):
        """
        Pull `batch_size` random samples from the MemoryStack `.memory` deque.

        Parameters:
            batch_size (int): The number of individual samples to pull from the memory.
        
        Returns:
            array: An array containing `batch_size` individual samples from memory.
        """
        return random.sample(self.memory, batch_size)

actor_mem = MemoryStack(1000000)


def try_learning():
    """
    Perform checks and start `model_train()`.
    """
    global step, last_c_loss

    if not LEARNING_ENABLED: return

    if len(actor_mem.memory) > BATCH_SIZE:
        if step % TRAIN_INTERVAL == 0:
            a_loss = model_train(BATCH_SIZE)
            multiplot.add_entry('a_loss', a_loss.cpu().detach().numpy())

            if last_c_loss > CODER_SHUTOFF_LOSS:
                c_loss = train_coder(BATCH_SIZE)
                last_c_loss = c_loss
            else: print("\n\nCODER DONE LEARNING :)\n")
            multiplot.add_entry('c_loss', last_c_loss.cpu().detach().numpy() * 50000)


    if step % SAVE_INTERVAL == 0 and SAVING_ENABLED:
        torch.save(actor_model, f"models/{environment_name}/actor_model_{step}.pth")
        torch.save(encoder_model, f"models/{environment_name}/encoder_model_{step}.pth")
        torch.save(decoder_model, f"models/{environment_name}/decoder_model_{step}.pth")


short_memory = []

def affect_short_mem(reward):
    """
    Alter the n=`REWARD_AFFECT_PAST_N` most recent `short_memory` reward values before they're passed into the MemoryStack.

    Parameters:
        reward (float): This value is compared against `MEMORY_REWARD_THRESH` and if its absolute value is higher, then apply the reward to the previous `REWARD_AFFECT_PAST_N` states. The effect is diminished for less recent samples.
    """
    global short_memory

    # If short_memory is long enough:
    if len(short_memory) > REWARD_AFFECT_PAST_N:
        send_short_to_long_mem(1)

    # Only apply if the current reward exceeds a threshold. 
    # Affect short_memory reward values based on reward recieved currently, diminishing for less recent events.
    if reward < REWARD_AFFECT_THRESH[0] or reward > REWARD_AFFECT_THRESH[1]:
        for i in range(0, len(short_memory)):
            short_memory[-(i + 1)][3] += reward / (i + 1)

def send_short_to_long_mem(n):
    """
    Sends the oldest `n` elements from short_memory to actor_mem.

    Parameters:
        n (int): The number of elements to send from short_memory to actor_mem.
    """
    for i in range(0, n):
        # Remove the first element
        short_mem = short_memory.pop(0)

        # Log it as real reward
        multiplot.add_entry('real_reward', short_mem[3][0].cpu().detach())

        # Put it into actor_mem (which is used for training), if the absolute value of the reward is high enough
        if abs(short_mem[3]) >= MEMORY_REWARD_THRESH:
            actor_mem.push(short_mem)


# initialize observation tensors
obs_stack = deque(maxlen=INPUT_N_STATES)

next_obs, info = env.reset()
next_obs = torch.tensor(next_obs).to(device)

while len(obs_stack) < INPUT_N_STATES:
    obs_stack.append(next_obs)

next_state_tensor = torch.cat([*obs_stack], dim=0).to(device)

cumulative_reward = 0
def model_infer():
    """
    1. Observe environment
    2. Make a prediction w/ epsilon greedy policy.
    3. Perform the action.
    4. Attempt to train.

    Repeat until the episode ends.
    """
    global step, obs_stack, cumulative_reward, next_obs, next_state_tensor

    done = False
    cumulative_reward = 0
    while not done:
        state_tensor = next_state_tensor.unsqueeze(0).float()
        actor_model.eval()
        encoder_model.eval()
        with torch.no_grad():
            encoded_state = encoder_model.forward(state_tensor)
            out, _ = actor_model.forward(encoded_state)

            multiplot.add_entry('output_0', float(out.clone()[0].tolist()[0]))
            multiplot.add_entry('output_1', float(out.clone()[0].tolist()[1]))
            multiplot.add_entry('output_2', float(out.clone()[0].tolist()[2]))
            multiplot.add_entry('output_3', float(out.clone()[0].tolist()[3]))

        max_a = torch.argmax(out, dim=1)

        next_obs, reward, terminated, truncated, info = env.step(max_a.cpu().numpy()[0])
        multiplot.add_entry('natural_reward', reward)

        cumulative_reward += reward
        multiplot.add_entry('cumulative_reward', cumulative_reward)

        # terminated is if the pole falls. truncated is when the game times out.
        if terminated or truncated:
            next_obs, info = env.reset()
            cumulative_reward = 0 # reset cumulative reward
            done = True # end episode


            if terminated:
                reward = -2 # punishment for losing

        affect_short_mem(reward)
        
        next_obs = torch.tensor(next_obs).to(device)
        obs_stack.append(next_obs)
        next_state_tensor = torch.cat([*obs_stack], dim=0).float().to(device)
        
        reward = torch.tensor(np.expand_dims(reward, 0), dtype=torch.float32).to(device)

        mem_block = [state_tensor, max_a, next_state_tensor.unsqueeze(0), reward]

        short_memory.append(mem_block)

        if done: send_short_to_long_mem(len(short_memory))

        try_learning()
        model_adjuster.soft_hard_copy(step, actor_model, pred_model)
        step += 1



def train_coder(batch_size):
    """
    This function trains the model using Double-DQN, where the actor_model predicts the next action and then the predictor
    predicts the Q-value of that action for stability reasons.

    Parameters:
        batch_size (int): The amount of samples to include in a minibatch of training.
    
    Returns:
        actor_loss (torch.tensor): Returns the loss of the actor, essentially its error from the target outputs.
    """
    encoder_model.train()
    decoder_model.train()

    transitions = actor_mem.sample(batch_size)
    mem_batch = Transition(*zip(*transitions))

    # Concatenate mem_batch elements to tensors batches
    state_batch = torch.cat(mem_batch.state, dim=0).to(device)
    next_state_batch = torch.cat(mem_batch.next_state, dim=0).to(device)

    # Get the new model output for each state in the batch, including a guess at the next state
    encoded_state = encoder_model.forward(state_batch)
    encoded_next_state = encoder_model.forward(next_state_batch)

    decoded_state = decoder_model.forward(encoded_state)
    decoded_next_state = decoder_model.forward(encoded_next_state)

    # Train Encoder/Decoder nets
    coder_criterion = nn.MSELoss()
    coder_loss = coder_criterion(decoded_state, state_batch) + coder_criterion(decoded_next_state, next_state_batch)
    coder_optimizer.zero_grad()
    coder_loss.backward()
    coder_optimizer.step()

    return coder_loss


def model_train(batch_size):
    """
    This function trains the model using Double-DQN, where the actor_model predicts the next action and then the predictor
    predicts the Q-value of that action for stability reasons.

    Parameters:
        batch_size (int): The amount of samples to include in a minibatch of training.
    
    Returns:
        actor_loss (torch.tensor): Returns the loss of the actor, essentially its error from the target outputs.
    """
    actor_model.train()

    transitions = actor_mem.sample(batch_size)
    mem_batch = Transition(*zip(*transitions))

    # Concatenate mem_batch elements to tensors batches
    state_batch = torch.cat(mem_batch.state, dim=0).to(device)
    action_batch = torch.cat(mem_batch.action, dim=0).to(device)
    next_state_batch = torch.cat(mem_batch.next_state, dim=0).to(device)
    reward_batch = torch.cat(mem_batch.reward, dim=0).to(device) # 64

    # Get the new model output for each state in the batch, including a guess at the next state
    encoded_state = encoder_model.forward(state_batch)
    encoded_next_state = encoder_model.forward(next_state_batch)

    # Train actor / policy net
    state_values, next_state_guess = actor_model.forward(encoded_state, real_actions=action_batch, training=True)

    # Calculate surprisal based on difference of guessed state and real state
    pred_diff = encoded_next_state - next_state_guess
    abs_pred_diff = torch.abs(pred_diff)
    
    diff_from_mean_pred_diff = abs_pred_diff - torch.mean(abs_pred_diff)
    surprisal = torch.sum(diff_from_mean_pred_diff, dim=1)
    scaled_surprisal = (surprisal + SURPRISAL_BIAS) * SURPRISAL_WEIGHT
    multiplot.add_entry("surprisal", (torch.max(scaled_surprisal) - torch.min(scaled_surprisal)).cpu().detach().numpy() * 50000)
    
    # Gather the Q-value of the actual actions chosen.
    state_actions = state_values.gather(1, action_batch.unsqueeze(1)) # 64, 1

    with torch.no_grad():
        # Select next action using current model
        actor_next_preds, _ = actor_model.forward(encoded_next_state, training=True) # 64, 2
        actor_pred_max_a = torch.argmax(actor_next_preds, dim=1) # 64
        
        # Predict target Q-value at next_state using the more stable prediction model
        pred_out, _ = pred_model.forward(encoded_next_state, training=True) # 64, 2
        next_state_actions = pred_out.gather(1, actor_pred_max_a.unsqueeze(1)) # 64, 1

    # Generate the target output, by adding the reward at each transition, to the Q-value of the next action (predicted reward) * GAMMA, a discount factor.
    target_output = scaled_surprisal.unsqueeze(1) + reward_batch.unsqueeze(1) + (next_state_actions * GAMMA)

    # Loss is the difference between the target outputs and the real outputs,
    # plus the difference between the next state and the predicted next state.
    actor_criterion = nn.HuberLoss()
    actor_loss = actor_criterion(state_actions, target_output) + actor_criterion(next_state_guess, encoded_next_state)
    actor_optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)

    # Log gradient norm
    grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in actor_model.parameters()]).detach().cpu())
    multiplot.add_entry('grad_norm', grad_norm)

    # Clip gradients for stability
    torch.nn.utils.clip_grad_value_(actor_model.parameters(), 1)
    actor_optimizer.step()

    return actor_loss



def main():
    global step, env, next_obs, obs_stack, next_state_tensor

    for epoch in tqdm(range(EPOCHS)):
        # Decide whether to display the environment
        if epoch % SNAPSHOT_INTERVAL == 0 and (epoch != 0 or SHOW_FIRST):
            render_mode = DISPLAY_MODE
        else:
            pass
            #render_mode = None

        # Load a new version of the environment with the chosen render_mode
        # env = gym.make(environment_name, render_mode=DISPLAY_MODE)
        # env.metadata['render_fps'] = 150
        # env = ImageToPyTorch(env)
        # TODO this doesn't work. Make render mode switching work if possible

        next_obs, info = env.reset()
        # env.render() - not necessary

        # Re-initialize obervations, etc.
        obs_stack = deque(maxlen=INPUT_N_STATES)
        next_obs = torch.tensor(next_obs).to(device)

        while len(obs_stack) < INPUT_N_STATES:
            obs_stack.append(next_obs)

        next_state_tensor = torch.cat([*obs_stack], dim=0).to(device)

        if len(info) > 0: print(info)

        for episode in tqdm(range(EPISODES_PER_EPOCH)):
            model_infer()

        multiplot.plot_all(step)


main()
