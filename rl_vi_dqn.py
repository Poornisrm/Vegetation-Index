# rl_vi_dqn.py
import random, os, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import f1_score
from collections import deque, namedtuple
from tqdm import trange

# Config (Table 7)
STATE_DIM = 10   # example: number of bands or features (set according to your data)
ACTION_SPACE = 100  # discretize choices of triplets/weights (you can encode triplets -> integer)
HIDDEN_SIZES = [128, 64]
LR = 0.001
GAMMA = 0.95
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.1, 0.995
BATCH_SIZE = 64
REPLAY_SIZE = 10000
EPISODES = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple MLP Q-network
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZES[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[1], action_dim)
        )
    def forward(self,x): return self.net(x)

# Replay buffer
Transition = namedtuple('Transition', ('s','a','r','s2','done'))
class Replay:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, *args): self.buf.append(Transition(*args))
    def sample(self, n): 
        batch = random.sample(self.buf, n)
        return Transition(*zip(*batch))
    def __len__(self): return len(self.buf)

# Dummy environment wrapper: you must implement get_state(), apply_action() and evaluate_index()
# Here is a skeleton. Replace data loader & eval with your dataset.
class Env:
    def __init__(self, X_train, y_train, classifier_trainer):
        self.X = X_train; self.y = y_train
        self.classifier_trainer = classifier_trainer
    def reset(self):
        # sample random state from dataset
        idx = np.random.randint(0, len(self.X))
        return self.X[idx]
    def step(self, state, action):
        # action -> create VI; for prototype use action as a selector which picks 3 band indices
        # decode action
        i = action % state.shape[0]
        j = (action//state.shape[0]) % state.shape[0]
        k = (action//(state.shape[0]**2)) % state.shape[0]
        # compute index vector: here simple linear combination
        vi_vector = state[[i,j,k]].mean()  # simplified
        # evaluate: train classifier on a small subset using generated VI as feature and compute F1
        f1 = self.classifier_trainer(vi_vector, self.y)  # implement classifier_trainer to return F1
        reward = f1
        next_state = state  # for simplicity
        done = True
        return next_state, reward, done

# Example classifier trainer skeleton (replace with SVM/CNN pipeline)
def example_classifier_trainer(feature_vector, y):
    # feature_vector would be a per-sample scalar/vector. For demo, return random F1
    return random.random()

# Initialize
action_dim = ACTION_SPACE
q = QNet(STATE_DIM, action_dim).to(device)
target_q = QNet(STATE_DIM, action_dim).to(device)
target_q.load_state_dict(q.state_dict())
opt = optim.Adam(q.parameters(), lr=LR)
replay = Replay(REPLAY_SIZE)
env = Env(np.random.randn(1000, STATE_DIM), np.random.randint(0,5,1000), example_classifier_trainer)

eps = EPS_START
for ep in trange(EPISODES):
    state = torch.tensor(env.reset(), dtype=torch.float32).to(device)
    done = False
    while not done:
        if random.random() < eps:
            action = random.randrange(action_dim)
        else:
            with torch.no_grad():
                qvals = q(state.unsqueeze(0))
                action = int(qvals.argmax().item())
        # apply action
        s2, reward, done = env.step(state.cpu().numpy(), action)
        replay.push(state.cpu().numpy(), action, reward, s2, done)
        state = torch.tensor(s2, dtype=torch.float32).to(device)
        if len(replay) >= BATCH_SIZE:
            batch = replay.sample(BATCH_SIZE)
            states = torch.tensor(np.vstack(batch.s), dtype=torch.float32).to(device)
            actions = torch.tensor(batch.a).long().to(device)
            rewards = torch.tensor(batch.r).float().to(device)
            next_states = torch.tensor(np.vstack(batch.s2), dtype=torch.float32).to(device)
            dones = torch.tensor(batch.done).float().to(device)

            qvals = q(states).gather(1, actions.unsqueeze(1)).squeeze()
            with torch.no_grad():
                target_next = target_q(next_states).max(1)[0]
                target = rewards + GAMMA * target_next * (1 - dones)
            loss = nn.MSELoss()(qvals, target)
            opt.zero_grad(); loss.backward(); opt.step()
    # decay epsilon
    eps = max(EPS_END, eps * EPS_DECAY)
    # periodic target update
    if ep % 10 == 0:
        target_q.load_state_dict(q.state_dict())
    # log progress
    if ep % 10 == 0:
        print(f"Episode {ep} eps {eps:.3f}")

print("RL training complete. Save model if desired.")
torch.save(q.state_dict(), "models/rlvi_qnet.pth")
