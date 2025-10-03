import torch, torch.nn as nn, torch.optim as optim
import random, numpy as np


class QNet(nn.Module):
    def __init__(self, in_dim=3, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, in_dim=3, out_dim=5, lr=1e-3, gamma=0.99):
        self.q = QNet(in_dim, out_dim)
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.mem = []
        self.gamma = gamma


    def act(self, obs, eps=0.1):
        if random.random() < eps:
            return random.randrange(5)
        with torch.no_grad():
            q = self.q(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax().item())


    def push(self, s,a,r,ns,done):
        self.mem.append((s,a,r,ns,done))
        if len(self.mem)>100000: self.mem.pop(0)


    def train_step(self, batch_size=64):
        if len(self.mem) < batch_size: return 0.0
        import random
        batch = random.sample(self.mem, batch_size)
        s,a,r,ns,d = zip(*batch)
        s = torch.tensor(np.array(s), dtype=torch.float32)
        ns = torch.tensor(np.array(ns), dtype=torch.float32)
        a = torch.tensor(a)
        r = torch.tensor(r, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)
        q = self.q(s).gather(1, a.view(-1,1)).squeeze(1)
        with torch.no_grad():
            tq = self.q(ns).max(1)[0]
            y = r + (1-d)*self.gamma*tq
        loss = ((q - y)**2).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss.item())