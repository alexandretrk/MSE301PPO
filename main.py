from Library import PolicyNetwork, ValueNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import gym


def optimize(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_ppo(policy_network, value_network, policy_opt, value_opt, env, epochs, steps, epsilon):
    policyLosses = []
    valueLosses = []
    rewardsTotal = []

    for epoch in range(epochs):
        #Recupération des trajectoires pour chaque step
        states = []
        actions = []
        rewards = []
        log_probs = []
        state = env.reset()

        for i in range(steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, std = policy_network(state_tensor)
            action_dist = torch.distributions.Normal(mean, std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=1, keepdim=True)
            next_state, reward, done, info = env.step(action.detach().numpy()[0])
            states.append(state)
            actions.append(action.detach().numpy()[0])
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state
            if done:
                state = env.reset()

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        log_probs = torch.FloatTensor(log_probs).unsqueeze(1)

        #Calcul des pertes
        values = value_network(states)
        advantages = rewards - values.detach()

        mean, std = policy_network(states)
        normal_dist = torch.distributions.Normal(mean, std)
        new_log_probs = normal_dist.log_prob(actions).sum(dim=1, keepdim=True)
        ratio = torch.exp(new_log_probs - log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_clipped_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.MSELoss()(values, rewards)

        #Optimisation

        optimize(policy_opt, policy_clipped_loss)
        optimize(value_opt, value_loss)

        print(f'Epoch {epoch+1}/{epochs}, Policy Loss: {policy_clipped_loss.item()}, Value Loss: {value_loss.item()}, Rewards: {rewards.sum()}')

        policyLosses.append(policy_clipped_loss.item())
        valueLosses.append(value_loss.item())
        rewardsTotal.append(rewards.sum())

    return policyLosses, valueLosses, rewardsTotal



def run(epochs = 1000, steps = 2000, lr = 3e-4, minibatch = 64, epsilon = 0.2):
    #torch.manual_seed(100)
    #env.seed(10)
    #Initialisation
    env = gym.make('Hopper-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy_network = PolicyNetwork(state_dim, action_dim, minibatch)
    value_network = ValueNetwork(state_dim, minibatch)

    policy_opt = optim.Adam(policy_network.parameters(), lr=lr)
    value_opt = optim.Adam(value_network.parameters(), lr=lr)

    #Entrainement
    policyLosses, valueLosses, rewardsTotal = train_ppo(policy_network, value_network, policy_opt, value_opt, env, epochs=epochs, steps=steps, epsilon=epsilon)

    return policyLosses, valueLosses, rewardsTotal

if __name__ == '__main__':
    run(1000,2000)

