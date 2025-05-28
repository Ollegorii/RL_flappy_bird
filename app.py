import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import game.wrapped_flappy_bird as game
import pygame
from pygame.locals import *
import sys
import cv2
import matplotlib.pyplot as plt

# Hyperparameters
GAMMA = 0.99
INITIAL_EPSILON = 0.05
FINAL_EPSILON = 0.001
EPSILON_DECAY = 0.99995
MEMORY_SIZE = 200_000
BATCH_SIZE = 128
UPDATE_TARGET_FREQ = 1000
LEARNING_RATE = 3e-4
INPUT_SHAPE = (4, 80, 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        
        # Рассчитываем размер выхода сверточных слоев
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(80, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(80, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = INITIAL_EPSILON
        self.model = DQN().to(device)
        self.target_model = DQN().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.steps = 0
        self.losses = []
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
            q_values = self.model(state)
            return torch.argmax(q_values).item()
        
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
            
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states = torch.FloatTensor(np.array([x[0] for x in minibatch])).to(device)
        actions = torch.LongTensor(np.array([x[1] for x in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([x[2] for x in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([x[3] for x in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([x[4] for x in minibatch])).to(device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * GAMMA * next_q
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        self.losses.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > FINAL_EPSILON:
            self.epsilon *= EPSILON_DECAY
            
        self.steps += 1
        if self.steps % UPDATE_TARGET_FREQ == 0:
            self.update_target_model()

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (80, 80), interpolation=cv2.INTER_AREA)
    return np.array(image, dtype=np.float32) / 255.0

def train_agent():
    env = game.GameState()
    agent = DQNAgent()
    
    image_data, _, _, _ = env.frame_step([1, 0])
    image = preprocess_image(image_data)
    state = np.stack([image] * 4, axis=0)
    
    max_score = 0
    episode = 0
    scores = []
    
    try:
        while True:
            action = agent.act(state)
            input_actions = [1, 0] if action == 0 else [0, 1]
            
            image_data, reward, terminal, score = env.frame_step(input_actions)
            next_image = preprocess_image(image_data)
            next_state = np.append(state[1:], [next_image], axis=0)
            
            agent.remember(state, action, reward, next_state, terminal)
            agent.replay()
            scores.append(score)
            if score > max_score:
                max_score = score
            
            state = next_state
            
            # if reward == 1:
            #     max_score += 1
            #     print(f"Score: {max_score}, Epsilon: {agent.epsilon:.4f}")
            
            if terminal:
                print(f"Episode: {episode}, Score: {scores[-2]}, max_score: {max_score}, Epsilon: {agent.epsilon:.4f}")

                if max_score >= 250:
                    print("Success! Achieved score of 250+")
                    break
                
                episode += 1
                image_data, _, _, _ = env.frame_step([1, 0])
                image = preprocess_image(image_data)
                state = np.stack([image] * 4, axis=0)
                
    except KeyboardInterrupt:
        pass
    
    torch.save(agent.model.state_dict(), 'flappy_bird_dqn.pth')
    print("Model saved")
    
    # Plot training results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Scores during training')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(agent.losses)
    plt.title('Training loss')
    plt.xlabel('Training step')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()
    
    pygame.quit()
    return agent

def play_with_trained_agent(model_path='flappy_bird_dqn.pth'):
    env = game.GameState()
    
    model = DQN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    image_data, _, _ , _ = env.frame_step([1, 0])
    image = preprocess_image(image_data)
    state = np.stack([image] * 4, axis=0)
    
    max_score = 0
    
    try:
        while True:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()
            
            input_actions = [1, 0] if action == 0 else [0, 1]
            image_data, reward, terminal, score = env.frame_step(input_actions)
            next_image = preprocess_image(image_data)
            state = np.append(state[1:], [next_image], axis=0)
            
            print(f"Score: {score}")

            if score > max_score:
                max_score = score
            
            if terminal:
                print(f"Final Score: {max_score}")
                break
                
    except KeyboardInterrupt:
        pass
    
    pygame.quit()

if __name__ == '__main__':
    # Train the agent
    # train_agent()

    
    # Play with trained agent
    play_with_trained_agent(model_path='flappy_bird_dqn_120.pth')