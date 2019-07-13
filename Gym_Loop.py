import gym
import numpy as np
import MimicModel
import torch
import torchvision

def run_env():
    env_name = "CarRacing-v0"
    env = gym.make(env_name)

    data =[]

    observation = env.reset()
    past_observation = observation
    for i in range(10):
        past_observation = observation

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        data.append((past_observation,action,observation))


    return data



if __name__ == "__main__":
    model = MimicModel.MimicNet()
    data = run_env()
    for obs,action,next in data:
        obs = np.ascontiguousarray([obs])
        obs = torch.from_numpy(obs)
        obs = obs.view(-1,3,96,96).float()
        obs = torchvision.Grayscale(obs)
        print(model.encode(obs).shape)


    
