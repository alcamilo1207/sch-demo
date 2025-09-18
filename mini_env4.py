import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self, w1 = 1, w2 = 1, w3 = 1, prices_fx = None, machine_eff = False, energy_prices = False, test = False, random_training = True, params = None): 
        super(CustomEnv, self).__init__()

        # Optimizer parameters
        self.machine_eff = machine_eff # Considering machine efficiency?
        self.energy_prices = energy_prices # Considering energy prices?
        self.w1 = w1 # Machine efficiency reward: How much reward is given to the most efficient machine in comparizon with the least efficient
        self.w2 = w2 # Idling reward: How much reward is given when idling at High prices. At Normal/Low prices the reward given for idling is zero
        self.w3 = w3 # Termination reward: How much reward is given if all jobs were completed

        if params is None:
            self.params = dict(dataset_job_size = 40, job_min_size = 4, job_max_size = 7)


        # Environment variables
        self.history = []
        self.prices_fx = np.ones(110)
        self.available_jobs = [] # Store the jobs that have not been scheduled. Changed in the reset and step function
        self.dataset_jobs = [] # Store the dataset of jobs and it does not change after initilizing the environment
        self.machine_times = [0 for _ in range(3)]
        self.step_count = 0

        # Dataset - Jobs
        if test or random_training:
            if prices_fx is None:
                #print("&&&& Generating random energy prices and job list dataset")
                self.shuffle_datasets()
            else:
                print("&&&& Testing with provided energy prices profile")
                self.prices_fx[:96] = prices_fx

        else:
            self.dataset_jobs.extend([7, 4, 5, 6, 5])
            self.dataset_jobs.extend([5, 7, 6, 4, 5])

            for _ in range(2):
                self.dataset_jobs.extend(self.dataset_jobs)

            self.prices_fx[20:28] = 2
            self.prices_fx[55:60] = 2


        
        # Define the observation space
        self.observation_space = spaces.Dict({
            'remaining_jobs' : spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'current_machine' : spaces.Discrete(3), 
            'remaining_time' : spaces.Box(low=-10, high=96, shape=(1,), dtype=np.int32),
            'next_jobs': spaces.Box(low=0, high=10, shape=(3,), dtype=np.int32),
            'future_prices': spaces.Box(low=-15, high=15, shape=(3,), dtype=np.int32),
        })
        
        # Define the action space
        self.action_space = spaces.Discrete(5) # 0: idling, 1-3: The position of the selected job in the job queue, 4 for do nothing
        
        # Initialize the obs
        self.obs = {
            'remaining_jobs' : np.array([0], dtype=np.float32),
            'current_machine' : 0, 
            'remaining_time': np.zeros((1,), dtype=np.int32),
            'next_jobs': np.zeros((3,), dtype=np.int32),
            'future_prices': np.zeros((3,), dtype=np.int32)
        }
        
    def reset(self, seed = None):
        # Reset step count
        self.step_count = 0
        self.machine_times = [0 for _ in range(3)]
        self.available_jobs = self.dataset_jobs

        # Reset history
        self.history = []

        # Reset the obs to some initial values
        self.obs = {
            'remaining_jobs' : np.array([0], dtype=np.float32),
            'current_machine' : 0, 
            'remaining_time': np.array([96], dtype=np.int32),
            'next_jobs': np.array(self.available_jobs[:3], dtype=np.int32),
            'future_prices': np.array(self.prices_fx[:3], dtype=np.int32),
        }

        self.available_jobs = self.available_jobs[3:]

        # info
        info = {
            "history" : [],
        }
        return (self.obs, info)

    def step(self, action):
        reward = 0

        current_machine = self.step_count % 3
        next_machine = (self.step_count + 1) % 3
        next_machine_time = self.machine_times[next_machine]
        job = int(action.item()) - 1
        size_of_job = np.int32(0)
        
        if job > -1 and job < 3:
            size_of_job += self.obs["next_jobs"][job]

        remaining_jobs = self.obs['remaining_jobs'] + np.float32(size_of_job/np.sum(self.dataset_jobs))
        
        # calculate cost
        reward += self.calculate_reward(job, current_machine, size_of_job,  duration=size_of_job)

        # Update history and machine times
        self.history.append(self.get_history(job, current_machine, size_of_job, reward))

        # Check if the episode is done
        terminated, termination_rew = self.is_terminated()
        truncated = self.is_truncated()

        # Update reward if all jobs are completed at termination
        reward += termination_rew

        if terminated or truncated:
            # Optionally, set additional info
            info = {
                "step count" : self.step_count,
                "rewards" : {"size of job": size_of_job, "reward": reward},
                "history" : self.history,
                "prices" : self.prices_fx
            }
        else:
            # Optionally, set additional info
            info = {
                "step count" : self.step_count,
                "rewards" : {"size of job": size_of_job, "reward": reward},
            }

        # Update environment
        self.update_next_jobs(job)
        
        # update observations
        self.obs['remaining_jobs'] = remaining_jobs
        self.obs['current_machine'] = next_machine
        self.obs['remaining_time'] = np.array([96 - self.machine_times[next_machine]], dtype=np.int32)
        self.obs['future_prices'] = np.array(self.prices_fx[next_machine_time: next_machine_time + 3], dtype=np.int32)

        self.step_count += 1

        return self.obs, reward, terminated, truncated, info

    def shuffle_datasets(self):
        # Reset jobs
        self.dataset_jobs = []
        self.dataset_jobs.extend(np.random.randint(low = self.params["job_min_size"], 
                                                   high = self.params["job_max_size"]+1, 
                                                   size = self.params["dataset_job_size"]
                                                  ).tolist())
        # Reset prices
        self.prices_fx = np.ones(110)
        high_zone_1_start = np.random.randint(low=0, high=30)
        high_zone_1_len = np.random.randint(low=1, high=15)
        high_zone_2_start = np.random.randint(low=1, high=60)
        high_zone_2_len = np.random.randint(low=4, high=15)
        self.prices_fx[high_zone_1_start:high_zone_1_start+high_zone_1_len] = 2
        self.prices_fx[-high_zone_2_start-high_zone_2_len:-high_zone_2_start] = 2


    def update_next_jobs(self, job): 
        if job > -1 and job < 3:
            if len(self.available_jobs) > 0:
                self.obs["next_jobs"][job] = self.available_jobs[0]
                self.available_jobs = self.available_jobs[1:]
            else:
                self.obs["next_jobs"][job] = np.int32(0)

    
    def get_history(self, job, machine, size_of_job, reward):
        if job > -1:
            self.machine_times[machine] += size_of_job.item()
            return [size_of_job.item(), machine, size_of_job.item(), reward] # Duration, machine, size_of_job, reward
        else:
            if job != 3:
                self.machine_times[machine] += 1
                return [1, machine, size_of_job, reward]
            else:
                return [0, machine, size_of_job, reward]
        
    def calculate_cost(self, current_machine, current_machine_time, duration):
        cost = 0
        for i in range(duration):
            cost += self.prices_fx[i + current_machine_time] * (1 + current_machine)

        return cost
    
    def calculate_reward(self, job, current_machine, size, duration):
        current_machine_time = self.machine_times[current_machine]
        cost = self.calculate_cost(current_machine, current_machine_time, duration)
        if size != 0:
            return size.item() * (1 + self.w1*(1 - current_machine) + (1/cost))
        else:
            if job != 3:
                if self.obs["future_prices"][0] == 2:
                    return (0.5 + current_machine) * self.w2 # 2 * self.w2
                else:
                    return 0
            else:
                return 0
      
    def is_terminated(self):
        termination_rew = 0
        terminated = max(self.obs["next_jobs"]) == 0 # len(self.available_jobs) < 1 and 
        if terminated:
            termination_rew += 100 * self.w3

        terminated = terminated or max(self.machine_times) > 96
        return terminated, termination_rew
    
    def is_truncated(self):
        truncated = self.step_count > 288
        return truncated


# env = CustomEnv(energy_prices=True, machine_eff=True, test=True)

# # Reset the environment
# obs = env.reset()
# total_rew = 0
# steps = 0
# while True:
#     print(f"\nObs: {obs}")
#     action = env.action_space.sample()
#     obs, reward, terminated, _ , info = env.step(action)
#     total_rew += reward
#     steps += 1
#     print(f"Action: {action}")
#     print(f"Rewars: {reward}")
#     if terminated:
#         print(info["rewards"])
#         print("step count:",info["step count"])
#         break

# env.close()
# print("total_rew",total_rew, "steps", steps)