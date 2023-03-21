import numpy as np
import matplotlib.pyplot as plt

import random


shapes = ("rectangle", "circle")
FILE = 'map_space.npy'


def load(map, start, goal):
    map[map > 0] = 1
    map[start[0], start[1]] = 0.5
    map[goal[0], goal[1]] = 0.5
    plt.set_cmap('binary')
    plt.imshow(map)
    plt.tight_layout()
    plt.show()

class DataGeneration(object):
    
    def __init__(self, dim, num_obs=1, obs_type = None) -> None:
        self.dataset = {'maps': [], 'starts': [], 'goals': [], 'paths': []}
        self.dim = dim
        self.num_obs = num_obs
        self.obs_type = obs_type

    def generate(self, num):
        itr = 0
        _samples = 0
        # for _ in range(num):
        while (_samples < num): 
            self.FlagFailed = False
            self.map = np.zeros((self.dim[0], self.dim[1]), dtype=np.float64)
            while (True):
                self.start = np.array([random.randint(1, self.dim[0] - 1), random.randint(1, self.dim[1] - 1 )])
                self.goal = np.array([random.randint(1, self.dim[0] - 1), random.randint(1, self.dim[1] - 1 )])
                if all(x == y for x, y in zip(self.start, self.goal)):
                    self.start = np.array([random.randint(1, self.dim[0] - 1), random.randint(1, self.dim[1] - 1 )])
                    self.goal = np.array([random.randint(1, self.dim[0] - 1), random.randint(1, self.dim[1] - 1 )])
                else:
                    break
            if self.obs_type == "rectangle":
                while (itr < self.num_obs): 
                    size_0, size_1 = (self.dim[0] - 1)//random.randint(2, 4), (self.dim[0] - 1)//random.randint(2, 4)
                    center = [random.randint(1, self.dim[0] - 1), random.randint(0, self.dim[1] - 1)]
                    height = random.randint(1, size_0) 
                    width = random.randint(1, size_1)
                    if self.check_geometry(center, height, width) and self._check_goal_start(center, height, width, self.start, self.goal):
                        if (self.rectangle(center, height, width)):
                            itr += 1
                            # print("itr - >", itr)
                    else:
                        print(f"failed, {center}, {height}, {width}")
                        # self.FlagFailed = True
                    
            elif self.obs_type == "circle":
                for _ in range(self.num_obs):
                    pass
            elif self.obs_type == "all" or self.obs_type == None:
                for _ in range(self.num_obs):
                    pass
            if self.FlagFailed == False:
                self.dataset['maps'].append(self.map)
                self.dataset['starts'].append(self.start)
                self.dataset['goals'].append(self.goal)
                itr = 0
                _samples += 1

        # self.dataset)

    def _check_goal_start(self, center, height, length, start, goal):
        for i in range(0, height):
            for j in range(0, length):
                if center[0] - (height // 2) + i == start[0] and center[1] - (length // 2) + j == start[1]:
                    return False
                if center[0] - (height // 2) + i == goal[0] and center[1] - (length // 2) + j == goal[1]:
                    return False
        return True

    def _check_rectangle(self, center, height, length):
        pass

    def check_geometry(self, center, height, length):
        for i in range(length // 2):
            if (center[1] + i > self.dim[1]):
                return False
            if (center[1] - i < 0):
                return False
        for i in range(height // 2):
            if (center[0] + i  > self.dim[0]):
                return False
            if (center[0] - i < 0):
                return False
        return True

    def circle(self, center, radius):
        pass

    def rectangle(self, center, height, length):
        try: 
            for i in range(0, height):
                for j in range(0, length):
                    self.map[center[0] - (height // 2) + i][center[1] - (length // 2) + j] = 1
            return True
        except:
            print("FAILED TO RECTANGLE")
            return False

    def save(path):
        pass



if __name__ == "__main__":
    pass
    # dim = (256, 256)
    # num_obs = 5
    # num_samples = 1
    # obs_type = "rectangle"
    # generator = DataGeneration(dim, num_obs, obs_type)
    # generator.generate(num_samples)
    # breakpoint()

