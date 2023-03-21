import numpy as np
import matplotlib.pyplot as plt
import random
import math

from dataset import DataGeneration
from model import Encoder

from matplotlib.pyplot import rcParams
np.set_printoptions(precision=3, suppress=True)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
plt.rcParams['font.size'] = 22


def load(map, start, goal):
    map[map > 0] = 1
    plt.set_cmap('binary')
    plt.imshow(map)
    plt.tight_layout()
    # plt.show()


class treeNode():
    def __init__(self, locationX, locationY):
        self.locationX = locationX
        self.locationY = locationY
        self.children = []
        self.parent = None

class RRTAlgoritm():
    def __init__(self, start, goal, numIterations, grid, stepSize):
        self.randomTree = treeNode(start[0], start[1])
        self.goal = treeNode(goal[0], goal[1])
        self.nearestNode = None
        self.iterations = min(numIterations, 1000)
        self.grid = grid
        self.rho = stepSize
        self.path_distance = 0
        self.nearestDist = 10000
        self.numWaypoints = 0
        self.Waypoints = []
        self.stepSize  = stepSize

    #add the node to the nearest node and goal when reached
    def addChild(self, locationX, locationY):
        """add the point to the nearest node and add goal when reached
        """
        if (locationX == self.goal.locationX):
            self.nearestNode.children.append(self.goal)
            self.goal.parent = self.nearestNode
        else:
            tempNode = treeNode(locationX, locationY)
            self.nearestNode.children.append(tempNode)
            tempNode.parent = self.nearestNode

    def sampleAPoint(self):
        """sample a random point within grid limits
        """
        pt_x = round(random.random() * (len(self.grid[0]) - 1)) 
        pt_y = round(random.random() * (len(self.grid) - 1))
        assert(pt_x < len(self.grid[0]))
        assert(pt_y < len(self.grid))
        assert(0 <= pt_x )
        assert(0 <= pt_y)
        return np.array([pt_x, pt_y])


    def steerToPoint(self, locationStart, locationEnd):
        """steer a distance stepsize from start to end location
        """
        offset = self.rho * self.unitVector(locationStart, locationEnd)
        point = np.array([locationStart.locationX + offset[0], locationStart.locationY + offset[1]])
        if point[0] >= self.grid.shape[1]:
            point[0] = self.grid.shape[1]
        elif point[0] < 0:
            point[0] = 0
        if point[1] >= self.grid.shape[0]:
            point[1] = self.grid.shape[1]
        elif point[1] < 0:
            point[0] = 0
        return point

    def isInObstacle(self, locationStart, locationEnd):
        """check if obstacle lies between the start and end nodes
        """
        u_hat = self.unitVector(locationStart, locationEnd)
        testPoint = np.array([0.0, 0.0])
        for i in range(self.rho):
            testPoint[0] = locationStart.locationX + i * u_hat[0]
            testPoint[1] = locationStart.locationY + i * u_hat[1]
            if testPoint[0] >= self.grid.shape[0]:
                testPoint[0] = self.grid.shape[0] - 1
            elif testPoint[0] < 0:
                testPoint[0] = 0
            if testPoint[1] >= self.grid.shape[1]:
                testPoint[1]= self.grid.shape[1] - 1
            elif testPoint[1] < 0:
                testPoint[1] = 0
            if self.grid[math.floor(testPoint[1]), math.floor(testPoint[0])] == 1:
                return True
        return False

 
    def unitVector(self, locationStart, locationEnd):
        """find unit vector between 2 points which form a vector
        """
        vec = np.array([locationEnd[0] - locationStart.locationX, locationEnd[1] - locationStart.locationY])
        u_vec = vec / np.linalg.norm(vec)
        return u_vec


    def findNearest(self, root, point):
        """find the nearest node from a given unconnected point (Euclidean distance)
        """
        if not root:
            return 
        if self.nearestDist >= self.distance(root, point):
                self.nearestDist = self.distance(root, point)
                self.nearestNode = root
        for child in root.children:
            self.findNearest(child, point)
           
    
    def distance(self, node1, point):
        """find euclidean distance between a node and an XY point
        """
        node_x = node1.locationX
        node_y = node1.locationY
        L2 = np.sqrt((node_x - point[0])**2 + (node_y - point[1])**2)
        return L2
    
    def goalFound(self, point):
        """check if the goal has been reached within step size
        """
        if self.distance(self.goal, point) < self.stepSize:
            return True
        else:
            return False


    def resetNearestValue(self):
        """reset nearestNode and nearest Distance
        """
        self.nearestNode = None
        self.nearestDist = 10000


    def retraceRRTPath(self, goal):
        """trace the path from goal to start
        """
        if goal.locationX == self.randomTree.locationX:
            return
        self.numWaypoints += 1
        currentPoint = np.array([goal.locationX, goal.locationY])
        self.Waypoints.insert(0, currentPoint)
        self.path_distance += self.rho
        self.retraceRRTPath(goal.parent)

    def expert_policy(self):
        pass


def create_dataset(dim, num_samples, num_obs, obs_type="rectangle", config=None, plot=False):
    generator = DataGeneration(dim, num_obs, obs_type)
    flagFound = False
    # try:
    generator.generate(num_samples)
    plot = True
    # except:
    #     print("Error in creating the dataset")
    assert(len(generator.dataset['maps']) == num_samples)
    for i in range(num_samples):
        start = generator.dataset['starts'][i] 
        goal = generator.dataset['goals'][i]
        # print(f"i -> {i}")
        # print(f"maps -> {generator.dataset['maps'][i]}")
        grid = generator.dataset['maps'][i]
        maxIteration = 1000
        stepSize = 25
        rrt = RRTAlgoritm(start, goal, maxIteration, grid, stepSize)
        if plot:
            goalRegion = plt.Circle((goal[0], goal[1]), stepSize, color='b', fill=False)
            fig = plt.figure("RRT Algorithm")
            plt.imshow(grid, cmap='binary')
            plt.plot(start[0], start[1], 'ro')
            plt.plot(goal[0], goal[1], 'bo')
            # plt.imshow(grid)
            ax = fig.gca()
            ax.add_patch(goalRegion)
            # breakpoint()
            # plt.show()
        for i in range(rrt.iterations):
            rrt.resetNearestValue()
            point = rrt.sampleAPoint()
            rrt.findNearest(rrt.randomTree, point)
            new = rrt.steerToPoint(rrt.nearestNode, point)
            bool = rrt.isInObstacle(rrt.nearestNode, new)
            if (bool == False):
                rrt.addChild(new[0], new[1])
                if plot:
                    plt.plot([rrt.nearestNode.locationX, new[0]], [rrt.nearestNode.locationY, new[1]], 'go', linestyle='--')
                    plt.pause(0.0001)
                if (rrt.goalFound(new)):
                    rrt.addChild(goal[0], goal[1])
                    flagFound = True
                    print("found goal!!")
                    break

        if flagFound:
            rrt.retraceRRTPath(rrt.goal)
            rrt.Waypoints.insert(0, start)
            generator.dataset['paths'].append(rrt.Waypoints)
            flagFound = False
        else:
            continue


        if plot:
            for i in range(rrt.numWaypoints- 1):
                plt.plot([rrt.Waypoints[i][0], rrt.Waypoints[i+1][0]], [rrt.Waypoints[i][1], rrt.Waypoints[i+1][1]], 'ro', linestyle='--')
                plt.pause(0.10)
            plt.close()
    
    return format(generator.dataset)


def format(dataset):
        max_length = 0
        encode_size = 28
        N = len(dataset['paths'])
        encoder = Encoder(dataset['maps'][0].shape[0]*dataset['maps'][0].shape[0], encode_size)
        path_length = np.zeros(len(dataset['paths']), dtype=np.int32)
        for i, path in enumerate(dataset['paths']):
            path_length[i] = np.array(path).shape[0]
            if np.array(path).shape[0] > max_length:
                max_length = np.array(path).shape[0]
        paths = np.zeros((N, max_length, 2))


        for i in range(N):
            for j in range(path_length[i]):
                paths[i][j] = dataset['paths'][i][j]

        train = []
        targets = []

        for i in range(0, N):
            for j in range(path_length[i] - 1):
                breakpoint()
                encoded_data = encoder(dataset['maps'][i])
                data = np.zeros(32)
                data[28] = paths[i][j][0]
                data[29] = paths[i][j][1]
                data[30] = paths[i][path_length[i] - 1][0]
                data[31] = paths[i][path_length[i] - 1][1]
                targets.append(paths[i][j + 1])

            


        
        breakpoint()
    


if __name__ == "__main__":
    dim = (500, 500)
    num_obs = 7
    num_samples = 1
    obs_type = "rectangle"
    dataset = create_dataset(dim, num_samples, num_obs, obs_type)
    breakpoint()
    

    # dim = (500, 500)
    # num_obs = 7
    # num_samples = 4
    # obs_type = "rectangle"
    # generator = DataGeneration(dim, num_obs, obs_type)
    # generator.generate(num_samples)
    # flagFound = False
    # n = 0
    # # breakpoint()
    # while n < num_samples:
    #     start = generator.dataset['starts'][n]
    #     goal = generator.dataset['goals'][n]
    #     grid = generator.dataset['maps'][n]
    #     # print(f"grid -> {grid}")
    #     # breakpoint()
    #     numIterations = 1000
    #     stepSize = 10
    #     # goalRegion = plt.Circle((goal[0], goal[1]), stepSize, color='b', fill=False)
    #     # fig = plt.figure("RRT Algorithm")
    #     # plt.imshow(grid, cmap='binary')
    #     # plt.plot(start[0], start[1], 'ro')
    #     # plt.plot(goal[0], goal[1], 'bo')
    #     # ax = fig.gca()
    #     # ax.add_patch(goalRegion)
    #     rrt = RRTAlgoritm(start, goal, numIterations, grid, stepSize)
    #     for i in range(rrt.iterations):
    #         rrt.resetNearestValue()
    #         point = rrt.sampleAPoint()
    #         rrt.findNearest(rrt.randomTree, point)
    #         new = rrt.steerToPoint(rrt.nearestNode, point)
    #         bool = rrt.isInObstacle(rrt.nearestNode, new)
    #         if (bool == False):
    #             rrt.addChild(new[0], new[1])
    #             # plt.plot([rrt.nearestNode.locationX, new[0]], [rrt.nearestNode.locationY, new[1]], 'go', linestyle='--')
    #             if (rrt.goalFound(new)):
    #                 print(f"itr +1 {n}")
    #                 rrt.addChild(goal[0], goal[1])
    #                 flagFound = True
    #                 n += 1
    #                 print('Goal found!')
    #                 break
    #     if flagFound == False:
    #         continue
    #     else:
    #         try: 
    #             rrt.retraceRRTPath(rrt.goal)
    #             rrt.Waypoints.insert(0, start)
    #             # print(f"Number of waypoints -> {rrt.numWaypoints}")
    #             # print(f"path -> {rrt.Waypoints}")
    #             for i in range(rrt.numWaypoints - 1):
    #                 pass
    #                 # plt.plot([rrt.Waypoints[i][0], rrt.Waypoints[i+1][0]], [rrt.Waypoints[i][1], rrt.Waypoints[i+1][1]], 'ro', linestyle='--')
    #                 # plt.plot([rrt.Waypoints[-1][0], goal[0]], [rrt.Waypoints[-1][1], goal[1]], 'ro', linestyle='--')
    #                 # plt.pause(0.01)
    #         except:
    #             print("There is no possible path") 
    #         flagFound = False
    #     breakpoint()
    #     # plt.close()
    #     # plt.close()
