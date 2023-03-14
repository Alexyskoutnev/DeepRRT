import numpy as np
import matplotlib.pyplot as plt
import random
import math

from dataset import DataGeneration

from matplotlib.pyplot import rcParams
np.set_printoptions(precision=3, suppress=True)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
plt.rcParams['font.size'] = 22


def load(map, start, goal):
    map[map > 0] = 1
    map[start[0], start[1]] = 0.5
    map[goal[0], goal[1]] = 0.5
    plt.set_cmap('binary')
    plt.imshow(map)
    plt.tight_layout()
    plt.show()


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
        if point[0] >= grid.shape[1]:
            point[0] = grid.shape[1]
        elif point[0] < 0:
            point[0] = 0
        if point[1] >= grid.shape[0]:
            point[1] = grid.shape[1]
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
            if self.grid[round(testPoint[1]), round(testPoint[0])] == 1:
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
        if self.distance(self.goal, point) < stepSize:
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
        print(f"goal - > {goal}")
        if goal.locationX == self.randomTree.locationX:
            return
        self.numWaypoints += 1
        currentPoint = np.array([goal.locationX, goal.locationY])
        self.Waypoints.insert(0, currentPoint)
        self.path_distance += self.rho
        self.retraceRRTPath(goal.parent)

    def expert_policy(self):
        pass


if __name__ == "__main__":
    # grid = np.load('map_space.npy')
    # start = np.array([50.0, 50.0])
    # goal = np.array([500, 220])
    dim = (500, 500)
    num_obs = 7
    num_samples = 3
    obs_type = "rectangle"
    generator = DataGeneration(dim, num_obs, obs_type)
    generator.generate(num_samples)
    flagFound = False
    i = 0
    breakpoint()
    while i < num_samples:
        print(f"ITRS: {i}")
        start = generator.dataset['starts'][i]
        goal = generator.dataset['goals'][i]
        grid = generator.dataset['maps'][i]
        numIterations = 1000
        stepSize = 15
        goalRegion = plt.Circle((goal[0], goal[1]), stepSize, color='b', fill=False)
        fig = plt.figure("RRT Algorithm")
        plt.imshow(grid, cmap='binary')
        plt.plot(start[0], start[1], 'ro')
        plt.plot(goal[0], goal[1], 'bo')
        ax = fig.gca()
        ax.add_patch(goalRegion)
        rrt = RRTAlgoritm(start, goal, numIterations, grid, stepSize)
        for i in range(rrt.iterations):
            rrt.resetNearestValue()
            point = rrt.sampleAPoint()
            rrt.findNearest(rrt.randomTree, point)
            new = rrt.steerToPoint(rrt.nearestNode, point)
            bool = rrt.isInObstacle(rrt.nearestNode, new)
            if (bool == False):
                rrt.addChild(new[0], new[1])
                plt.pause(0.001)
                plt.plot([rrt.nearestNode.locationX, new[0]], [rrt.nearestNode.locationY, new[1]], 'go', linestyle='--')
                # breakpoint()
                if (rrt.goalFound(new)):
                    rrt.addChild(goal[0], goal[1])
                    flagFound = True
                    i += 1
                    print('Goal found!')
                    break
        if flagFound == False:
            continue
        else:
            try: 
                rrt.retraceRRTPath(rrt.goal)
                rrt.Waypoints.insert(0, start)
                # print(f"Number of waypoints -> {rrt.numWaypoints}")
                # print(f"path -> {rrt.Waypoints}")
                for i in range(rrt.numWaypoints- 1):
                    plt.plot([rrt.Waypoints[i][0], rrt.Waypoints[i+1][0]], [rrt.Waypoints[i][1], rrt.Waypoints[i+1][1]], 'ro', linestyle='--')
                    plt.pause(0.01)
            except:
                print("There is no possible path") 
            flagFound = False
        breakpoint()
        plt.close()
