import cv2
import sys
sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner
import numpy as np

class PlannerAStar(Planner):
    def __init__(self, m, inter=5):
        super().__init__(m)
        self.inter = inter
        self.initialize()

    def initialize(self):
        self.queue = []
        self.finished = []
        self.parent = {}
        self.h = {} # Distance from start to node
        self.g = {} # Distance from node to goal
        self.goal_node = None

    def planning(self, start=(100,200), goal=(375,520), inter=None, img=None):
        if inter is None:
            inter = self.inter
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        # Initialize 
        self.initialize()
        self.queue.append(start)
        self.parent[start] = None
        self.g[start] = 0
        self.h[start] = utils.distance(start, goal)


        while(len(self.queue) > 0):
            # TODO: A Star Algorithm

            # find the current node from the nodes that has been visited with the lowest cost
            current_node = self.queue[0]
            
            for item in self.queue:
                current_f = self.g[current_node] + self.h[current_node]
                item_f = self.g[item] + self.h[item]
                if item_f < current_f:
                    current_node = item
            
            # take the current node away from queue
            self.queue.remove(current_node)

            # if the current node is the goal, break the loop
            if utils.distance(current_node, goal) <= 20:
                self.goal_node = current_node
                break

            # go through the adjacent nodes and add them to queue
            children_node = []
            for new_pos in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                near_node = (int(current_node[0] + inter*new_pos[0]), int(current_node[1] + inter*new_pos[1]))
                
                # skip the node if it is out of the image or is wall
                if near_node[1]<0 or near_node[1]>=self.map.shape[0] or near_node[0]<0 or near_node[0]>=self.map.shape[1]:
                    continue
                
                elif self.map[int(near_node[1]),int(near_node[0])] == 0:
                    continue
                else:
                    children_node.append(near_node)

            for child_node in children_node:
                G = self.g[current_node] + utils.distance(current_node, child_node)
                
                if child_node in self.queue:
                    if self.g[child_node] <= G:
                        continue

                elif child_node in self.finished:
                    if self.g[child_node] <= G:
                        continue
                    self.finished.remove(child_node)
                    self.queue.append(child_node)

                else:
                    self.queue.append(child_node)
                    self.h[child_node] = utils.distance(child_node, goal)

                self.g[child_node] = G
                self.parent[child_node] = current_node
            
            
            self.finished.append(current_node)
        
        # Extract path
        path = []
        p = self.goal_node

        if p is None:
            return path
        while(True):
            path.insert(0,p)
            if self.parent[p] is None:
                break
            p = self.parent[p]
        if path[-1] != goal:
            path.append(goal)
        return path
