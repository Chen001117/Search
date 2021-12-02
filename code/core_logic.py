import numpy as np
import queue
import copy
import sys
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

COLOR = [
    "black", "blue", "yellow", "pink", "red", "brown", "purple", "green", "orange", "gray", "white"
]

TRUE_VALUE_TABLE = dict()
ALGO_VALUE_TABLE = dict()
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
EPOCHS = 2000
GAMMA = 0.95
ETA = 0.1
MAX_EPSILON = 0.95
MIN_EPSILON = 0.05
INIT = -50


class Board():
    def __init__(self, width=0, group=0, pairs=0, block=0, seed_num=0):
        np.random.seed(seed_num)
        self.group = group
        self.width = width
        self.size = (width+2) ** 2
        self.space = (self.width)**2-pairs*2-block
        self.block = block
        len = width**2 - self.space - block
        idx = np.random.randint(1, group+1, [len//2])
        board = np.concatenate([idx, idx.copy(), [0]*self.space, [-1]*self.block])
        np.random.shuffle(board)
        board = board.reshape([self.width, self.width])
        self.board = np.zeros([self.width+2, self.width+2])
        self.board[1:-1, 1:-1] = board
    
    def avail_action(self):
        actions = []
        for group in range(1, self.group+1):
            idx = np.where(self.board==group)
            for i in range(idx[0].shape[0]):
                for j in range(i+1, idx[0].shape[0]):
                    idx1 = [idx[0][i], idx[1][i]]
                    idx2 = [idx[0][j], idx[1][j]]
                    actions.append([idx1, idx2])
        return actions                

    def step(self, action):
        board = self.board.copy()
        board[board!=0] = -1
        board[action[1][0], action[1][1]] = 0
        board[action[0][0], action[0][1]] = 1
        nodes = queue.Queue()
        nodes.put(action[0])
        while not nodes.empty():
            node = nodes.get()
            if node == action[1]:
                reward = 2 - board[action[1][0], action[1][1]]
                board = copy.deepcopy(self)
                board.board[action[0][0], action[0][1]] = 0
                board.board[action[1][0], action[1][1]] = 0
                return board, reward
            for i in range(node[0]+1, self.width+2):
                if board[i, node[1]] == 0:
                    nodes.put([i, node[1]])
                    board[i, node[1]] = board[node[0], node[1]] + 1
                elif board[i, node[1]] == -1:
                    break
            for i in range(node[0]-1, -1, -1):
                if board[i, node[1]] == 0:
                    nodes.put([i, node[1]])
                    board[i, node[1]] = board[node[0], node[1]] + 1
                elif board[i, node[1]] == -1:
                    break
            for i in range(node[1]+1, self.width+2):
                if board[node[0], i] == 0:
                    nodes.put([node[0], i])
                    board[node[0], i] = board[node[0], node[1]] + 1
                elif board[node[0], i] == -1:
                    break
            for i in range(node[1]-1, -1, -1):
                if board[node[0], i] == 0:
                    nodes.put([node[0], i])
                    board[node[0], i] = board[node[0], node[1]] + 1
                elif board[node[0], i] == -1:
                    break
        return None, None

    def finish(self):
        return np.sum(self.board==0)==(self.size-self.block)

class Node():
    def __init__(self, board, parent=None, reward=None, use_h=True, h=None):
        self.h = h
        self.value = 0
        self.board = board
        self.parent = parent
        if self.parent:
            self.depth = parent.depth + 1
            self.cost = parent.cost + reward
            self.fph = self.cost - self.get_h(use_h)
        else:
            self.fph = -self.get_h(use_h)
            self.depth = 0
            self.cost = 0
    
    def get_h(self, use_h=True):
        if not use_h:
            return 0
        cnt = 0
        for group in range(1, self.board.group+1):
            idx = np.where(self.board.board==group)
            pair = np.array([0] * len(idx[0]))
            for i in range(idx[0].shape[0]):
                for j in range(i+1, idx[0].shape[0]):
                    if idx[0][i]==idx[0][j] or idx[1][i]==idx[1][j]:
                        pair[i] = 1
                        pair[j] = 1
            cnt += np.sum(pair==0)//2
        if self.h is not None:
            return (1-ALPHA) * cnt + ALPHA * self.h
        return cnt

    def make_child(self, board, reward):
        return Node(board, self, reward)
        
    def __lt__(self, other):
        return self.fph > other.fph

def TD(root):
    value_table = dict()
    
    cost_avg = []
    cost_table = []
    for epoch in range(EPOCHS):
        buffer = []
        epsilon = MAX_EPSILON * np.exp(-epoch*20/EPOCHS)
        epsilon = np.maximum(MIN_EPSILON, epsilon)
        node = root
        while not node.board.finish():
            actions = node.board.avail_action()
            while True:
                values = np.zeros(len(actions))
                random = np.random.random(len(actions))
                board_str = l2s(node.board.board)
                for aid, action in enumerate(actions):
                    key = board_str + l2s(action)
                    if key in value_table:
                        values[aid] = value_table[key]
                if np.random.rand() < epsilon:
                    max_idx = np.random.randint(len(actions))
                else:
                    max_idx = np.argmax(values+random)
                new_board, reward = node.board.step(actions[max_idx])
                if reward is None:
                    key = l2s(node.board.board)+l2s(actions[max_idx])
                    value_table[key] = -1e6
                else:
                    new_value = -1e6
                    board_str = l2s(node.board.board)
                    for action in new_board.avail_action():
                        key = board_str + l2s(action)
                        if key in value_table:
                            if value_table[key]>new_value:
                                new_value = value_table[key]
                    new_value = 0 if new_value<-1e5 else new_value
                    obj = reward + GAMMA * new_value
                    key = board_str + l2s(actions[max_idx])
                    if key in value_table:
                        value_table[key] = value_table[key] + ETA * (obj-value_table[key])
                    else:
                        value_table[key] = INIT + ETA * (obj-INIT)
                    node = node.make_child(new_board, reward)
                    break

        # evaluate 
        if epoch%10!=0:
            continue

        while not node.board.finish():
            actions = node.board.avail_action()
            values = np.zeros(len(actions))-1e6
            board_str = l2s(node.board.board)
            for aid, action in enumerate(actions):
                key = board_str + l2s(action)
                if key in value_table:
                    values[aid] = value_table[key]
            max_idx = np.argmax(values)
            new_board, reward = node.board.step(actions[max_idx])
            while reward is None:
                max_idx = np.random.randint(len(actions))
                new_board, reward = node.board.step(actions[max_idx])
            node = node.make_child(new_board, reward)
        cost_table.append(-node.cost)
        cost_avg.append((np.array(cost_table).mean()))
        print("epoch", epoch, "cost", -node.cost, "avg",  np.array(cost_table).mean())

    plt.plot(np.arange(len(cost_table)), cost_table)
    plt.plot(np.arange(len(cost_avg)), cost_avg)
    plt.show()  
                    
    node = root
    while not node.board.finish():
        actions = node.board.avail_action()
        values = np.zeros(len(actions))-1e6
        board_str = l2s(node.board.board)
        for aid, action in enumerate(actions):
            key = board_str + l2s(action)
            if key in value_table:
                values[aid] = value_table[key]
        max_idx = np.argmax(values)
        new_board, reward = node.board.step(actions[max_idx])
        while reward is None:
            max_idx = np.random.randint(len(actions))
            new_board, reward = node.board.step(actions[max_idx])
        node = node.make_child(new_board, reward)

    return node

def MC(root):
    value_table = dict()
    
    cost_avg = []
    cost_table = []
    for epoch in range(EPOCHS):
        buffer = []
        epsilon = MAX_EPSILON * np.exp(-epoch*30/EPOCHS)
        epsilon = np.maximum(MIN_EPSILON, epsilon)
        # roll out
        node = root
        while not node.board.finish():
            
            actions = node.board.avail_action()
            while True:
                values = np.zeros(len(actions))
                random = np.random.random(len(actions))
                board_str = l2s(node.board.board)
                for aid, action in enumerate(actions):
                    key = board_str + l2s(action)
                    if key in value_table:
                        values[aid] = value_table[key]
                if np.random.rand() < epsilon:
                    max_idx = np.random.randint(len(actions))
                else:
                    max_idx = np.argmax(values+random)
                new_board, reward = node.board.step(actions[max_idx])
                if reward is None:
                    key = board_str + l2s(actions[max_idx])
                    value_table[key] = -1e6
                else:
                    buffer.append([node, actions[max_idx], reward])
                    node = node.make_child(new_board, reward)
                    break
                          
        # update 
        value = 0
        for i, data in enumerate(buffer[::-1]):
            value = value * GAMMA + data[2]
            key = np.array(data[0].board.board).tobytes()+np.array(data[1]).tobytes()
            if key in value_table:
                value_table[key] = value_table[key] + ETA * (value-value_table[key])
            else:
                value_table[key] = INIT + ETA * (value-INIT)  

        # evaluate 
        if epoch%10!=0:
            continue

        while not node.board.finish():
            actions = node.board.avail_action()
            values = np.zeros(len(actions))-1e6
            board_str = l2s(node.board.board)
            for aid, action in enumerate(actions):
                key = board_str + l2s(action)
                if key in value_table:
                    values[aid] = value_table[key]
            max_idx = np.argmax(values)
            new_board, reward = node.board.step(actions[max_idx])
            while reward is None:
                max_idx = np.random.randint(len(actions))
                new_board, reward = node.board.step(actions[max_idx])
            node = node.make_child(new_board, reward)
        cost_table.append(-node.cost)
        cost_avg.append((np.array(cost_table).mean()))
        print("epoch", epoch, "cost", -node.cost, "avg",  np.array(cost_table).mean())
    

    plt.plot(np.arange(len(cost_table)), cost_table)
    plt.plot(np.arange(len(cost_avg)), cost_avg)
    plt.show() 


    node = root
    while not node.board.finish():
        actions = node.board.avail_action()
        values = np.zeros(len(actions))-1e6
        board_str = to_str(node.board.board)
        for aid, action in enumerate(actions):
            key = to_str(action)
            if key in value_table:
                values[aid] = value_table[key]
        max_idx = np.argmax(values)
        new_board, reward = node.board.step(actions[max_idx])
        while reward is None:
            max_idx = np.random.randint(len(actions))
            new_board, reward = node.board.step(actions[max_idx])
        node = node.make_child(new_board, reward)

    return node

def DP(root):
    if root.board.finish():
        return root, 0
    max_value = -1e6
    best_child = None
    for action in root.board.avail_action():
        key = str(root.board.board)+str(action)
        new_board, reward = root.board.step(action)
        if reward is None:
            TRUE_VALUE_TABLE[key] = -1e6
            continue
        child, value = DP(Node(new_board, root, reward))
        TRUE_VALUE_TABLE[key] = value + reward
        best_child = child if value+reward > max_value else best_child
        max_value = value+reward if value+reward > max_value else max_value
    return best_child, max_value

def A_star(root, star=True):
    global CNT
    CNT = 0
    openlist = queue.PriorityQueue()
    closelist = set()
    openlist.put(root)
    while not openlist.empty():
        node = openlist.get()
        if str(node.board.board) in closelist:
            continue
        else:
            closelist.add(str(node.board.board))
        CNT += 1
        if node.board.finish():
            return node
        board_str = l2s(node.board.board)
        for action in node.board.avail_action():
            new_board, reward = node.board.step(action)
            if reward is not None:
                h_val = TRUE_VALUE_TABLE[board_str+l2s(action)]
                openlist.put(Node(new_board, node, reward, use_h=star, h=-h_val))
    return None
      
def l2s(data):
    return np.array(data).tobytes()

if __name__ == "__main__":

    info = {
        "width":15, 
        "group":15, 
        "pairs":56, 
        "block":56, 
        "algo": "Astar"
    }
    
    for seed in range(3):

        root = Node(Board(width=info["width"], group=info["group"], \
                pairs=info["pairs"], block=info["block"], seed_num=seed))

        print("start searching...")
        result = TD(root)
        print("TD", result.cost)

 
             
        