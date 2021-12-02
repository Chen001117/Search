import numpy as np
import queue
import copy
import pygame

COLOR = [
    "black", "blue", "yellow", "pink", "red", "brown", "purple", "green", "orange", "gray", "white"
]

TRUE_VALUE_TABLE = dict()
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
GROUP_NUM = 10
EPOCHS = 100
GAMMA = 0.99
ETA = 0.4
MAX_EPSILON = 0.95
MIN_EPSILON = 0.05

class UI:
    def __init__(self):
        size = SCREEN_WIDTH, SCREEN_HEIGHT 
        self.screen = pygame.display.set_mode(size) 
        self.clock = pygame.time.Clock()
        btn1 = Button(1,"       start       ",(130, 80),font=30,bg="white")
        btn2 = Button(2,"width = 05",(120, 130),font=30,bg="white")
        btn3 = Button(3,"+",(250, 130),font=30,bg="white")
        btn4 = Button(4,"-",(270, 130),font=30,bg="white")
        btn5 = Button(5," pairs = 06",(120, 230),font=30,bg="white")
        btn6 = Button(6,"+",(250, 230),font=30,bg="white")
        btn7 = Button(7,"-",(270, 230),font=30,bg="white")
        btn8 = Button(8,"block = 05",(120, 280),font=30,bg="white")
        btn9 = Button(9,"+",(250, 280),font=30,bg="white")
        btn10 = Button(10,"-",(270, 280),font=30,bg="white")
        btn11 = Button(11,"A*",(100, 330),font=30,bg="red")
        btn12 = Button(12," A ", (140, 330),font=30,bg="white")
        btn13 = Button(13,"MC",(175, 330),font=30,bg="white")
        btn14 = Button(14,"TD",(225, 330),font=30,bg="white")
        btn15 = Button(15,"DP",(270, 330),font=30,bg="white")
        btn16 = Button(16,"group = 05",(120, 180),font=30,bg="white")
        btn17 = Button(17,"+",(250, 180),font=30,bg="white")
        btn18 = Button(18,"-",(270, 180),font=30,bg="white")
        self.btns = [btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8, btn9, \
            btn10, btn11, btn12, btn13, btn14, btn15, btn16, btn17, btn18]
    
    def DrawRect(self, x, y, result):
        board = result.board
        for row in range(board.width+2):
            for col in range(board.width+2):
                color = COLOR[int(board.board[row][col])]
                pos_x = SCREEN_WIDTH/(board.width+2)*row
                pos_y = SCREEN_HEIGHT/(board.width+2)*col
                len_x = SCREEN_WIDTH/(board.width+2)-2
                len_y = SCREEN_HEIGHT/(board.width+2)-2
                position = (pos_x, pos_y, len_x, len_y)
                width = 0
                pygame.draw.rect(self.screen, color, position, width)
    
    def PrintText(self, result):
        font = pygame.font.Font("freesansbold.ttf", 20)
        text = 'score:{:03d}'.format(int(-result.cost))
        text = font.render(text, True, COLOR[-1], COLOR[0])
        self.screen.blit(text, (100, 10))    

class Button:
    """Create a button, then blit the surface in the while loop"""
 
    def __init__(self, idx, text, pos, font, bg="black"):
        self.idx = idx
        self.x, self.y = pos
        self.font = pygame.font.SysFont("Corbel", font)
        self.bg = bg
        self.my_text = text
        self.change_text(text, bg)
 
    def change_text(self, text, bg="black"):
        """Change the text whe you click"""
        self.text = self.font.render(text, 1, pygame.Color("Black"))
        self.size = self.text.get_size()
        self.surface = pygame.Surface(self.size)
        self.surface.fill(bg)
        self.surface.blit(self.text, (0, 0))
        self.rect = pygame.Rect(self.x, self.y, self.size[0], self.size[1])
 
    def show(self, screen):
        screen.blit(self.surface, (self.x, self.y))
 
    def click(self, event, info, btns):
        x, y = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_pressed()[0]:
                if self.rect.collidepoint(x, y):
                    algo = ["Astar", " A ", "MC", "TD", "DP"]
                    if self.idx == 1:
                        btns[0].change_text("searching...", bg="white")
                        return True
                    elif self.idx == 3:
                        info["width"] += 1
                    elif self.idx == 4:
                        info["width"] -= 1
                    elif self.idx == 6:
                        info["pairs"] += 1
                    elif self.idx == 7:
                        info["pairs"] -= 1
                    elif self.idx == 9:
                        info["block"] += 1
                    elif self.idx == 10:
                        info["block"] -= 1
                    elif self.idx == 17:
                        info["group"] += 1
                    elif self.idx == 18:
                        info["group"] -= 1
                    elif self.idx in range(11, 16):
                        info["algo"] = algo[self.idx-11]
                    btns[1].change_text("width = {:02d}".format(info["width"]), btns[1].bg)
                    btns[4].change_text(" pairs = {:02d}".format(info["pairs"]), btns[4].bg)
                    btns[7].change_text("block = {:02d}".format(info["block"]), btns[7].bg)
                    btns[15].change_text("group = {:02d}".format(info["group"]), btns[15].bg)
                    for aid, btn in enumerate(btns[10:15]):
                        if algo[aid] == info["algo"]:
                            btn.change_text(btn.my_text, bg="red")
                        else:
                            btn.change_text(btn.my_text, bg="white")
        return False            

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
    def __init__(self, board, parent=None, reward=None, use_h=True):
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
        return cnt

    def make_child(self, board, reward):
        return Node(board, self, reward)
        
    def __lt__(self, other):
        return self.fph > other.fph

def TD(root):
    value_table = dict()
    # training 
    for epoch in range(EPOCHS):
        buffer = []
        epsilon = MIN_EPSILON + (MAX_EPSILON-MIN_EPSILON) * epoch / EPOCHS
        node = root
        while not node.board.finish():
            actions = node.board.avail_action()
            while True:
                values = np.zeros(len(actions))
                random = np.random.random(len(actions))
                for aid, action in enumerate(actions):
                    key = str(node.board.board)+str(action)
                    if key in value_table:
                        values[aid] = value_table[key]
                if np.random.rand() < epsilon:
                    max_idx = np.random.randint(len(actions))
                else:
                    max_idx = np.argmax(values+random)
                new_board, reward = node.board.step(actions[max_idx])
                if reward is None:
                    key = str(node.board.board)+str(actions[max_idx])
                    value_table[key] = -1e6
                else:
                    new_value = -1e6
                    for action in new_board.avail_action():
                        key = str(new_board.board)+str(action)
                        if key in value_table:
                            if value_table[key]>new_value:
                                new_value = value_table[key]
                    new_value = 0 if new_value<-1e5 else new_value
                    obj = reward + GAMMA * new_value
                    key = str(node.board.board)+str(actions[max_idx])
                    if key in value_table:
                        value_table[key] = value_table[key] + ETA * (obj-value_table[key])
                    else:
                        value_table[key] = ETA * obj
                    node = node.make_child(new_board, reward)
                    break
    # evaluating           
    node = root
    while not node.board.finish():
        actions = node.board.avail_action()
        values = np.zeros(len(actions))-1e6
        for aid, action in enumerate(actions):
            key = str(node.board.board)+str(action)
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
    # training
    for epoch in range(EPOCHS):
        buffer = []
        epsilon = MIN_EPSILON + (MAX_EPSILON-MIN_EPSILON) * epoch / EPOCHS
        node = root
        while not node.board.finish():
            actions = node.board.avail_action()
            while True:
                values = np.zeros(len(actions))
                random = np.random.random(len(actions))
                for aid, action in enumerate(actions):
                    key = str(node.board.board)+str(action)
                    if key in value_table:
                        values[aid] = value_table[key]
                if np.random.rand() < epsilon:
                    max_idx = np.random.randint(len(actions))
                else:
                    max_idx = np.argmax(values+random)
                new_board, reward = node.board.step(actions[max_idx])
                if reward is None:
                    key = str(node.board.board)+str(actions[max_idx])
                    value_table[key] = -1e6
                else:
                    buffer.append([node, actions[max_idx], reward])
                    node = node.make_child(new_board, reward)
                    break
        value = 0
        for i, data in enumerate(buffer[::-1]):
            value = value * GAMMA + data[2]
            key = str(data[0].board.board)+str(data[1])
            if key in value_table:
                value_table[key] = value_table[key] + ETA * (value-value_table[key])
            else:
                value_table[key] = ETA * value   
    # evaluating   
    node = root
    while not node.board.finish():
        actions = node.board.avail_action()
        values = np.zeros(len(actions))-1e6
        for aid, action in enumerate(actions):
            key = str(node.board.board)+str(action)
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
    openlist = queue.PriorityQueue()
    closelist = set()
    openlist.put(root)
    while not openlist.empty():
        node = openlist.get()
        if str(node.board.board) in closelist:
            continue
        else:
            closelist.add(str(node.board.board))
        if node.board.finish():
            return node
        for action in node.board.avail_action():
            new_board, reward = node.board.step(action)
            if reward is not None:
                openlist.put(Node(new_board, node, reward, use_h=star))
    return None
         
if __name__ == "__main__":
    # UI
    pygame.init() 
    game = UI()
    
    info = {"width":5, "group":5, "pairs":6, "block":5, "algo": "Astar"}
    start = False
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            for btn in game.btns:
                start |= btn.click(event, info, game.btns)
        for btn in game.btns:
            btn.show(game.screen)
        game.clock.tick(30)
        pygame.display.update()
    # prepare
    root = Node(Board(width=info["width"], group=info["group"], pairs=info["pairs"], block=info["block"]))

    if info["algo"] == "Astar":
        result = A_star(root, star=True)
    elif info["algo"] == " A ":
        result = A_star(root, star=False)
    elif info["algo"] == "MC":
        result = MC(root)
    elif info["algo"] == "TD":
        result = TD(root)
    elif info["algo"] == "DP":
        result, max_val = DP(root)

    ans = []
    while result:
        ans.append(result)
        result = result.parent

    while True:       
        for result in ans[::-1]:
            game.clock.tick(1)   
            game.screen.fill((255, 255, 255)) 
            game.DrawRect(100, 100, result)
            game.PrintText(result)
            pygame.display.update() 
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT:
                    pygame.quit()                      
             
        