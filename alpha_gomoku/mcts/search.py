import math
import random
from tqdm import tqdm
from queue import Queue

from ..cppboard import Board
from alpha_gomoku import cppboard


class Node(object):
    Cpuct = 1.0
    
    def __init__(self, board, prob):
        self.board = board
        self.prob = prob
        self.key = board.key
        self.player = board.player
        self.reset()
    
    def reset(self):
        self.visit = 0
        self.value = 0.0
        self.children = dict()
        self.parents = dict()
        self.expanded = False
        
    def expand(self, nodes):
        assert not self.expanded
        for action, node in nodes.items():
            self.children[action] = node
        self.expanded = True
        
    def forward(self, action, parent):
        self.parents[action] = parent
        if not self.expanded:
            return self
        scores = self.get_scores()
        best_action, best_score = random.choice(scores.items())
        for action, score in scores.items():
            if score > best_score:
                best_action = action
                best_score = score
        best_child = self.children[best_action]
        return best_child(best_action, self)
    
    def update(self, player, value):
        sign = 1 if player == self.player else -1
        self.value += sign * value
    
    @property
    def Q(self):
        if self.visit:
            return self.value / self.visit
        return 0.0
        
    def get_scores(self):
        assert self.expanded
        scores = dict()
        for action, node in self.children.items():
            u = self.__class__.Cpuct * \
                node.prob * math.sqrt(self.visit) / (1 + node.visit)
            scores[action] = -node.Q + u
        return scores
    
    
class MonteCarloTreeSearch(object):
    nodes = dict()
    
    def __init__(self, board, evaluator):
        self.board = board
        self.evaluator = evaluator
        self.root = self.get_node(board)
    
    def backward(self, node, value):
        queue = Queue()
        nodes = set()
        queue.put(node)
        nodes.add(node)
        player = node.player
        while not queue.empty():
            node = queue.get()
            node.update(player, value)
            for node in node.parents.values():
                if node not in nodes:
                    queue.put(node)
                    nodes.add(node)
                    
    def develop(self, node):
        board = node.board
        probs, value = self.evaluator(board)
        nodes = dict()
        for action, prob in probs.items():
            copy_board = board.copy()
            copy_board.move(action)
            nodes[action] = self.get_node(copy_board, prob)
        if len(nodes):
            node.expand(nodes)
        return value
                    
    @classmethod
    def get_node(cls, board, prob=0.0):
        return cls.nodes.setdefault(board.key, Node(board, prob))
    
    @classmethod
    def clear_table(cls):
        cls.nodes.clear()
        
    @classmethod
    def clear_parents(cls):
        for node in cls.values():
            node.parents.clear()
        
        
        
            
    
            
        
    
    