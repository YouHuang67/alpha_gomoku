import random
import numpy as np
from tqdm import tqdm

from .model_wrapper import NetWrapper


class Node(object):

    def __init__(self, prob, key, player, cpuct=2.5, board=None):
        self._prob = prob
        self._prob_n = 1
        self.key = key
        self.player = player
        self.cpuct = cpuct
        self.board = board

        self.visit = 0
        self.value = 0.0
        self.children = dict()
        self.parents = dict()
        self.is_expanded = False
        self.leaf_value = None
        self.attack_action = None

    def evaluate_leaf(self, board, max_num_nodes=100000):
        assert len(self.children) == 0
        if board.is_over:
            if board.winner == 1 - board.player:
                self.leaf_value = -1.0
            else:
                self.leaf_value = 0.0
            return True
        else:
            actions = board.evaluate(max_num_nodes)
            if board.attacker == board.player:
                self.leaf_value = 1.0
                self.attack_action = actions[0]
                return True
            return False

    def expand(self, probs, board, node_table):
        assert not self.is_expanded
        for action, prob in probs.items():
            self.children[action] = \
                self.get_node(board, node_table, action, prob, self.cpuct)
        self.is_expanded = True

    def forward(self, board, parent=None, depth=0):
        self.visit += 1
        if parent is not None:
            self.parents[parent.key] = parent
        if not self.is_expanded:
            return self, depth
        scores = self.get_scores()
        best_action, best_score = random.choice(list(scores.items()))
        for action, score in scores.items():
            if score > best_score:
                best_action = action
                best_score = score
        return self.children[best_action].forward(board.move(best_action),
                                                  self, depth + 1)

    def backward(self, value, depth):
        self.value += value
        if depth:
            for key in list(self.parents.keys()):
                self.parents.pop(key).backward(-value, depth - 1)

    @property
    def is_leaf(self):
        return self.leaf_value is not None

    @property
    def prob(self):
        return self._prob / self._prob_n

    @property
    def score(self):
        return self.value / self.visit if self.visit else 0.0

    def visit_bonus(self, total_visit):
        return self.prob * total_visit ** 0.5 / (1 + self.visit)

    def get_scores(self):
        total_visit = sum(node.visit for node in self.children.values()) + 1
        return {action: -node.score + node.cpuct * node.visit_bonus(total_visit)
                for action, node in self.children.items()}

    def update_prob(self, prob):
        self._prob += prob
        self._prob_n += 1
        return self

    @classmethod
    def get_node(cls, board, node_table, action=None, prob=1.0, cpuct=2.5):
        if action is None:
            key = board.key
            player = board.player
            board = board.copy()
        else:
            key = board.next_key(action)
            player = 1 - board.player
            board = None
        return node_table.setdefault(key, cls(prob, key, player, cpuct, board))


class MonteCarloTreeSearch(object):

    def __init__(self, model, num_iters=200, cpuct=2.5,
                 max_num_nodes=100000, verbose=1, gap=3):
        self.model = NetWrapper(model, gap).eval()
        self.num_iters, self.cpuct = num_iters, cpuct
        self.max_num_nodes, self.verbose = max_num_nodes, verbose
        self.node_table = dict()
        self.visit_table = dict()

    def evaluate(self, board, description=''):
        copy_board = board.copy()
        actions = copy_board.evaluate(self.max_num_nodes)
        if len(actions) and copy_board.attacker == copy_board.player:
            return self.visit_table.setdefault(board.key, {actions[0]: 1})

        root = Node.get_node(board, self.node_table, cpuct=self.cpuct)
        root.board = board
        iters = range(self.num_iters)
        iters = tqdm(iters, description) if self.verbose else iters
        for _ in iters:
            board = root.board.copy()
            node, depth = root.forward(board)

            max_num_nodes = self.max_num_nodes
            if root not in list(node.parents.values()) + [node]:
                max_num_nodes //= 10
            if node.is_leaf or node.evaluate_leaf(board.copy(), max_num_nodes):
                node.backward(node.leaf_value, depth)
                continue

            probs, value = self.model(board)
            node.expand(probs, board, self.node_table)
            node.backward(value, depth)

        return self.visit_table.setdefault(root.key,
                                           {act: int(node.visit) for act, node
                                            in root.children.items()})

    @staticmethod
    def sample_actions(visit_table, tau=0.0):
        actions, visits = list(zip(*list(visit_table.items())))
        if tau == 0.0:
            return actions[np.argmax(visits)]
        else:
            visits = [float(v) ** (1.0 / tau) for v in visits]
            total_visits = sum(visits)
            probs = [v / total_visits for v in visits]
            rand, s = random.random(), 0.0
            for i, p in enumerate(probs):
                if s <= rand < s + p:
                    return actions[i]
                s += p

    def search(self, board, description='', tau=0.0):
        return self.sample_actions(self.evaluate(board, description), tau)
