import random
from tqdm import tqdm
from pathlib import Path

import torch

from alpha_gomoku.cppboard import Board
from alpha_gomoku.explore import models
from alpha_gomoku.explore.ensemble_regular_resnet_v1_0 import BoardWrapper
from alpha_gomoku.mcts.evaluator import Evaluator


class Node(object):

    def __init__(self, board, prob, cpuct=1.0):
        self.board = board
        self._prob = prob
        self._prob_n = 1
        self.cpuct = cpuct
        self.player = board.player
        self.key = board.key
        self.level = len(board.history)
        self.reset()

    def reset(self):
        self.visit = 0
        self.value = 0.0
        self.children = dict()
        self.parents = dict()
        self.is_expanded = False

    def expand(self, nodes):
        assert not self.is_expanded
        for action, node in nodes.items():
            self.children[action] = node
        self.is_expanded = True

    def forward(self, parent=None, depth=0):
        self.visit += 1
        if parent is not None:
            self.parents[parent.key] = parent
        if not self.is_expanded or len(self.children) == 0:
            return self, depth
        scores = self.get_scores()
        best_action, best_score = random.choice(list(scores.items()))
        for action, score in scores.items():
            if score > best_score:
                best_action = action
                best_score = score
        return self.children[best_action].forward(self, depth + 1)

    def backward(self, value, depth):
        if depth:
            for parent in self.parents.values():
                parent.backward(-value, depth - 1)

    @property
    def prob(self):
        return self._prob / self._prob_n

    @property
    def score(self):
        return self.value / self.visit if self.visit else 0.0

    @property
    def visit_bonus(self):
        return self.prob * self.visit ** 0.5 / (1 + self.visit) \
            if self.visit else self.prob

    def get_scores(self):
        return {
            action: -node.score + node.cpuct * node.visit_bonus
            for action, node in self.children.items()
        }

    def update_prob(self, prob):
        self._prob += prob
        self._prob_n += 1
        return self

    def detach(self):
        for child in self.children.values():
            child.parents.pop(self.key, None)
        return self.key


class Tree(object):

    def __init__(
        self, evaluator, visit_times=200,
        board=None, cpuct=1.0, verbose=1,
        max_node_num=100000
    ):
        self.evaluator = evaluator
        self.visit_times = visit_times
        self.root = Node(Board() if board is None else board, 1.0, cpuct)
        self.nodes = {self.root.key: self.root}
        self.cpuct = cpuct
        self.verbose = verbose
        self.max_node_num = max_node_num

    def evaluate(self):
        root = self.root
        board = root.board
        actions = board.copy().evaluate(self.max_node_num)
        if len(actions) and board.attacker == board.player:
            return random.choice(actions)

        evaluator = self.evaluator
        if not root.is_expanded:
            root.expand(self.expand(board, evaluator(board)[0]))

        iterator = range(self.visit_times)
        if self.verbose:
            iterator = tqdm(iterator, 'evaluating', ncols=80)
        for _ in iterator:
            node, depth = root.forward()
            board = node.board
            if node.is_expanded:
                if board.player in [board.attacker, board.winner]:
                    node.backward(1, depth)
                else:
                    node.backward(0, depth)
                continue

            max_node_num = self.max_node_num
            if root not in node.parents.values():
                max_node_num //= 10
            actions = board.copy().evaluate(max_node_num)
            if len(actions):
                if board.attacker == board.player:
                    probs, value = dict(), 1.0
                else:
                    probs, value = evaluator(board, actions)
            else:
                probs, value = evaluator(board)
            node.expand(self.expand(board, probs))
            node.backward(value, depth)
        children = list(root.children.items())
        random.shuffle(children)
        return sorted(children, key=lambda x: x[1].visit)[-1][0]

    def expand(self, board, probs):
        nodes = dict()
        for action, prob in probs.items():
            board_copy = board.copy().move(action)
            key = board_copy.key
            if key in self.nodes:
                self.nodes[key].update_prob(prob)
            else:
                self.nodes[key] = Node(board_copy, prob, self.cpuct)
            nodes[action] = self.nodes[key]
        return nodes

    def move(self, action):
        root = self.root
        if action in root.children:
            self.root = root.children[action]
        else:
            board = root.board.copy().move(action)
            self.root = self.nodes.setdefault(board.key, Node(board, 1.0, self.cpuct))
        return self


def main():
    board = Board()
    model = models.ValueWrapper(BoardWrapper(models.SEWideResnet16_2()))
    # weight_path = Path(__file__).parents[1] / 'data/weights/explore/ensemble/regular_resnet/v1/sewideresnet16_2.pth'
    # model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    evaluator = Evaluator(model)
    tree = Tree(evaluator, visit_times=200, board=board, verbose=1)
    while not board.is_over:
        print(board)
        action = tree.evaluate()
        print(board.player, action)
        board.move(action)
        tree.move(action)
    print(board)


if __name__ == '__main__':
    main()







