import numpy as np
import sys
from functools import reduce
from operator import add
import random
import math

random.seed()


class EnvNMoku:
    def __init__(self, verbose=False, opponent_epsilon=0.3, opponent_agent=None, n_moku=3, board_size=3):
        self.n_moku = n_moku
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size, 2))
        self.n_actions = self.board_size * self.board_size
        self.verbose = verbose
        self.opponent_epsilon = opponent_epsilon
        self.opponent_agent = opponent_agent

        self.PLAYER_AGENT = 0
        self.PLAYER_OPPONENT = 1

    '''
    エージェントからactionを受け取り、その行動後の状態や報酬を返す。
    1. エージェントから受け取ったactionに従って、石を置いて、勝敗のチェックをする
    2. 対戦相手が
    '''
    def step(self, action):
        x, y = self.action_to_xy(action)
        if self.verbose:
            print("agent's move: {}, {}".format(x, y))
        info = None

        if not self.is_valid_move([x, y]):
            if self.verbose:
                print("not a valid move!")
            reward = -1
            done = True
            return self.get_obs(), reward, done, info

        # actionに応じて石を置く
        self.board[y][x][self.PLAYER_AGENT] = 1

        # 勝敗をチェックする
        agent_win = self.check_finish((x, y))
        if agent_win:
            reward = 1
            done = True
            return self.get_obs(), reward, done, info
        elif not self.is_ok_to_continue():
            reward = 0.5
            done = True
            return self.get_obs(), reward, done, info

        # 対戦相手が石を置く
        if self.opponent_agent is not None and random.random() > self.opponent_epsilon:
            opponent_action = self.opponent_agent.act(self.get_obs_inverse())
            x, y = self.action_to_xy(opponent_action)
        else:
            x, y = self.get_move_random()
        if self.verbose:
            print("opponent's move: {}, {}".format(x, y))
        self.board[y][x][self.PLAYER_OPPONENT] = 1

        # 勝敗をチェックする
        opponent_win = self.check_finish((x, y))
        if opponent_win:
            if self.verbose:
                print("opponent won!")
            reward = -1
            done = True
            return self.get_obs(), reward, done, info
        elif not self.is_ok_to_continue():
            reward = 0.5
            done = True
            return self.get_obs(), reward, done, info

        # 勝敗が決まらなかった場合
        reward = -0.001
        done = False
        return self.get_obs(), reward, done, info

    # 環境を初期化して状態を返す
    def reset(self, opponent_first=None):
        if self.verbose:
            print('****** game reset ******')

        # 全てのマスを空にする
        self.board = np.zeros(self.board.shape)

        # 先攻後攻をランダムに決める
        # 後攻の場合は相手が石を置いた状態を初期状態とする
        if opponent_first is None:
            opponent_first = random.choice([True, False])
        if opponent_first:
            if self.opponent_agent is not None and random.random() > self.opponent_epsilon:
                opponent_action = self.opponent_agent.act(self.get_obs_inverse())
                x, y = self.action_to_xy(opponent_action)
            else:
                x, y = self.get_move_random()
            if self.verbose:
                print("opponent's move: {}, {}".format(x, y))
            self.board[y][x][self.PLAYER_OPPONENT] = 1

        return self.get_obs()

    def set_opponent_agent(self, agent):
        self.opponent_agent = agent

    def is_valid_move(self, move):
        return np.sum(self.board[move[1]][move[0]]) == 0

    def action_to_xy(self, action):
        x = action % self.board_size
        y = action // self.board_size
        return x, y

    def xy_to_action(self, x, y):
        return x + y * self.board_size

    def get_obs(self):
        return self.board.flatten()

    def get_obs_inverse(self):
        board_inverse = np.zeros(self.board.shape)
        for y, r in enumerate(self.board):
            for x, c in enumerate(r):
                if c[self.PLAYER_AGENT] == 1:
                    board_inverse[y][x][self.PLAYER_AGENT] = 0
                    board_inverse[y][x][self.PLAYER_OPPONENT] = 1
                elif c[self.PLAYER_OPPONENT] == 1:
                    board_inverse[y][x][self.PLAYER_AGENT] = 1
                    board_inverse[y][x][self.PLAYER_OPPONENT] = 0
        return board_inverse.flatten()

    def get_move_candidates(self):
        candidates = reduce(add, [[(x, y) for y in range(self.board_size) if np.sum(self.board[y][x]) == 0]
                                  for x in range(self.board_size)])
        if self.verbose:
            print(candidates)
        return candidates

    def is_ok_to_continue(self):
        return np.sum(self.board) < self.n_actions

    def check_finish(self, last_move):
        board = self.board
        last_x = last_move[0]
        last_y = last_move[1]

        start_x = last_x + 1 - self.n_moku
        start_y = last_y + 1 - self.n_moku

        row = board[last_y]
        for x in range(start_x, last_x + 1):
            if x < 0:
                continue
            if x + self.n_moku - 1 >= self.board_size:
                break
            line = row[x:x + self.n_moku]
            assert len(line) == self.n_moku
            if self.check_finish_line(line):
                return True

        col = np.transpose(board, axes=(1, 0, 2))[last_x]
        for y in range(start_y, last_y + 1):
            if y < 0:
                continue
            if y + self.n_moku - 1 >= self.board_size:
                break
            line = col[y:y + self.n_moku]
            assert len(line) == self.n_moku
            if self.check_finish_line(line):
                return True

        # diag1
        for x, y in zip(range(start_x, last_x + 1), range(start_y, last_y + 1)):
            if x < 0 or y < 0:
                continue
            if x + self.n_moku - 1 >= self.board_size or y + self.n_moku - 1 >= self.board_size:
                break
            line = [board[y + i][x + i] for i in range(self.n_moku)]
            if self.check_finish_line(line):
                return True

        # diag2
        for x, y in zip(range(start_x, last_x + 1), reversed(range(last_y, last_y + self.n_moku))):
            if x < 0 or y >= self.board_size:
                continue
            if x + self.n_moku - 1 >= self.board_size or y - self.n_moku + 1 < 0:
                break
            line = [board[y - i][x + i] for i in range(self.n_moku)]
            if self.check_finish_line(line):
                return True

        return False

    @staticmethod
    def check_finish_line(line):
        p1win = True
        p2win = True

        for e in line:
            p1win = p1win and e[0] == 1
            p2win = p2win and e[1] == 1

        return p1win or p2win

    def get_move_random(self):
        move_candidates = self.get_move_candidates()
        return random.choice(move_candidates)

    def get_action_random(self):
        move = self.get_move_random()
        return self.xy_to_action(move[0], move[1])

    def render(self):
        print("-----------------------------")
        for r in self.board:
            for c in r:
                if c[0] == 1:
                    sys.stdout.write('o ')
                elif c[1] == 1:
                    sys.stdout.write('x ')
                else:
                    sys.stdout.write('_ ')
            sys.stdout.write('\n')
        sys.stdout.write('\n')
        sys.stdout.flush()

    def render_for_interactive(self):
        print("-----------------------------")
        i = 1
        fill = int(math.log10(self.n_actions) + 1)

        def write_with_padding(message):
            sys.stdout.write('{message: <{fill}} '.format(message=message, fill=fill))
            #sys.stdout.write('{message: <5} '.format(message=message, fill=fill))

        for r in self.board:
            for c in r:
                if c[0] == 1:
                    write_with_padding('o')
                elif c[1] == 1:
                    write_with_padding('x')
                else:
                    write_with_padding(i)
                i += 1
            sys.stdout.write('\n')


def test_interactive(env):
    env.reset()

    def move_player():
        print('your turn:')
        env.render_for_interactive()
        x = None
        y = None
        while x is None:
            sys.stdout.write('choose your move(1-{}):'.format(env.n_actions))
            sys.stdout.flush()
            input = sys.stdin.readline()
            try:
                action_human = int(input) - 1
                if action_human < 0 or env.n_actions - 1 < action_human:
                    raise ValueError
                x, y = env.action_to_xy(action_human)
                if not env.is_valid_move([x, y]):
                    raise ValueError
            except ValueError:
                print("please specify valid value.")

        _, reward, done, _ = env.step(action_human)
        env.render()

        if reward == 1:
            print("Congrats! You Won!")

        if reward == -1:
            print("Agent Won! Keep Trying.")
            return True

        if reward == 0.5:
            print("Draw. Well Done!")
            return True

        return done

    while True:
        done = move_player()
        if done:
            break