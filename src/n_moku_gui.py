import tkinter as tk
import tkinter.messagebox as tkmsg
import numpy as np
from enum import IntEnum
import random


class PlayerStatus(IntEnum):
    INITIAL = 0
    AGENT = 1
    PLAYER = 2


class Dispatcher:
    def __init__(self):
        self.events = {}
        self.root = None
        self.event_stack = []

    def set_root(self, root: tk.Widget):
        self.root = root

    def trigger(self, event_name, delay=1, **kwargs):
        self.event_stack.append([event_name, kwargs])
        self.root.after(delay, self.do_trigger)

    def do_trigger(self):
        event = self.event_stack.pop()
        event_name = event[0]
        kwargs = event[1]

        for f in self.events[event_name]:
            f(**kwargs)

    def on(self, event_name, func):
        if event_name not in self.events:
            self.events[event_name] = []

        self.events[event_name].append(func)


dispatcher = Dispatcher()


class StoneButton(tk.Button):
    def __init__(self, parent: tk.Widget, x: int, y: int, val: tk.IntVar, hl: tk.BooleanVar, *args: object, **kwargs: object) -> object:
        tk.Button.__init__(self, parent, command=self.on_click, text="", *args, **kwargs, font=("", 60))
        self.x = x
        self.y = y
        val.trace('w', self.render)
        self.val = val
        hl.trace('w', self.render)
        self.highlight = hl

    def on_click(self):
        dispatcher.trigger('button_pressed', x=self.x, y=self.y)


    def render(self, *args):
        status = self.val.get()
        color = "#000"

        if status == int(PlayerStatus.INITIAL):
            text = ""
        elif status == int(PlayerStatus.AGENT):
            text = "○"
            if self.highlight.get():
                color = "#f55"
        else:
            text = "×"

        self.config(text=text, fg=color)


class Board(tk.Frame):
    def __init__(self, parent: tk.Widget, width: int, height: int, *args: object, **kwargs: object) -> object:
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.board = []
        self.width = width
        self.height = height
        self.n_moves = 0

        for x in range(width):
            board_row = []
            for y in range(height):
                stone_val = tk.IntVar()
                stone_val.set(0)

                stone_highlight = tk.BooleanVar()
                stone_highlight.set(False)

                button = StoneButton(self, x, y, stone_val, stone_highlight, width=3, height=1)
                button.grid(row=x, column=y)

                board_row.append([stone_val, stone_highlight])
            self.board.append(board_row)

    def initialize_state(self):
        for x in range(self.width):
            for y in range(self.height):
                self.board[x][y][0].set(int(PlayerStatus.INITIAL))
                self.board[x][y][1].set(False)

    def get_state(self):
        result = np.zeros([self.height, self.width, 2])
        for x in range(self.width):
            for y in range(self.height):
                val = self.board[x][y][0].get()
                if val > 0:
                    result[y][x][val - 1] = 1
        return result

    def set_state(self, x, y, status):
        val = int(status)

        for x_ in range(self.width):
            for y_ in range(self.height):
                self.board[x_][y_][1].set(False)

        self.board[x][y][1].set(True)
        self.board[x][y][0].set(val)


class NMokuGui:
    def __init__(self, board_size, n_moku, dirname, env, agent):
        self.board_size = board_size
        self.n_moku = n_moku
        self.dirname = dirname
        self.indicator = None
        self.board = None
        self.env = env
        self.agent = agent
        self.game_over = False

        dispatcher.on("players_turn", self.wait_for_player)
        dispatcher.on('button_pressed', self.on_player_move)
        dispatcher.on("agents_turn", self.move_agent)
        dispatcher.on("game_start", self.on_game_start)
        dispatcher.on("game_end", self.on_game_end)

    def on_game_start(self):
        self.game_over = False
        self.env.reset(opponent_first=False)
        self.board.initialize_state()

        players_turn = random.choice([True, False])

        if players_turn:
            dispatcher.trigger("players_turn")
        else:
            dispatcher.trigger("agents_turn")

    def wait_for_player(self):
        pass

    def on_player_move(self, *, x, y):
        if self.game_over:
            return

        env = self.env

        if not env.is_valid_move([x, y]):
            print("*** Player misplaced his stone to ({}, {})".format(x, y))
            dispatcher.trigger("game_end", agent_won=True, player_won=False, foul=True)
            return

        env.board[y][x][1] = 1
        self.board.set_state(x, y, PlayerStatus.PLAYER)

        player_won = env.check_finish([x, y])
        if player_won:
            dispatcher.trigger("game_end", agent_won=False, player_won=True)
            return

        if not env.is_ok_to_continue():
            dispatcher.trigger("game_end", agent_won=False, player_won=False)
            return

        dispatcher.trigger("agents_turn", delay=500)

    def move_agent(self):
        env = self.env

        action_agent = self.agent.act(env.get_obs())
        x, y = env.action_to_xy(action_agent)
        if not env.is_valid_move([x, y]):
            print("*** Agent misplaced his stone to ({}, {})".format(x, y))
            dispatcher.trigger("game_end", agent_won=False, player_won=True, foul=True)
            return

        env.board[y][x][0] = 1
        self.board.set_state(x, y, PlayerStatus.AGENT)

        agent_won = env.check_finish([x, y])
        if agent_won:
            dispatcher.trigger("game_end", agent_won=True, player_won=False)
            return

        if not env.is_ok_to_continue():
            dispatcher.trigger("game_end", agent_won=False, player_won=False)
            return

        dispatcher.trigger("players_turn")

    def on_game_end(self, agent_won, player_won, foul=False):

        if agent_won:
            if foul:
                result = "あなたの反則負け"
            else:
                result = "エージェントの勝利"
        elif player_won:
            if foul:
                result = "エージェントの反則負け"
            else:
                result = "あなたの勝利"
        else:
            result = "引き分け"
        tkmsg.showinfo(message=result+"です。")
        self.game_over = True

    def game_restart(self):
        dispatcher.trigger("game_start")

    def start(self):
        root = tk.Tk()
        root.title("{ro}路盤{moku}目並べ エージェント動作確認".format(ro=self.board_size, moku=self.n_moku))
        dispatcher.set_root(root)

        board = Board(root, self.board_size, self.board_size)
        board.pack(padx=5, pady=(0,5))
        self.board = board

        restart_button = tk.Button(root, text="リセット", font=("", 30), width=10, height=2, command=self.game_restart)
        restart_button.pack(padx=5, pady=5)

        modeldir_label = tk.Label(root, text="使用モデル：{}".format(self.dirname))
        modeldir_label.pack(padx=5, pady=5, anchor=tk.W)

        dispatcher.trigger("game_start")

        root.mainloop()