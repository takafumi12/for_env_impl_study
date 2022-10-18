
import os
import yaml
import numpy as np
import random
from typing import Tuple

# 設定ファイルの読み込み
config_dir = '../config/'
with open(os.path.join(config_dir,'config.yaml'), 'r') as yml:
    config = yaml.safe_load(yml)

EMPTY=config['EMPTY'] # 盤面が空いていることを示す値
PLAYER_AGENT=config['PLAYER_AGENT'] # PLAYER_AGENTを示す値
PLAYER_OPPONENT=config['PLAYER_OPPONENT'] # PLAYER_OPPONENTを示す値
DRAW=config['DRAW'] # 引き分けを示す値
SIZE=config['SIZE'] # 盤面の縦・横のマス目
REWARD_DICT = config['REWARD_DICT'] # WIN、LOSE、DRAWの報酬の辞書

class TTTBoard:
    
    def __init__(self, board:list=None):
        """
        TTTの環境を定義するクラス

        Args:
            board (list, optional): ボードの盤面のlist. Defaults to None.
        """
        if board == None:
            self.board = []
            for i in range(SIZE*SIZE):
                self.board.append(EMPTY)
        else:
            self.board=board
        self.winner=None # 勝敗が決まったときに PLAYER_AGENT or PLAYER_OPPONENT or DRAWに更新される

    def get_possible_pos(self)->list[int]:
        """
        配置できるポジションを取得する

        Returns:
            list[int]: board上でEMPTYとなっているidxのlist
        """
        pos=[]
        for i in range(SIZE*SIZE):
            if self.board[i] == EMPTY:
                pos.append(i)
        return pos

    def mk_win_cond_list(self)->list[int]:
        """
        win状態(縦、横、斜めのidx)を判定する用のlistを作成する
        ex) 3目並べの場合は[[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

        Returns:
            list[int]: win状態(縦、横、斜めのidx)を判定する用のlist
        """
        borad_idx = np.arange(0, SIZE*SIZE).reshape(SIZE, SIZE)
        win_cond_list = [list(borad_idx[i]) for i in range(SIZE)]
        win_cond_list += [list(borad_idx.T[i]) for i in range(SIZE)]
        win_cond_list += [list(np.diag(borad_idx)), list(np.diag(np.flipud(borad_idx)))]

        return win_cond_list
                       
    def check_winner(self)->int:
        """
        勝敗が決しているかを判定し、決している場合は勝者を返す
        判定方法：縦、横、斜めが全て同じ値かチェックかつEMPTY以外の値が入っている場合は勝者

        Returns:
            int: 勝者を表すint (PLAYER_AGENT or PLAYER_OPPONENT)
        """
        win_cond_list = self.mk_win_cond_list()
        for win_cond in win_cond_list:
            current_cond = [self.board[idx] for idx in win_cond] 
            if all(val == current_cond[0] for val in current_cond): # 縦、横、斜めが全て同じ値かチェック
                if current_cond[0] != EMPTY: # ↑の内EMPTY以外の値が入っているかチェック
                    self.winner = current_cond[0]
                    return self.winner
        return None
    
    def check_draw(self)->int:
        """
        引き分け状態かチェック
        判定方法：置ける場所が0かつ勝者が決まっていない場合は引き分け

        Returns:
            int: 引き分けを表すint（DRAW）
        """
        if len(self.get_possible_pos())==0 and self.winner is None:
            self.winner=DRAW
            return DRAW
        return None
    
    def move(self, pos:int, player:int):
        """
        石を置いたあとに勝敗＆引き分け判定を行う

        Args:
            pos (int): 石を置く場所を表すidx
            player (int): プレイヤーを表すint（PLAYER_AGENT or PLAYER_OPPONENT）
        """
        if self.board[pos] == EMPTY:
            self.board[pos] = player
        else:
            self.winner = -1*player # 置けない場所に置いたら相手の勝ち
        self.check_winner()
        self.check_draw()
    
    def clone(self)->object:
        """
        盤面の状態をコピーしたTTTBoardインスタンスを生成する

        Returns:
            object: 盤面の状態をコピーしたTTTBoardインスタンス
        """
        return TTTBoard(self.board.copy())


class TTTEnv:

    def __init__(self, player_agent:object, player_opponent:object):
        """
        Gymのインターフェースで行動を受け取り環境を変化させるクラス

        Args:
            player_agent (object): playerインスタンス
            player_opponent (object): playerインスタンス
        """
        self.player_agent = player_agent # agentのplayerインスタンス
        self.player_opponent = player_opponent # opponentのplayerインスタンス
        self.nwon = {player_agent.myturn:0, player_opponent.myturn:0, DRAW:0} # 勝敗、引き分け数のLog
        self.players = (self.player_agent, self.player_opponent) # agentとopponentのインスタンスのタプル
        self.board_env = None # TTTBoardの新スタンス
        self.player_turn = self.players[random.randrange(2)] #行動するプレイヤーのインスタンス

    def reset(self)->object:
        """
        状態（盤面）のリセット

        Returns:
            object: リセット後の盤面のリスト
        """

        self.board_env = TTTBoard()
        
        return self.get_obs()

    def step(self, action:int)->Tuple[list[int], int, bool, dict]:
        """
        マイターンのPlayerが行動して結果を反映させる

        Args:
            action (int): 石を置くidx

        Returns:
            Tuple[list[int], int, bool, dict]: [盤面のリスト、報酬、終了判定、勝敗のLog]
        """

        # マイターンプレイヤーが石を置き、勝敗・引き分け判定を行う
        self.board_env.move(action, self.player_turn.myturn)

        # 勝敗・引き分けが確定している
        if self.board_env.winner != None:

            self.nwon[self.board_env.winner]+=1
            done = True

            # 勝敗パターンによって報酬を取得
            if self.board_env.winner == DRAW:
                reward = REWARD_DICT['DRAW']
            elif self.board_env.winner == PLAYER_AGENT:
                reward = REWARD_DICT['WIN']
            else:
                reward = REWARD_DICT['LOSE']

        # 勝敗・引き分けが確定していない
        else:
            reward = 0
            done = False
            self.switch_player() # プレイヤーを交代する

        return self.get_obs(), reward, done, self.nwon

    def get_obs(self)->list[int]:
        """
        状態（盤面）の取得

        Returns:
            list[int]: ボードの盤面のlist
        """
        return self.board_env.board

    def switch_player(self):
        """
        プレイするプレイヤーを交代する
        """
        if self.player_turn == self.player_agent:
            self.player_turn=self.player_opponent
        else:
            self.player_turn=self.player_agent