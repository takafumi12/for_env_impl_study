import random

class PlayerRandom:
    def __init__(self, turn:int, name:str="Random"):
        """
        ランダムに行動するエージェント

        Args:
            turn (int): PLAYER_AGENT or PLAYER_OPPONENTを表す
            name (str, optional): エージェントの名称. Defaults to "Random".
        """
        self.myturn = turn
        self.name = name
        
    def act(self, board_env:object)->int:
        """
        利用可能な行動からランダムに1つ選ぶ

        Args:
            board_env (object): TTTBoardのインスタンス

        Returns:
            int: 石を置くidx
        """

        # 配置可能な盤面のidxlistを受け取る
        acts = board_env.get_possible_pos()

        # 配置可能な盤面のidxのうちランダムで1つ選ぶ
        i = random.randrange(len(acts))

        return acts[i]