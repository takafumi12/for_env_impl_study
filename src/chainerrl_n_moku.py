import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import logging
import sys
from env_n_moku import EnvNMoku
import argparse
from n_moku_gui import NMokuGui
import os

"""
以下のような2層の全結合ニューラルネットワークを実装
  入力：状態
  出力：最適な行動
"""
# エージェントの判断器となるニューラルネットワークを定義
class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False):
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))


# エージェントを定義
def get_agent(env, obs_size, n_actions):
    # 上記のニューラルネットワークを利用
    q_func = QFunction(obs_size, n_actions)
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)

    # DoubleDQNに必要なパラメータを定義
    gamma = 0.95
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.3, random_action_func=env.get_action_random)
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    phi = lambda x: x.astype(np.float32, copy=False)

    # ChainerRLに用意されたDoubleDQNアルゴリズムを利用
    agent = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=500, update_interval=1,
        target_update_interval=100, phi=phi)

    return agent


def train(env, agent, save_dir, steps):
    chainerrl.experiments.train_agent_with_evaluation(
        agent, env,
        steps=steps,
        eval_n_runs=10,  # 10 episodes are sampled for each evaluation
        max_episode_len=200,  # Maximum length of each episodes
        eval_interval=1000,  # Evaluate the agent after every 1000 steps
        outdir='result')  # Save everything to 'result' directory
    agent.save(save_dir)


def test_episode(env, agent):
    for i in range(10):
        obs = env.reset()
        done = False
        R = 0
        t = 0
        env.render()
        while not done and t < 200:
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            t += 1
            env.render()
        print('test episode:', i, 'R:', R)
        agent.stop_episode()


def test_interactive(env, agent, board_size, n_moku, dirname):
    gui = NMokuGui(board_size=board_size, n_moku=n_moku, dirname=dirname, agent=agent, env=env)
    gui.start()


def main():
    parser = argparse.ArgumentParser(description='Learner and tester for agent of tictactoe game.')
    parser.add_argument('-d', '--save_dir', type=str, default='nmokuagent', nargs='?', help='エージェントの保存／読み込み先ディレクトリ')
    parser.add_argument('--test', const=True, nargs='?', help='学習済みのエージェントをテストします')
    parser.add_argument('--test_interactive', const=True, nargs='?', help='学習済みのエージェントとインタラクティブに対戦します')
    parser.add_argument('-v', '--verbose', const=True, nargs='?', help='デバッグ用の詳細なログを出力します')
    parser.add_argument('--steps', default=50000, type=int, help='学習するステップ数を指定します')
    parser.add_argument('--board_size', default=3, type=int, help='盤のサイズ')
    parser.add_argument('--n_moku', default=3, type=int, help='何目並べたら勝ちとするか')
    args = parser.parse_args()

    assert args.n_moku <= args.board_size, 'n_mokuはboard_sizeより大きくは設定できません'

    env = EnvNMoku(board_size=args.board_size, n_moku=args.n_moku, verbose=args.verbose)
    obs = env.reset(opponent_first=False)
    print('initial observation:', obs)

    obs_size = obs.flatten().shape[0]
    n_actions = env.n_actions

    # Set up the logger to print info messages for understandability.
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

    agent = get_agent(env, obs_size, n_actions)
    if os.path.exists(os.path.join(args.save_dir, 'model.npz')):
        agent.load(args.save_dir)

    if args.test:
        test_episode(env, agent)
    elif args.test_interactive:
        test_interactive(env, agent, board_size=args.board_size, n_moku=args.n_moku, dirname=args.save_dir)
    else:
        # 自分自身を対戦相手に設定する
        env.set_opponent_agent(agent)

        train(env, agent, args.save_dir, args.steps)


if __name__ == '__main__':
    main()
