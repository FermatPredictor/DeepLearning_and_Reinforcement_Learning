# -*- coding: utf-8 -*-
from game_2048 import Game_2048
import time
import pickle
from collections import defaultdict
from datetime import datetime

class TD_weight_table():
    def __init__(self, lr, index_tuples, MAX_INDEX):
        """
        index_tuples: n-tuples 的index list
        MAX_INDEX: 方塊編號為 0,1,...,MAX_INDEX
        """
        self.weights = [[0]*pow(MAX_INDEX,len(index_tuples[0])) for _ in range(len(index_tuples))]
        self.index_tuples = index_tuples[:]
        self.MAX_INDEX = MAX_INDEX
        self.lr = lr
    
    def close_episode(self, episode):
		# train the n-tuple network by TD(0)
        self.train_end_board_weights(episode[-1].after)
        for i in range(len(episode)-2,-1,-1):
            step_next = episode[i + 1]
            self.train_weights(episode[i].after, step_next.after, step_next.reward)
  
    def get_feature(self, board, index_tuple):
        result = 0
        height, width = len(board), len(board[0])
        MAX_INDEX = self.MAX_INDEX-1
        for i in index_tuple:
            result *= MAX_INDEX
            tile = min(board[i // height][i % width], MAX_INDEX)
            result += tile
        return result
            
    def board_value(self, board):
        return sum(self.weights[i][self.get_feature(board, self.index_tuples[i])] for i in range(len(self.weights)))
    
    def train_weights(self, board, next_board, reward:int):
        delta = self.lr * (reward + self.board_value(next_board) - self.board_value(board))
        for i in range(len(self.weights)):
            self.weights[i][self.get_feature(board, self.index_tuples[i])] += delta

    def train_end_board_weights(self, board):
        delta = - self.lr * self.board_value(board)
        for i in range(len(self.weights)):
            self.weights[i][self.get_feature(board, self.index_tuples[i])] += delta
            
    def save(self, path='./weights.pickle'):
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)
            
    def load(self, path='./weights.pickle'):
        with open(path, 'rb') as f:
            self.weights = pickle.load(f)


def td_ai(mat):
    game = Game_2048(len(mat[0]),len(mat))
    valid_move = game.get_valid_move(mat)
    move_func = {'w': game.move_up,
                 's': game.move_down,
                 'a': game.move_left,
                 'd': game.move_right}
    best_score = -99999
    best_move = valid_move[0]
    for move in valid_move:
        new_mat = move_func[move](mat)
        s = td.board_value(new_mat)
        if s>best_score:
            best_score = s
            best_move = move  
    return best_move  

def statistic(score_dict):
    """
    對遊戲結束時達到的最高分的tile進行統計
    """
    total_game = sum(score_dict.values())
    for key in sorted(score_dict):
        print(2**key, f"{score_dict[key]/total_game*100:.2f}%")
    print('-'*10)
            
if __name__ == '__main__':
    """
    The best record in 1000 games:
    128 0.20%
    256 0.80%
    512 4.10%
    1024 28.10%
    2048 66.20%
    4096 0.60%
    
    lr = 0.01
    index_tuples = [(0,1,2,3),(4,5,6,7),(8,9,10,11),(12,13,14,15),
                    (0,4,8,12),(1,5,9,13),(2,6,10,14),(3,7,11,15)]
    MAX_INDEX = 16
    td = TD_weight_table(lr, index_tuples,MAX_INDEX)
    """
    
    lr = 0.01
    index_tuples = [(0,1,2,3),(4,5,6,7),(8,9,10,11),(12,13,14,15),
                    (0,4,8,12),(1,5,9,13),(2,6,10,14),(3,7,11,15)]
    MAX_INDEX = 16
    td = TD_weight_table(lr, index_tuples,MAX_INDEX)
    td.load('./4x4_weights_2021-01-31_08_54_45.pickle')
    
    Game = Game_2048(4,4)
    train_num = 10
    game_num = 1000
    
    for idx in range(train_num):
        score_dict = defaultdict(int)
        s = time.time()
        path = f'./4x4_weights_{datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}.pickle'
        for i in range(game_num):
            max_tile = Game.gameloop(td_ai, hidden_print=True)
            score_dict[max_tile] += 1
            episode = Game.episodes
            td.close_episode(episode)
        td.save(path)
        statistic(score_dict)
        print(f'第{idx+1}次training, 執行時間', time.time()-s)
