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
    128 0.10%
    256 0.50%
    512 3.50%
    1024 25.70%
    2048 68.10%
    4096 2.10%
    
    lr = 0.01
    index_tuples = [(0,1,2,3),(4,5,6,7),(8,9,10,11),(12,13,14,15),
                    (0,4,8,12),(1,5,9,13),(2,6,10,14),(3,7,11,15)]
    MAX_INDEX = 16
    td = TD_weight_table(lr, index_tuples,MAX_INDEX)
    """
    
    """
    6-tuple solution:
    index_tuples = [ (0, 4, 8, 9, 12, 13), 
                    (1, 5, 9, 10, 13, 14),
                	(1, 2, 5, 6, 9, 10),
                	(2, 3, 6, 7, 10, 11),
                	
                	(3, 2, 1, 5, 0, 4),
                	(7, 6, 5, 9, 4, 8),
                	(7, 11, 6, 10, 5, 9),
                	(11, 15, 10, 14, 9, 13),
                
                	(15, 11, 7, 6, 3, 2),
                	(14, 10, 6, 5, 2, 1),
                	(14, 13, 10, 9, 6, 5),
                	(13, 12, 9, 8, 5, 4),
                
                	(12, 13, 14, 10, 15, 11),
                	(8, 9, 10, 6, 11, 7),
                	(8, 4, 9, 5, 10, 6),
                	(4, 0, 5, 1, 6, 2),
                
                
                	(3, 7, 11, 10, 15, 14),
                	(2, 6, 10, 9, 14, 13),
                	(2, 1, 6, 5, 10, 9),
                	(1, 0, 5, 4, 9, 8),
                
                	(0, 1, 2, 6, 3, 7),
                	(4, 5, 6, 10, 7, 11),
                	(4, 8, 5, 9, 6, 10),
                	(8, 12, 9, 13, 10, 14),
                
                	(12, 8, 4, 5, 0, 1),
                	(13, 9, 5, 6, 1, 2),
                	(13, 14, 9, 10, 5, 6),
                	(14, 15, 10, 11, 6, 7),
                
                	(15, 14, 13, 9, 12, 8),
                	(11, 10, 9, 5, 8, 4),
                	(11, 7, 10, 6, 9, 5),
                	(7, 3, 6, 2, 5, 1)]
    """
    
    lr = 0.01
    index_tuples = [(0,1,2,3),(4,5,6,7),(8,9,10,11),(12,13,14,15),
                    (0,4,8,12),(1,5,9,13),(2,6,10,14),(3,7,11,15)]
    MAX_INDEX = 16
    td = TD_weight_table(lr, index_tuples,MAX_INDEX)
    td.load('./4x4_weights_2021-01-31_15_39_41.pickle')
    
    Game = Game_2048(4,4)
    train_num = 1
    game_num = 100
    
    for idx in range(train_num):
        score_dict = defaultdict(int)
        s = time.time()
        path = f'./4x4_weights_{datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}.pickle'
        for i in range(game_num):
            max_tile = Game.gameloop(td_ai, hidden_print=True)
            score_dict[max_tile] += 1
            episode = Game.episodes
            td.close_episode(episode)
        statistic(score_dict)
        td.save(path)
        print(f'第{idx+1}次training, 執行時間', time.time()-s)
