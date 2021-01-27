# -*- coding: utf-8 -*-
from game_2048 import Game_2048
import time

class TD_weight_table():
    def __init__(self, lr, index_tuples, table_size):
        self.weights = [[0]*table_size for _ in range(len(index_tuples))]
        self.index_tuples = index_tuples[:]
        self.lr = lr
    
    def close_episode(self, episode):
		# train the n-tuple network by TD(0)
        self.train_end_board_weights(episode[-1].after)
        for i in range(len(episode)-2,-1,-1):
            step_next = episode[i + 1]
            self.train_weights(episode[i].after, step_next.after, step_next.reward)
  
    def get_feature(self, board, index_tuple):
        result = 0
        MAX_INDEX = 4
        for i in index_tuple:
            result *= MAX_INDEX
            tile = min(board[i // 3][i % 3], MAX_INDEX)
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

td = TD_weight_table(0.01, [(0,1,2,3)],5**4)
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
        print(key, f"{score_dict[key]/total_game*100:.2f}%")
    print('-'*10)
            
if __name__ == '__main__': 
    Game = Game_2048(3,3)
    s = time.time()
    game_num = 5
    for i in range(game_num):
        Game.play_many_game(10, td_ai)
    episode = Game.episodes
    td.close_episode(episode)
    print(td.weights)
    print('執行時間', time.time()-s)
