# -*- coding: utf-8 -*-
class TD_weight_table():
    def __init__(self, index_tuples, table_size):
        self.weights = [[0]*table_size for _ in range(len(index_tuples))]
        self.index_tuples = index_tuples[:]
    
    def close_episode(self, episode):
		# train the n-tuple network by TD(0)
        self.train_end_board_weights(episode[-1].after)
        for i in range(len(episode)-2,-1,-1):
            step_next = episode[i + 1]
            self.train_weights(episode[i].after, step_next.after, step_next.reward)
            
  
    def get_feature(self, board, index_tuple):
        result = 0
        MAX_INDEX = 24
        for i in index_tuple:
            result *= MAX_INDEX
            tile = min(board[i // 4][i % 4], MAX_INDEX)
            result += tile
        return result
            
    def board_value(self, board):
        return sum(self.weights[i][self.get_feature(board, self.index_tuples[i])] for i in range(len(self.weights)))
    
    def train_weights(self, alpha, board, next_board, reward:int):
        delta = alpha * (reward + self.board_value(next_board) - self.board_value(board))
        for i in range(len(self.weights)):
            self.weights[i][self.get_feature(board, self.index_tuples[i])] += delta;
		

    def train_end_board_weights(self, alpha, board):
        delta = - alpha * self.board_value(board)
        for i in range(len(self.weights)):
            self.weights[i][self.get_feature(board, self.index_tuples[i])] += delta;
