# -*- coding: utf-8 -*-
from random import randrange, choice
from copy import deepcopy
from collections import defaultdict
import sys
import os
import time

"""
ref: https://www.geeksforgeeks.org/2048-game-in-python/

訓練AI的部分可參考將C code改寫成python:
1. (觀念)https://junmo1215.github.io/machine-learning/2017/11/27/practice-TDLearning-in-2584-fibonacci-2nd.html
2. (code)https://github.com/junmo1215/rl_games/blob/8809ae2a9eefb8e492e24658d28268c67f00281e/2584_C%2B%2B/agent.h
"""

class HiddenPrints():
    """ ref: https://cloud.tencent.com/developer/ask/188486 """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
class Episode():
    def __init__(self, before_state, after_state, reward):
        self.before = before_state
        self.after = after_state
        self.reward = reward
        
class Game_2048():
    """
    盤面表示: 計算上，用i表示數值pow(2,i)
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.mat =[[0] * self.width for i in range(self.height)]
        self.episodes = [] #儲存遊戲記錄
    
    def start_game(self): 
        self.mat =[[0] * self.width for i in range(self.height)] 
        self.episodes = [] #儲存遊戲記錄
        # printing controls for user 
        print("Commands are as follows : ") 
        print("'W' or 'w' : Move Up") 
        print("'S' or 's' : Move Down") 
        print("'A' or 'a' : Move Left") 
        print("'D' or 'd' : Move Right") 
        # calling the function to add a new 2 in grid after every step 
        self.add_new_2()
    
    def add_new_2(self): 
        # function to add a new 2 or 4 in grid at any random empty cell 
        new_element = 2 if randrange(100) > 89 else 1
        i,j = choice([(i,j) for i in range(self.height) for j in range(self.width) if self.mat[i][j] == 0])
        self.mat[i][j] = new_element
    
    def can_move_left(self, mat):
        """ 判斷mat盤面向左滑是否是合法棋步 """
        def row_is_left_move(row):
            for i in range(len(row) - 1):
                if (row[i] == 0 and row[i + 1] != 0) or (row[i] != 0 and row[i + 1] == row[i]):
                    return True
            return False
        return any(row_is_left_move(row) for row in mat)
    
    def get_valid_move(self, mat):
        valid_moves = []
        for move in ['s','d','w','a']:
            mat = self.turn_right(mat)
            if self.can_move_left(mat):
                valid_moves.append(move)
        return valid_moves
        
    def compress(self, mat): 
        """ 只定義往左滑的功能，上、下、右滑用鏡射、旋轉解 """
        return [sorted(row, key = lambda x: bool(x == 0)) for row in mat]
    
    def merge(self, mat): 
        # function to merge the cells in matrix after compressing 
        for i in range(len(mat)): 
            for j in range(len(mat[0])-1): 
                # 合併兩個相同方塊
                if(mat[i][j] == mat[i][j + 1] and mat[i][j] != 0): 
                    mat[i][j] += 1
                    mat[i][j + 1] = 0
        return mat
      
    def reverse(self, mat):
        return [row[::-1] for row in mat]
    
    def transpose(self, mat): 
        return [list(row) for row in zip(*mat)]
    
    def turn_right(self, mat):
        """ 矩陣右旋 """
        return self.reverse(self.transpose(mat))
    
    def move_left(self, grid): 
        new_grid = self.compress(grid) 
        new_grid = self.merge(new_grid) 
        new_grid = self.compress(new_grid)  # again compress after merging. 
        return new_grid
    
    def move_right(self, grid): 
        new_grid = self.reverse(grid) 
        new_grid = self.move_left(new_grid) 
        new_grid = self.reverse(new_grid) 
        return new_grid 
    
    def move_up(self, grid): 
        new_grid = self.transpose(grid) 
        new_grid = self.move_left(new_grid) 
        new_grid = self.transpose(new_grid) 
        return new_grid
    
    def move_down(self, grid): 
        new_grid = self.transpose(grid) 
        new_grid = self.move_right(new_grid) 
        new_grid = self.transpose(new_grid) 
        return new_grid
    
    def show_board(self):
        for m in self.mat:
            print(m)
        print('-'*10)
        
    def __gameloop(self, ai_agent=None):
        """
        ai_agent(mat): ai做選擇的函數，input盤面mat，輸出要往上、下、左、右哪邊滑，輸入None時由人類玩家玩。
        """
        self.start_game()
        self.show_board()
        while True:
            before_state = deepcopy(self.mat)
            valid_move = self.get_valid_move(self.mat)
            print('目前合法棋步:', valid_move)
            if not valid_move:
                print('GAME_OVER')
                break
            if any(e == 2048 for row in self.mat for e in row):
                print('You won.')
                break
            
            if callable(ai_agent):
                act = ai_agent(self.mat)
            else:
                act = input("Press the command : ").strip().lower()
            print(f"Take action '{act}'")
            move_func = {'w': self.move_up,
                         's': self.move_down,
                         'a': self.move_left,
                         'd': self.move_right}
    
            if act in valid_move:    
                self.mat = move_func[act](self.mat)
                after_state = deepcopy(self.mat)
                self.episodes.append(Episode(before_state, after_state,1))
                self.add_new_2()
            elif act=='q':
                print('Bye~')
                break
            else: 
                print("Invalid Key Pressed")
            self.show_board()
        return max(max(row) for row in self.mat)
    
    def gameloop(self, ai_agent=None, hidden_print=False):
        if hidden_print:
            with HiddenPrints():
                return self.__gameloop(ai_agent)
        return self.__gameloop(ai_agent)
    
    def play_many_game(self, game_num, ai_agent=None, hidden_print=True):
        score_dict = defaultdict(int)
        if hidden_print:
            with HiddenPrints():
                for i in range(game_num):
                    max_tile = self.gameloop(ai_agent)
                    score_dict[max_tile] += 1
        else:
            for i in range(game_num):
                max_tile = self.gameloop(ai_agent)
                score_dict[max_tile] += 1
        statistic(score_dict)
        

def random_ai(mat):
    game = Game_2048(len(mat[0]),len(mat))
    valid_move = game.get_valid_move(mat)
    return choice(valid_move)  

def statistic(score_dict):
    """
    對遊戲結束時達到的最高分的tile進行統計
    """
    total_game = sum(score_dict.values())
    for key in sorted(score_dict):
        print(2**key, f"{score_dict[key]/total_game*100:.2f}%")
    print('-'*10)
            
if __name__ == '__main__': 
    Game = Game_2048(4,4)
    s = time.time()
    Game.play_many_game(1000, random_ai)
    print('執行時間', time.time()-s)

    

