# -*- coding: utf-8 -*-
from random import randrange, choice

""" ref: https://www.geeksforgeeks.org/2048-game-in-python/"""
# function to initialize game / grid at the start 
def start_game(): 
    mat =[[0] * 4 for i in range(4)] 
    # printing controls for user 
    print("Commands are as follows : ") 
    print("'W' or 'w' : Move Up") 
    print("'S' or 's' : Move Down") 
    print("'A' or 'a' : Move Left") 
    print("'D' or 'd' : Move Right") 
    # calling the function to add a new 2 in grid after every step 
    add_new_2(mat) 
    return mat

def add_new_2(mat): 
    # function to add a new 2 or 4 in grid at any random empty cell 
    new_element = 4 if randrange(100) > 89 else 2
    (i,j) = choice([(i,j) for i in range(4) for j in range(4) if mat[i][j] == 0])
    mat[i][j] = new_element
  
# function to get the current state of game 
def get_current_state(mat): 
    # if any cell contains 2048 we have won 
    for i in range(4): 
        for j in range(4): 
            if(mat[i][j]== 2048): 
                return 'WON'
    if get_valid_move(mat):
        return 'GAME NOT OVER'
    return 'LOST'

def can_move_left(mat):
    """ 判斷mat盤面向左滑是否是合法棋步 """
    def row_is_left_move(row):
        for i in range(len(row) - 1):
            if (row[i] == 0 and row[i + 1] != 0) or (row[i] != 0 and row[i + 1] == row[i]):
                return True
        return False
    return any(row_is_left_move(row) for row in mat)

def get_valid_move(mat):
    move_list = ['s','d','w','a']
    valid_moves = []
    for i in range(4):
        mat = turn_right(mat)
        if can_move_left(mat):
            valid_moves.append(move_list[i])
    return valid_moves
    
def compress(mat): 
    """ 只定義往左滑的功能，上、下、右滑用鏡射、旋轉解 """
    new_mat = [sorted(mat[i], key = lambda x: bool(x == 0)) for i in range(4)]
    return new_mat 

def merge(mat): 
    # function to merge the cells in matrix after compressing 
    for i in range(4): 
        for j in range(3): 
            # 合併兩個相同方塊
            if(mat[i][j] == mat[i][j + 1] and mat[i][j] != 0): 
                # double current cell value and empty the next cell 
                mat[i][j] *= 2
                mat[i][j + 1] = 0
    return mat
  
def reverse(mat):
    return [row[::-1] for row in mat]

def transpose(mat): 
    return [list(row) for row in zip(*mat)]

def turn_right(mat):
    """ 矩陣右旋 """
    return reverse(transpose(mat))

def move_left(grid): 
    new_grid = compress(grid) 
    new_grid = merge(new_grid) 
    new_grid = compress(new_grid)  # again compress after merging. 
    return new_grid

def move_right(grid): 
    new_grid = reverse(grid) 
    new_grid = move_left(new_grid) 
    new_grid = reverse(new_grid) 
    return new_grid 

def move_up(grid): 
    new_grid = transpose(grid) 
    new_grid = move_left(new_grid) 
    new_grid = transpose(new_grid) 
    return new_grid

def move_down(grid): 
    new_grid = transpose(grid) 
    new_grid = move_right(new_grid) 
    new_grid = transpose(new_grid) 
    return new_grid
  
if __name__ == '__main__': 
      
    # calling start_game function to initialze the matrix 
    mat = start_game()
    print('Game Start')
    for m in mat:
        print(m)
    print('-'*10)
  
    while True: 
      
        # taking the user input  for next step 
        valid_move = get_valid_move(mat)
        print('目前合法棋步:', valid_move)
        if not valid_move:
            print('GAME_OVER')
            break
        
        act = input("Press the command : ").strip().lower()
        move_func = {'w': move_up,
                     's': move_down,
                     'a': move_left,
                     'd': move_right}

        if act in valid_move:    
            mat = move_func[act](mat)
            add_new_2(mat)
        elif act=='q':
            break
        else: 
            print("Invalid Key Pressed") 
      
        # print the matrix after each move. 
        for m in mat:
            print(m)
        print('-'*10)
