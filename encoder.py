import numpy as np
import importlib
from gorule import Move, Player, Point

class Encoder():
    def __init__(self, board_size=9):
        self.board_size = board_size
        self.board_width,self.board_height = board_size, board_size
        self.num_planes = 11
    
    def encode(self, game_state):
        '''
        0-3: player1 with 1,2,3,4+ liberties
        4-7: player2 with 1,2,3,4+ liberties
        8: 1 means player1 get komi 6.5 (white player)
        9: 1 means player2 get komi 6.5 (black player)
        10: illegal move due to ko
        '''
        board_tensor = np.zeros(self.shape()) # the encoded plane
        next_player = game_state.next_player
        
        # 8 and 9 plane, find which player plays white stone
        if game_state.next_player == Player.white:
            board_tensor[8] = 1
        else:
            board_tensor[9] = 1
        
        '''
        1. get the string of the point
        2. check legality
        3. find the liberties of the point
        '''
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = Point(row=r+1, col=c+1)
                go_string = game_state.board.get_go_string(p)
                
                if go_string is None: # check legality
                    if game_state.does_move_violate_ko(next_player, Move.play(p)):
                        board_tensor[10][r][c] = 1
                else: # count the number of liberities
                    liberty_plane = min(4, go_string.num_liberties) -1
                    if go_string.colour != next_player:
                        liberty_plane += 4 # player 2
                    board_tensor[liberty_plane][r][c] = 1
        return board_tensor
            
    
    def name(self):
        return 'InfinityGo'
    
    def encode_move(self, move):
        '''
        Represented the move as a number
           a  b  c  d  e  f  g  h  i
        9| 72 73 74 75 76 77 78 79 80
                .........
        1|  0  1  2  3  4  5  6  7  8
        '''
        if move.is_play:
            return (self.board_size*(move.point.row-1)+(move.point.col-1))
        elif move.is_pass:
            return self.board_size * self.board_size
        return ValueError('Cannot encode resign move')
        
    def decode_move_index(self, index):
        '''
        decode the move to the Point(row,col) format
        '''
        if index == self.board_size * self.board_size:
            return Move.pass_turn()
        row = index // self.board_size
        col = index % self.board_size
        return Move.play(Point(row+1, col+1))
    
    def num_moves(self):
        '''
        return the total number of the moves (include pass move)
        '''
        return self.board_size * self.board_size + 1
    
    def shape(self):
        '''
        return the shape of the encoder: 
        - number_planes * board_size * board_size
        '''
        return self.num_planes, self.board_size, self.board_size

def get_encoder_by_name(name, board_size):
    '''Dynamically load an return the encoder'''
    if isinstance(board_size,int):
        board_size = (board_size,board_size)
    module = importlib.import_module('infinitygo.encoders.'+name)
    constructor = getattr(module, 'create')
    return constructor(board_size)