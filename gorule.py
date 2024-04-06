import random
import enum
import copy

from collections import namedtuple

import numpy as np
from goscore import GoScore

neighbour_tables = {}
corner_tables = {}

def init_neighbour_table(dim):
    rows, cols = dim
    new_table = {}
    for r in range(1, rows+1):
        for c in range(1, cols+1):
            p =  Point(row=r, col=c)
            full_neighbours = p.neighbours()
            true_neighbors = [
                n for n in full_neighbours
                if 1<=n.row<=rows and 1<=n.col<=cols
            ]
            new_table[p] = true_neighbors
    neighbour_tables[dim] = new_table
    
def init_corner_table(dim):
    rows, cols = dim
    new_table = {}
    for r in range(1,rows+1):
        for c in range(1,cols+1):
            p = Point(row=r,col=c)
            full_corners = [
                Point(row=p.row-1, col=p.col-1),
                Point(row=p.row-1, col=p.col+1),
                Point(row=p.row+1, col=p.col-1),
                Point(row=p.row+1, col=p.col+1)
            ]
            true_corners = [
                n for n in full_corners
                if 1<=n.row<=rows and 1<=n.col<=cols
            ]
            new_table[p] = true_corners
    corner_tables[dim] = new_table

class Player(enum.Enum):
    black = 1
    white = 2
    @property
    def other(self):
        '''
        Return the colour of the opponent
        '''
        if self == Player.black:
            return Player.white
        else:
            return Player.black

class Point(namedtuple('Point','row col')):
    def neighbours(self):
        '''
        return the adjacent points
        '''
        return [
            Point(self.row-1,self.col),
            Point(self.row+1,self.col),
            Point(self.row, self.col-1),
            Point(self.row, self.col+1)
        ]

class Move():
    '''
    Three move conditions:
        - play
        - pass
        - resign
    '''
    def __init__(self, point=None, is_pass=False, is_resign=False):
        # only one condition is true
        assert(point is not None) ^ is_pass ^ is_resign
        self.point = point
        self.is_pass = is_pass
        self.is_resign = is_resign
        self.is_play = (self.point is not None)
    
    @classmethod
    def play(cls,point):
        '''
        play: place the stone
        '''
        return Move(point=point)
    
    @classmethod
    def pass_turn(cls):
        '''
        One player pass
        '''
        return Move(is_pass=True)
    
    @classmethod
    def resign(cls):
        '''
        One player resign
        '''
        return Move(is_resign=True)
    
    def __str__(self):
        if self.is_pass:
            return 'pass'
        if self.is_resign:
            return 'resign'
        return '(r %d, c %d)' % (self.point.row, self.point.col)
    
    def __hash__(self):
        return hash((
            self.is_play,
            self.is_pass,
            self.is_resign,
            self.point
        ))
        
    def __eq__(self,other):
        return (
            self.is_play,
            self.is_pass,
            self.is_resign,
            self.point
        )   ==  (
            other.is_play,
            other.is_pass,
            other.is_resign,
            other.point
        )

class GoString():
    def __init__(self,colour, stones, liberties):
        self.colour = colour
        self.stones = frozenset(stones)
        self.liberties = frozenset(liberties)
        
    def without_liberty(self, point):
        '''
        remove the liberty
        '''
        new_liberties = self.liberties - set([point])
        return GoString(self.colour, self.stones, new_liberties)
    
    def with_liberty(self, point):
        '''
        add the liberty
        '''
        new_liberties = self.liberties | set([point])
        return GoString(self.colour, self.stones, new_liberties)
    
    def merged_with(self, go_string):
        '''
        after placing the stone, two group of adjacent pices may merge into one
        '''
        assert go_string.colour == self.colour
        combined_stones = self.stones | go_string.stones
        return GoString(self.colour, combined_stones, 
                        (self.liberties|go_string.liberties)-combined_stones)
        
    @property
    def num_liberties(self):
        '''
        the number of liberties
        '''
        return len(self.liberties)
        
    def __eq__(self, other):
        '''
        check two strings' attributs equal or not
        '''
        return isinstance(other, GoString) and \
            self.colour == other.colour and \
            self.stones == other.stones and \
            self.liberties == other.liberties
            
class MoveAge():
    def __init__(self,board):
        self.move_ages = - np.ones((board.num_rows, board.num_cols))
    def get(self,row,col):
        return self.move_ages[row,col]
    def reset_age(self,point):
        self.move_ages[point.row-1,point.col-1] = -1
    def add(self,point):
        self.move_ages[point.row-1,point.col-1] = 0
    def increment_all(self):
        self.move_ages[self.move_ages > -1] += 1

class Board():
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {}
        self._hash = zobrist_EMPTY_BOARD
        
        global neighbour_tables
        dim = (num_rows,num_cols)
        if dim not in neighbour_tables:
            init_neighbour_table(dim)
        if dim not in corner_tables:
            init_corner_table(dim)
        self.neighbour_table = neighbour_tables[dim]
        self.corner_table = corner_tables[dim]
        self.move_ages = MoveAge(self)
        
    def zobrist_hash(self):
        return self._hash
    
    def zobrist_grid(self):
        return self._grid
    
    def is_in_zobrist(self, point):
        return self._grid.get(point)    
    
    def place_stone(self,player,point):
        '''
        the player place the stone
        '''
        assert self.is_on_grid(point)
        assert self._grid.get(point) is None
        
        adjacent_same_colour = []
        adjacent_opposite_colour = []
        liberties = []
        self.move_ages.increment_all()
        self.move_ages.add(point)
        
        for neighbour in self.neighbour_table[point]:
            neighbour_string = self._grid.get(neighbour)
            if neighbour_string is None:
                liberties.append(neighbour)
            elif neighbour_string.colour == player:
                if neighbour_string not in adjacent_same_colour:
                    adjacent_same_colour.append(neighbour_string)
            else:
                if neighbour_string not in adjacent_opposite_colour:
                    adjacent_opposite_colour.append(neighbour_string)
        
        # for neighbour in point.neighbours():
        #     # Point not exist in the board
        #     if not self.is_on_grid(neighbour):
        #         continue
            
            # neighbour_string = self._grid.get(neighbour)
            # if neighbour_string is None: # the neighbour is empty and is the liberty of this stone
            #     liberties.append(neighbour)
            # elif neighbour_string.colour == player: # adjecent the same colour stone
            #     if neighbour_string not in adjacent_same_colour:
            #         adjacent_same_colour.append(neighbour_string)
            # else: # adjecent the different colour stone
            #     if neighbour_string not in adjacent_opposite_colour:
            #         adjacent_opposite_colour.append(neighbour_string)
                    
        # Combine the current stone with adjacent stones on the chessboard into one
        new_string = GoString(player, [point], liberties)
        for same_colour_string in adjacent_same_colour:
            new_string = new_string.merged_with(same_colour_string)
        
        for new_string_point in new_string.stones:
            # When accessing a stone on the board, return the set of adjacent stones
            self._grid[new_string_point] = new_string
            
        self._hash ^= zobrist_HASH_CODE[point, None]
        self._hash ^= zobrist_HASH_CODE[point, player]
            
        for other_colour_string in adjacent_opposite_colour:
            # remove the liberty where the stone placed
            replacement = other_colour_string.without_liberty(point)
            if replacement.num_liberties:
                self._replace_string(other_colour_string.without_liberty(point))
            else:
                self._remove_string(other_colour_string)
                    
    
    def _replace_string(self, new_string):
        '''
        replace the point's string with the new string
        '''
        for point in new_string.stones:
            self._grid[point] = new_string
      
    def is_on_grid(self,point):
        '''
        determine if the position is on the board
        '''
        return 1<=point.row<=self.num_cols and 1<=point.col<=self.num_cols
    
    def get(self,point):
        '''
        get the colour of the stone
        '''
        string=self._grid.get(point)
        if string is None:
            return None
        return string.colour

    def get_go_string(self,point):
        '''
        get the string this point belongs to
        '''
        string = self._grid.get(point)
        if string is None:
            return None
        return string
    
    def _remove_string(self, string):
        '''
        remove the stones that being taken from the grid
        '''
        for point in string.stones:
            self.move_ages.reset_age(point)
            for neighbour in self.neighbour_table[point]:
            # for neighbour in point.neighbours():
                neighbour_string = self._grid.get(neighbour)
                if neighbour_string is None:
                    continue
                if neighbour_string is not string:
                    self._replace_string(neighbour_string.with_liberty(point))
            
            self._grid[point] = None # remove the string of this point
            self._hash ^= zobrist_HASH_CODE[point, string.colour] 
            self._hash ^= zobrist_HASH_CODE[point, None]
    
    def is_self_capture(self, player, point):
        friendly_strings = []
        for neighbour in self.neighbour_table[point]:
            neighbour_string = self._grid.get(neighbour)
            if neighbour_string is None:
                return False
            elif neighbour_string.colour == player:
                friendly_strings.append(neighbour_string)
            else:
                if neighbour_string.num_liberties == 1:
                    return False
        if all(neighbour.num_liberties == 1 for neighbour in friendly_strings):
            return True
        return False

    def will_capture(self, player, point):
        for neighbour in self.neighbour_table[point]:
            neighbour_string = self._grid.get(neighbour)
            if neighbour_string is None:
                continue
            if neighbour_string.colour == player:
                continue
            else:
                if neighbour_string.num_liberties == 1:
                    return True
            

class GameState():
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move
        
        if self.previous_state is None:
            self.previous_states = frozenset()
        else:
            self.previous_states = frozenset(previous.previous_states | {(previous.next_player, previous.board.zobrist_hash())})
            
    def board_state(self):
        pass

    def apply_move(self, move):
        '''
        apply the move:
            - place the stone
            - pass / resign
        '''
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
            
        else:
            next_board = self.board
            
        return GameState(next_board, self.next_player.other, self, move)

    @classmethod
    def new_game(cls, board_size):
        '''
        restart a new game
        '''
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
            
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)

    def is_over(self):
        '''
        determine whether the game is over or continue
            - resign / both pass: game over
        '''
        if self.last_move is None:
            return False
        if self.last_move.is_resign:
            return True
        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        
        # if both sides pass, game over
        return self.last_move.is_pass and second_last_move.is_pass
    
    def is_move_self_capture(self, player, move):
        '''
        self capture - illegal
        '''
        if not move.is_play:
            return False
        return self.board.is_self_capture(player,move.point)
        # next_board = copy.deepcopy(self.board)
        # # make a move first, complete the capture of opponent's stones, and then verify whether it leads to self-capture.
        # next_board.place_stone(player, move.point)
        # new_string = next_board.get_go_string(move.point)
        # # if 0 liberty, the move is illegal, else it is legal
        # return new_string.num_liberties == 0

    def does_move_violate_ko(self, player, move):
        '''
        determine whether the move violate ko
            - simulate the move and check if the situation is similar to the last board's
        '''
        if not move.is_play:
            return False
        if not self.board.will_capture(player, move.point):
            return False
        
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board.zobrist_hash())
        return next_situation in self.previous_states

    def is_valid_move(self, move):
        '''
        if the move is not self-captured and not violate the ko rule
        then it is a valid move
        '''
        if self.is_over():
            return False
        
        if move.is_pass or move.is_resign:
            return True
        
        return self.board.get(move.point) is None and \
                not self.is_move_self_capture(self.next_player,move) and \
                not self.does_move_violate_ko(self.next_player,move)

    def legal_moves(self):
        '''
        Determine the legality of a move with three conditions:
            - the chosen position is unoccupied
            - the move does not result in self-capture
            - the move adheres to the ko rule
        '''
        moves = []
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                move = Move.play(Point(row, col))
                if self.is_valid_move(move):
                    moves.append(move)
                    
        # these moves are always legal
        moves.append(Move.pass_turn())
        moves.append(Move.resign())
        return moves

    def board_matrix(self):
        '''
        get the board state in the matrix format
        '''
        matrix = []
        for row in range(self.board.num_rows, 0, -1):
            rows = []
            for col in range(1,self.board.num_cols+1):
                stone = self.board.get(Point(row=row, col=col))
                if stone is None:
                    rows.append(0)
                elif stone == Player.black:
                    rows.append(1)
                else:
                    rows.append(2)
            matrix.append(rows)
        return matrix

    @property
    def situation(self):
        '''
        finish the move and pass the turn to the opponent
        '''
        return (self.next_player, self.board)

class Agent:
    def __init__(self):
        pass
    def select_move(self, game_state):
        raise NotImplementedError()
    def diagnostics(self):
        return {}

class RandomBot(Agent):
    def select_move(self, game_state):
        '''
        Randomly select a move
        If the valid move is less than 3, then resign
        '''
        candidates = []
        for r in range(1, game_state.board.num_rows + 1):
            for c in range(1, game_state.board.num_cols + 1):
                candidate = Point(row=r, col=c)
                if game_state.is_valid_move(Move.play(candidate)) and not \
                    is_point_an_eye(game_state.board, candidate, game_state.next_player):
                        candidates.append(candidate)
                        
        if len(candidates) <= 5:
            return Move.resign()
        
        # randomly choose one position to make the move
        return Move.play(random.choice(candidates))

# ********************************************            

def is_point_an_eye(board, point, colour):
    '''
    check if the point is in the eye
    '''
    if point is None:
        return True
    if board.get(point) is not None:
        return False
    for neighbour in point.neighbours():
        # all the neighbours have to have the same colour
        # if at least one different colour, then is not an eye
        if board.is_on_grid(neighbour):
            neighbour_colour = board.get(neighbour)
            if neighbour_colour != colour:
                return False
    
    # At least three out of the four diagonal positions are occupied by one's own pieces.
    friendly_corners = 0
    off_board_corners = 0
    corners = [
        Point(point.row - 1, point.col - 1),
        Point(point.row - 1, point.col + 1),
        Point(point.row + 1, point.col - 1),
        Point(point.row + 1, point.col + 1)
    ]
    for corner in corners:
        if board.is_on_grid(corner):
            corner_colour = board.get(corner)
            if corner_colour == colour:
                friendly_corners += 1
                
        else:
            off_board_corners += 1
    if off_board_corners > 0:
        return off_board_corners + friendly_corners == 4
    
    return friendly_corners >= 3

def print_move(player,move):
    '''
    print the player's move in the terminal
    '''
    if move.is_pass:
        move_str = 'passes'
    elif move.is_resign:
        move_str = 'resign'
    else:
        move_str = '%s%d' % (COLS[move.point.col - 1], move.point.row)
    print('%s %s' % (player, move_str))

def print_board(board):
    '''
    Visualize the board in the terminal
    '''
    for row in range(board.num_rows, 0, -1):
        bump = ' ' if row <= 9 else ''
        line = []
        for col in range(1, board.num_cols+1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%s%d %s' % (bump, row, ''.join(line)))
    
    print('    '+'  '.join(COLS[:board.num_cols]))    

 
def to_python(player_state):
    '''
    return the current state of the board - which player's turn
    '''
    if player_state is None:
        return 'None'
    if player_state == Player.black:
        return Player.black
    return Player.white
        
def point_from_coords(coords):
    '''
    Transform the coordinate to Point
    '''
    col = COLS.index(coords[0]) + 1
    row = int(coords[1:])
    return Point(row=row, col=col)

def coords_from_point(point):
    '''
    Transform the Point to coordinates
    '''
    return '%s%d' % (
        COLS[point.col-1],
        point.row
    )

def to_sgf(stones):
    '''
    save the board results in a sgf file
    '''
    dictionary = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',8:'i'}
    path = 'test.sgf'
    
    with open(path,'w') as file:
        file.write(f'(;SZ[9]')
        for [point,player] in stones:
            row = dictionary.get(9-point.row)
            col = dictionary.get(point.col - 1)
            if player == 'black':
                file.write(f';B[{col}{row}]')
            else:
                file.write(f';W[{col}{row}]')
        file.write(')')

def get_zobrist_items():
    return zobrist_HASH_CODE

# ********************************************
COLS = 'abcdefjhi'
STONE_TO_CHAR = {
    None: ' . ',
    Player.black: ' x ',
    Player.white:' o '
}
        
# Use a 64-bit integer to represent each position on the board.
MAX63 = 0x7fffffffffffffff
zobrist_HASH_CODE = {}
zobrist_EMPTY_BOARD = 9181944435492932548

for row in range(1,20):
    for col in range(1,20):
        for state in (None, Player.black, Player.white):
            code = random.randint(0,MAX63)
            zobrist_HASH_CODE[Point(row,col),state] = code
# print('HASH_CODE = {')
# for (pt, state), hash_code in zobrist_HASH_CODE.items():
#     print(' (%r, %s): %r,' % (pt, to_python(state), hash_code))
# print('}')
# print(' ')
# print('EMPTY_BOARD = %d' % (zobrist_EMPTY_BOARD))
       