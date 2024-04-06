import os
import random
import time
import sgf
import numpy as np
from copy import copy

from tqdm import tqdm

from gorule import GameState, Move, Point
from gotrain import GameRecord, network
from score_new import compute_game_result

class SGFLoader(object):
    def __init__(self, filename):
        self.open(filename)
        
    def open(self,filename):
        self.filepath = filename
        with open(filename,'r') as f:
            self.sgf = sgf.parse(f.read())
        self.nodes = self.sgf.children[0].nodes
        self.root = self.sgf.children[0].root
        self.total = len(self.nodes)
        self.step = 0
        self.cur = self.root
        self.end = False
        # print('self.nodes:\n',self.nodes)
        # print('--------------------------')
        # print('self.root:\n',self.root)
        # print('--------------------------')
        # print('self.total:\n',self.total)
    def collect_moves(self):
        moves = []
        for node in self.nodes:
            keys = list(node.properties.keys())
            if keys[0] == 'B':
                sgf_coord = node.properties['B']
                moves = add_moves(sgf_coord[0],moves)
                
            elif keys[0] == 'W':
                sgf_coord = node.properties['W']
                moves = add_moves(sgf_coord[0],moves)
        return moves

def add_moves(sgf_coord,moves):
    position = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9}
    if len(sgf_coord) == 2:
        col,row = sgf_coord[0], sgf_coord[1]
        point = Point(position[row],position[col])
        move = Move(point)
    elif len(sgf_coord) == 0:
        move = Move(is_pass=True)
    moves.append(move)
    return moves

def load_game(sgf_file):
    loader = SGFLoader(sgf_file)
    moves = loader.collect_moves()
    
    game = GameState.new_game(9)
    
    for move in moves:
        next_move = move
        game = game.apply_move(next_move)
    game_result = compute_game_result(game)
    return GameRecord(moves=moves,
                      winner=game_result.winner,
                      margin=game_result.winning_margin)
    
    
sgf_folder = './sgf_data'
sgf_files = os.listdir(sgf_folder)
i=0

for sgf_file in sgf_files:
    sgf_file = os.path.join(sgf_folder, sgf_file)

def do_human_play(board_size,agent1_filename,agent2_filename,
                 num_games, temperature,
                 experience_filename, gpu_frac):
    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time())+os.getpid())
    model,encoder = network(board_size)
    
    
# loader = SGFLoader('./1489666.sgf')

# import shutil
# for i in range(1,12):
#     if i < 10:
#         i = f'0{i}'
#     filepath = f"C:/Users/86187/Downloads/9x9_2022_{i}/2022/{i}"
#     destination = "./sgf_data"
#     dirs = os.listdir(filepath)
#     print(f'copying file {filepath}')
#     for file in tqdm(dirs):
#         childfile = os.path.join(filepath,file)
#         sgfs = os.listdir(childfile)
#         for sgf_file in sgfs:
#             sgf_file = os.path.join(childfile,sgf_file)
#             shutil.copy(sgf_file,destination)
        