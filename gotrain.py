# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from collections import namedtuple
from encoder import Encoder
# from keras import Input, Model
# from keras.layers import Conv2D,Flatten,Dense
from goboard import draw, go_board

from goexperience import ZeroExperienceCollector, combine_experience
from gorule import GameState, Player, get_zobrist_items
from goscore import GoScore
from gomain import print_board

import torch
import torch.nn as nn
import torch.nn.functional as F

from score_new import compute_game_result


board_size = 9
# Check if a GPU is available and set PyTorch to use the GPU, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# encoder = Encoder(board_size)

class ConvBlock(nn.Module):
    def __init__(self, input_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64,64,kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        for i in range(15):
            x = self.relu(self.conv2(x))
        return x

class PolicyHead(nn.Module):
    def __init__(self, input_channels, num_moves):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(input_channels, 2, kernel_size=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2 * board_size * board_size, num_moves)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        return self.softmax(self.fc(x))

class ValueHead(nn.Module):
    def __init__(self, input_channels):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(board_size * board_size, 256)
        self.fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.tanh(self.fc2(x))

class Network(nn.Module):
    def __init__(self, board_size, num_planes, num_moves):
        super(Network, self).__init__()
        self.board_size = board_size
        self.encoder = Encoder(board_size)  # Move encoder to GPU if you use it actively in computations
        self.conv_blocks = ConvBlock(num_planes).to(device)
        self.policy_head = PolicyHead(64, num_moves).to(device)
        self.value_head = ValueHead(64).to(device)
        self.to(device)

    def forward(self, x):
        # for block in self.conv_blocks:
        #     print(x.shape)
        #     x = block(x)
        x = self.conv_blocks(x)
        policy_output = self.policy_head(x)
        value_output = self.value_head(x)
        return policy_output, value_output

# Usage
def network(board_size):
    num_planes = 11 # Number of planes in the input
    num_moves = board_size * board_size # Adjust based on your game's rules (e.g., number of possible moves)
    model = Network(board_size, num_planes, num_moves).to(device)
    encoder = Encoder(board_size)
    return model,encoder


class GameRecord(namedtuple('GameRecord','moves winner margin')):
    pass

def simulate_game(board_size, black_agent,  white_agent,num_games,temp,simulate):
    # screen,coords = draw(num_games)
    coords = draw(num_games)
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent
    }

    # black_collector.begin_episode()
    # white_collector.begin_episode()
    moves = []
    over = False
    # while not game.is_over() and not over:
    # n = 0
    while not game.is_over() or not over:
        # n+=1
    # while not game.is_over():
        next_move = agents[game.next_player].select_move(game,temp,simulate=simulate)
        print(next_move)
        if next_move is None:
            game.is_over = True
            break
        else:
            moves.append(next_move)
            game = game.apply_move(next_move)
            stones = []
            zobrist_dict = get_zobrist_items()
            board_matrix = [[0 for _ in range(board_size)] for _ in range(board_size)]
            for item,code in zobrist_dict.items():
                point = list(item)[0]
                player = list(item)[1]
                if point.row <= board_size and point.col <= board_size:
                    row,col = point.row, point.col
                    check_on_board = game.board.is_in_zobrist(point)
                    if check_on_board is not None:
                        if check_on_board.colour == Player.black:
                            board_matrix[board_size-row][col-1] = 1
                            stones.append([point,'black'])
                        elif check_on_board.colour == Player.white:
                            board_matrix[board_size-row][col-1] = 2
                            stones.append([point,'white'])
            over = go_board(stones, coords)
        # if n > 40:
        #     over = True

    print_board(game.board)
    game_result = compute_game_result(game)
    print(game_result)

    # print_board(game.board)
    # game_score = GoScore(board_matrix)
    # black_score, white_score = game_score.territory()
    # if black_score > white_score:
    #     winner = "black"
    #     margin = black_score - white_score
    #     # black_collector.complete_episode(1)
    #     # white_collector.complete_episode(-1)
    # else:
    #     winner = "white"
    #     margin = white_score - black_score
        # black_collector.complete_episode(-1)
        # white_collector.complete_episode(1)
    return GameRecord(
        moves = moves,
        winner = game_result.winner,
        margin = game_result.winning_margin
    )

