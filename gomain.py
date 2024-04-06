import copy
import h5py
from goagent import ZeroAgent, load_zero_encoder, load_zero_model
from gorule import Move, print_board,print_move,point_from_coords, to_sgf, get_zobrist_items
from gorule import GameState,Player,RandomBot,Point,COLS
from goscore import GoScore
from goboard import go_board, draw
# from gomcts import MCTSAgent
import time

def random_bot():
    screen, coords = draw(1)
    board_size = int(len(COLS))
    game = GameState.new_game(board_size)
    
    '''
    BOT vs BOT
    '''
    with h5py.File('./data3/agent_00001340.hdf5','r') as f:
        encoder = load_zero_encoder(f)
    model = load_zero_model('./data1/model.pt')
    bot = ZeroAgent(model, encoder)
    while not game.is_over():
        time.sleep(0.3)
        print(chr(27)+'[2]')
        print_board(game.board)
        
        if game.next_player == Player.black:
            human_move = input('--')
            point = point_from_coords(human_move.strip())
            move = Move.play(point)
        else:
            move = bot.select_move(game)
            point = point_from_coords('%s%d' % (COLS[move.point.col -1], move.point.row))
            print(point)
        print_move(game.next_player,move)
        game = game.apply_move(move)
            
    
    
    # stones_no_remove = []
    # over = False
    
    # while not game.is_over():
    #     time.sleep(0.5)
    #     print(chr(27)+'[2J]')
    #     print_board(game.board)
    #     print(game.board)
        
    #     bot_move = bots[game.next_player].select_move(game) # randomly select a move
    #     if bot_move.point is not None:
    #         point = point_from_coords('%s%d' % (COLS[bot_move.point.col - 1], bot_move.point.row)) # point = Point(row, col)
    #         print('player',game.next_player)
    #         if game.next_player == Player.black:
    #             player = "black"
    #         else:
    #             player = "white"
    #         # stones.append([point,player]) # stones: records placing stones
    #         stones_no_remove.append([point,player]) # for sgf use
        

    #     print_move(game.next_player, bot_move)
    #     game = game.apply_move(bot_move) # apply this move
        
    #     # get the state of the board in a matrix format
    #     matrix = game.board_matrix()
    #     stones = []
    #     for i in range(9):
    #         for j in range(9):
    #             if matrix[i][j] == 1:
    #                 stones.append([Point(row=9-i,col=j+1),'black'])
    #             if matrix[i][j] == 2:
    #                 stones.append([Point(row=9-i,col=j+1),'white'])
        
    #     over = go_board(stones,coords,screen)
        
    #     if over:
    #         time.sleep(0.5)
    #         print(chr(27)+'[2J]')
    #         print_board(game.board)
    #         print(game.board)
    #         break
             
    
    # if game.is_over():
    #     go_score = GoScore(matrix)
    #     black_score, white_score = go_score.territory()
        
    #     # black_score, white_score = GoScore.go_score(white_territory_list, black_territory_list, white_stones, black_stones)
    #     print(f'White Player: {white_score}')
    #     print(f'Black Player: {black_score}')
    #     if black_score > white_score:
    #         print(f'Black Pleyer Win: {(black_score-white_score)/2}')
    #     elif black_score < white_score:
    #         print(f'White Player Wins: {(white_score-black_score)/2}')
    #     else:
    #         print('Draw')
    
    # to_sgf(stones_no_remove)

if __name__ == '__main__':
    random_bot()        