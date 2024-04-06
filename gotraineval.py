import argparse
import datetime
import multiprocessing
import os
import random
import shutil
import time
import tempfile
from collections import namedtuple
# from keras import backend as K

import h5py
import numpy as np
import torch
from goexperience import ZeroExperienceCollector, combine_experience, load_experience
# from gopolicyagent import PolicyAgent
from encoder import get_encoder_by_name
from gorule import GameState, Player, Point
from gomain import print_board
from gotrain import simulate_game,network
# from gohtf5 import load_model_from_hdf5_group
from goagent import ZeroAgent, load_zero_model, load_zero_encoder
# import tensorflow as tf
# import keras

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

COLS = 'ABCDEFJHIGKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'O',
}


def load_agent(modelfile,board_size):
    if os.path.exists(modelfile):
        with h5py.File('agent.hdf5', 'r') as h5file:
            encoder = load_zero_encoder(h5file)
        model = load_zero_model(modelfile)
        return ZeroAgent(model,encoder)
    else:
        print('file does not exist')
        model,encoder = network(board_size)
        agent = ZeroAgent(model, encoder)
        torch.save(model,modelfile)
        # with h5py.File(filename, 'w') as f:
        #     agent.serialize(f)
        
        return agent


def avg(items):
    '''
    sum(items)/len(items)
    '''
    if not items:
        return 0.0
    return sum(items)/float(len(items))

def name(player):
    '''
    Return player: black('B') or white('W')
    '''
    if player == Player.black:
        return 'B'
    return 'W'

def get_temp_file():
    '''
    Create a temp file
    - return file path
    '''
    fd,fname = tempfile.mkstemp(prefix='infinity-train')
    os.close(fd)
    fname += '.h5'
    return fname

def collect_collecters(collector_copies,collector):
    for c in range(7):
        collector_copies[c] = collector
    print('The len of the states is:',len(collector_copies[0].states))
    print('The len of the visit counts is:',len(collector_copies[0].visit_counts))
    for k in range(len(collector.states)):
        collector_copies[0].states[k] = np.array([np.rot90(m,k=1,axes=(1,0)) for m in collector.states[k]])
        collector_copies[1].states[k] = np.array([np.rot90(m,k=2,axes=(1,0)) for m in collector.states[k]])
        collector_copies[2].states[k] = np.array([np.rot90(m,k=3,axes=(1,0)) for m in collector.states[k]])
        collector_copies[3].states[k] = np.array([np.rot90(np.flip(m,0),k=-1) for m in collector.states[k]])
        collector_copies[4].states[k] = np.array([np.rot90(np.flip(m),k=-1) for m in collector.states[k]])
        collector_copies[5].states[k] = np.array([np.rot90(np.flip(m),k=-1) for m in collector.states[k]])
        collector_copies[6].states[k] = np.array([np.rot90(np.flip(m),k=-1) for m in collector.states[k]])
    # 7 different types of visit counts for rotated and flipped boards
    c = 0
    for state in collector.visit_counts:
        for x in range(9):
            for y in range(9):
                table = [72-9*x+y,80-x-9*y,8+9*x-y,72-9*y+x,80-9*x-y,8-x+9*y,9*x+y]
                for m in range(7):
                    collector_copies[m].visit_counts[c][table[m]] = state[x+9*y]
        c += 1
    
    collector_copies.append(collector)
    
    return collector_copies
    

def do_self_play(board_size, agent1_filename, agent2_filename,
                 num_games, temperature,
                 experience_filename, gpu_frac):
    # set_gpu_memory_target()
    # tf.device('/CPU:0')

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time())+os.getpid())
    model,encoder = network(board_size)

    # if not (os.path.exists(agent1_filename) and os.path.exists(agent2_filename)):
    #     agent1 = ZeroAgent(model,encoder, rounds_pre_move=10, c=2)
    #     agent2 = ZeroAgent(model,encoder, rounds_pre_move=10, c=2)
    # else:
    agent1 = load_agent(agent1_filename,board_size)
    agent1.set_temperature(temperature)
    agent2 = load_agent(agent2_filename,board_size)
    # with h5py.File(agent1_filename,'r') as agent1f:
    #     # print(agent1f['model'].get('pytorchmodel'))
    #     # print('~~~~~~')
    #     agent1 = load_zero_agent(agent1f)
    # with h5py.File(agent2_filename,'r') as agent2f:
    #     agent2 = load_zero_agent(agent2f)

    collector = ZeroExperienceCollector()
    collector_copies = []
    for i in range(7):
        collector_copies.append(ZeroExperienceCollector())
    
    colour1 = Player.black
    # write = False
    for i in range(num_games):
        print('Simulating game %d-%d...'%(i+1,num_games))
        collector.begin_episode()
        for copy in collector_copies:
            copy.begin_episode()
        agent1.set_collector(collector)

        if colour1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(board_size, black_player, white_player,i,temperature,simulate=True)
        
        # for copy in collector_copies:
        #     copy.states = collector.states
        #     copy.visit_counts = collector.visit_counts
        
        if game_record.winner == colour1:
            print('Agent 1 wins')
            # for collector in collectors:
            #     collector.comlete_episode(reward=1)
            collector.complete_episode(reward=1)
        else:
            print('Agent 2 wins.')
            # for collector in collectors:
                # collector.comlete_episode(reward=-1)
            collector.complete_episode(reward=-1)
        colour1 = colour1.other
        
    # for c in range(7):
    #     collector_copies[c] = collector
    # print('The len of the states is:',len(collector_copies[0].states))
    # print('The len of the visit counts is:',len(collector_copies[0].visit_counts))
    # for k in range(len(collector.states)):
    #     collector_copies[0].states[k] = np.array([np.rot90(m,k=1,axes=(1,0)) for m in collector.states[k]])
    #     collector_copies[1].states[k] = np.array([np.rot90(m,k=2,axes=(1,0)) for m in collector.states[k]])
    #     collector_copies[2].states[k] = np.array([np.rot90(m,k=3,axes=(1,0)) for m in collector.states[k]])
    #     collector_copies[3].states[k] = np.array([np.rot90(np.flip(m,0),k=-1) for m in collector.states[k]])
    #     collector_copies[4].states[k] = np.array([np.rot90(np.flip(m),k=-1) for m in collector.states[k]])
    #     collector_copies[5].states[k] = np.array([np.rot90(np.flip(m),k=-1) for m in collector.states[k]])
    #     collector_copies[6].states[k] = np.array([np.rot90(np.flip(m),k=-1) for m in collector.states[k]])
        
    
    # # 7 different types of visit counts for rotated and flipped boards
    # c = 0
    # for state in collector.visit_counts:
    #     for x in range(9):
    #         for y in range(9):
    #             table = [72-9*x+y,80-x-9*y,8+9*x-y,72-9*y+x,80-9*x-y,8-x+9*y,9*x+y]
    #             for m in range(7):
    #                 collector_copies[m].visit_counts[c][table[m]] = state[x+9*y]
    #     c += 1
            
    # collector_copies.append(collector)
    collector_copies = collect_collecters(collector_copies,collector)
    experience = combine_experience(collector_copies)
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename,'a') as experience_outf:
        experience.serialize(experience_outf)
        print('Successfully saved experience buffer to %s\n' % experience_filename)
        

def generate_experience(learning_agent, reference_agent, exp_file,
                        num_games, board_size, num_workers, temperature):
    '''
    Collect and generate Go experience data
    '''
    experience_files = []
    workers = []
    gpu_frac = 0.95/float(num_workers)
    games_per_worker = num_games//num_workers
    
    ctx = torch.multiprocessing.get_context('spawn')
    
    # start multi processors
    for i in range(num_workers):
        # print(i)
        filename = get_temp_file()
        # print(filename)
        experience_files.append(filename)
        worker = multiprocessing.Process(
            target=do_self_play,
            args=(
                board_size,
                learning_agent,
                reference_agent,
                games_per_worker,
                temperature,
                filename,
                gpu_frac,
            )
        )
        worker.start()
        workers.append(worker)

    print('Waiting for workers...')
    for worker in workers:
        worker.join()

    # merge experience buffers 
    print('Merging experience buffers...')
    first_filename = experience_files[0]
    other_filename = experience_files[1:]
    # first_filename += '.h5'
    # print(first_filename,other_filename)
    # save the merged experience
    # if os.path.exists(first_filename):
    print('All experienced files:', experience_files)
    with h5py.File(first_filename,'r') as expf:
        combined_buffer = load_experience(expf)
        expf.close()
    # else:
    #   with h5py.File(first_filename,'w') as expf:
    #       combined_buffer = None
    if len(other_filename) > 0:
        for filename in other_filename:
            with h5py.File(filename, 'r') as expf:
                next_buffer = load_experience(expf)
                expf.close()
            combined_buffer = combine_experience([combined_buffer,next_buffer])
    print('combined_buffer:',combined_buffer)
    print('Saving into %s...'%exp_file)
    with h5py.File(exp_file,'a') as experience_outf:
        # if combined_buffer is not None:
        combined_buffer.serialize(experience_outf,type='a')
    
    # clean up the temporary files
    for fname in experience_files:
        os.unlink(fname)

def train_worker(learning_agent, output_file, experience_file,
                 lr, batch_size, board_size):
    '''
    Load AI agent and train it using the collected experience
    '''
    # with h5py.File(learning_agent, 'r') as learning_agentf:
    #   learning_agent = load_zero_agent(learning_agentf)
    learning_agent = load_agent(learning_agent,board_size)
    with h5py.File(experience_file,'r') as expf:
        exp_buffer = load_experience(expf)
    print('The loaded experienced buffer: ',exp_buffer)
    learning_agent.train(exp_buffer,learning_rate=lr,batch_size=batch_size)

    # with h5py.File(output_file,'w') as updated_agent_outf:
        # learning_agent.serialize(updated_agent_outf)
    torch.save(learning_agent.model,output_file)

def train_on_experience(learning_agent, output_file, experience_file,
                        lr, batch_size,board_size):
    '''
    Train a learning agent in a seperate background process,
    using experience data collected from the game
    '''
    worker = multiprocessing.Process(
        target=train_worker,
        args = (
            learning_agent,
            output_file,
            experience_file,
            lr,
            batch_size,
            board_size
        )
    )
    worker.start()
    worker.join()

def play_games(agent1_name,agent2_name,num_games,board_size, experience_filename, temperature,q):
    '''
    Automatically simulate a series of Go games between two AI agents,
    record the wins and losses for the first agent, and adjust GPU memory
    uasge as required
    '''
    # agent1_name,agent2_name, num_games, board_size, experience_filename, temperature = args
    # set_gpu_memory_target()
    # import tensorflow as tf
    # config = tf.compat.v1.ConfigProto(device_count={'GPU':1})
    # sess = tf.compat.v1.Session(config=config)
    # tf.device('/CPU:0')

    random.seed(int(time.time())+os.getpid())
    np,random.seed(int(time.time())+os.getpid())

    agent1 = load_agent(agent1_name,board_size)
    agent2 = load_agent(agent2_name,board_size)
    
    collector = ZeroExperienceCollector()
    collector_copies = []
    for i in range(7):
        collector_copies.append(ZeroExperienceCollector())
        

    wins, losses = 0,0
    colour1 = Player.black
    print('Start simulating...')
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i+1,num_games))
        
        collector.begin_episode()
        for copy in collector_copies:
            copy.begin_episode()
        agent1.set_collector(collector)
        
        if colour1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(board_size,black_player,white_player,i,temperature,simulate=False)
        if game_record.winner == colour1:
            print('Agent 1 wins')
            collector.complete_episode(reward=1)
            wins += 1
        else:
            print('Agent 2 wins')
            collector.complete_episode(reward=-1)
            losses += 1
        print('Agent 1 record: %d/%d'%(wins, wins+losses))
        colour1 = colour1.other
        
    collector_copies =  collect_collecters(collector_copies,collector)
    experience = combine_experience(collector_copies)
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename,'w') as experience_outf:
        experience.serialize(experience_outf)
        print('Successfully saved experience buffer to %s\n' % experience_filename)
    q.put((wins,losses))
        
    # return wins,losses

def evaluate(learning_agent, reference_agent, num_games, num_workers,
             board_size, temperature,experience_filename):
    '''
    evaluate the performance of learning agents 
    relative to reference agents
    '''
    print('Enter the simulation function...')
    games_per_worker = num_games//num_workers
    gpu_frac = 0.95/float(num_workers)
    # game_results = []
    experience_files = []
    workers = []
    q = multiprocessing.Queue()
    
    for _ in range(num_workers):
        filename = get_temp_file()
        experience_files.append(filename)
        worker = multiprocessing.Process(
            target=play_games,
            args=(
                learning_agent, reference_agent,
                games_per_worker, board_size, filename,
                temperature, q
            )
        )
        worker.start()
        workers.append(worker)
        
    print('Waiting for workers...')
    for worker in workers:
        worker.join()
        
    # merge experience buffers 
    print('Merging experience buffers...')
    first_filename = experience_files[0]
    other_filename = experience_files[1:]
    print('All experienced files:', experience_files)
    with h5py.File(first_filename,'r') as expf:
        combined_buffer = load_experience(expf)
        expf.close()

    if len(other_filename) > 0:
        for filename in other_filename:
            with h5py.File(filename, 'r') as expf:
                next_buffer = load_experience(expf)
                expf.close()
            combined_buffer = combine_experience([combined_buffer,next_buffer])
    print('combined_buffer:',combined_buffer)
    print('Saving into %s...'%experience_filename)
    with h5py.File(experience_filename,'w') as experience_outf:
        combined_buffer.serialize(experience_outf)
    
        
    game_results = [q.get() for j in workers]
        
    # print('game_result:',game_result)
    # print('game_results:',game_results)
        
    # pool = multiprocessing.Pool(num_workers)
    # worker_args = [
    #     (
    #         learning_agent, reference_agent,
    #         games_per_worker, board_size, gpu_frac,
    #         temperature
    #     )
    #     for _ in range(num_workers)
    # ]
    # game_result = pool.map(play_games, worker_args)
    
    # worker_args = (learning_agent,reference_agent,
    #                games_per_worker,board_size,gpu_frac,
    #                temperature)
    # game_result = play_games(worker_args)

    total_wins, total_losses = 0,0
    for (wins, losses) in game_results:
        total_wins += wins
        total_losses += losses
    print('FINAL RESULTS:')
    print('Learner: %d' % total_wins)
    print('Reference: %d' % total_losses)
    # pool.close()
    # pool.join()
    return total_wins

def main():
    board_size = 9
    # model,encoder = network(board_size)
    # agent = ZeroAgent(model,encoder)
    games_per_batch = 5
    num_workers = 1
    temperature = 1
    lr = 0.01
    batch_size = 1024
    work_dir = './data'
    log_file = 'training_file'
    model = 'model_file'
    # agent1_filename = './data/agent_cur.h5'
    # agent2_filename = './data/agent_cur.h5'


    logf = open(log_file,'a')
    logf.write('----------------------------\n')
    # logf.write('Starting from %s at %s \n'% (
    #     agent1_filename, datetime.datetime.now()
    # ))

    temp_decay = 0.98
    min_temp = 0.01

    # learning_agent = agent1_filename
    # reference_agent = agent2_filename
    # if not os.path.exists(work_dir):
    #   os.makedirs(work_dir)
    experience_file = os.path.join(work_dir,'exp_temp.h5')
    
    tmp_model = os.path.join(work_dir,'model_temp.pt')
    working_model = os.path.join(work_dir,'model_cur.pt')
    
    tmp_agent = os.path.join(work_dir,'agent_temp.h5')
    working_agent = os.path.join(work_dir,'agent_cur.h5')
    modelfile = os.path.join(work_dir,model)
    total_games = 0

    while True:
        print('Reference: %s' % (reference_agent,))
        logf.write('Total games so far %d\n'%(total_games))
        generate_experience(
            working_agent,reference_agent,experience_file, modelfile,
            num_games=games_per_batch, board_size=board_size,
            num_workers=num_workers,temperature=temperature
        )

        train_on_experience(working_agent, tmp_agent, modelfile, experience_file,
                  lr,batch_size,board_size=board_size)
        total_games += games_per_batch
        wins = evaluate(
            working_agent, reference_agent, modelfile, num_games=games_per_batch,
            num_workers=num_workers, board_size=board_size,
            temperature=temperature, experience_filename=experience_file
        )
        print('Won %d / 400 games (%.3f)' % (
            wins, float(wins)/100
        ))
        logf.write('Won %d / 400 games (%.3f)\n'%(
            wins, float(wins) / 100
        ))
        shutil.copy(tmp_agent,working_agent) # copy the tmp_agent to working agent
        # learning_agent = working_agent
        if wins >= 2:
            next_filename = os.path.join(work_dir,
                                'agent_%08d.hdf5'%(total_games))
            shutil.move(tmp_agent,next_filename)
            reference_agent = next_filename
            logf.write('New reference is %s\n'%next_filename)
            temperature = max(min_temp,temp_decay*temperature)
            logf.write('New temperature is %f\n'%temperature)
        else:
            print('Keep learning\n')

        logf.flush()

# if __name__ == '__main__':
#     main()

