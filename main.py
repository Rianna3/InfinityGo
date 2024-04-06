import shutil
import gotraineval
import datetime
import os


if __name__ == "__main__":
    board_size = 9
    # model,encoder = network(board_size)
    # agent = ZeroAgent(model,encoder)
    num_games = 40
    num_workers = 10
    temperature = 0.85
    lr = 0.01
    batch_size = 1024
    work_dir = './training_data/data2'
    log_file = 'training_file'
    # agent1_filename = './data/agent_temp.h5'
    agent1_filename = './training_data/data2/model_temp.pt'
    agent2_filename = './training_data/data2/model_cur.pt'

    logf = open(log_file,'a')
    # logf.write('----------------------------\n')
    # logf.write('Starting from %s at %s \n'% (
    #     agent1_filename, datetime.datetime.now()
    # ))

    temp_decay = 0.99
    min_temp = 0.01

    learning_agent = agent1_filename
    reference_agent = agent2_filename
    # if not os.path.exists(work_dir):
    #   os.makedirs(work_dir)
    experience_file = os.path.join(work_dir,'exp_temp.h5')
    # tmp_agent = os.path.join(work_dir,'agent_temp.h5')
    # working_agent = os.path.join(work_dir,'agent_cur.h5')
    
    tmp_model = os.path.join(work_dir,'model_temp.pt')
    working_model = os.path.join(work_dir,'model_cur.pt')

    # modelfile = './training_data/data1/model.pt'
    total_games = 2630

    while True:
        print('Reference: %s' % (reference_agent,))
        logf.write('Total games so far %d\n'%(total_games))
        gotraineval.generate_experience(
            learning_agent,reference_agent, experience_file,
            num_games=num_games, board_size=board_size,
            num_workers=num_workers,temperature=temperature
        )

        gotraineval.train_on_experience(learning_agent, tmp_model,
                                        experience_file,
                                        lr,batch_size,board_size=board_size)
        total_games += num_games
        wins = gotraineval.evaluate(
            learning_agent, reference_agent, num_games,
            num_workers=num_workers, board_size=board_size,
            temperature=temperature,experience_filename=experience_file
        )
        print('Won %d / %d games (%.3f)' % (
            wins, num_games, float(wins)/num_games
        ))
        logf.write('Won %d / %d games (%.3f)\n'%(
            wins, num_games, float(wins) / num_games
        ))
        # shutil.copy(tmp_agent,working_agent)
        shutil.copy(tmp_model,working_model)
        # learning_agent = working_agent
        learning_agent = working_model
        if wins >= num_games*0.6:
            next_filename = os.path.join(work_dir,
                                'agent_%08d.pt'%(total_games))
            shutil.move(tmp_model,next_filename)
            reference_agent = next_filename
            logf.write('New reference is %s\n'%next_filename)
            temperature = max(min_temp,temp_decay*temperature)
            logf.write('New temperature is %f\n'%temperature)
        else:
            print('Keep learning\n')

        logf.flush()
