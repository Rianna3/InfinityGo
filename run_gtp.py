#!/usr/local/bin/python2
import h5py

from frontend import GTPFrontend
from gotraineval import load_agent
from gtp import termination

# model_file = h5py.File("data1/agent_00000034.hdf5", "r")
# modelfile = './training_data/data7/model.pt'
networkfile = './training_data/data2/agent_00010700.pt'
board_size = 9

agent = load_agent(networkfile,board_size)
strategy = termination.get("opponent_passes")
print('The strategy is ',strategy)
termination_agent = termination.TerminationAgent(agent, strategy)

frontend = GTPFrontend(termination_agent)
frontend.run()
