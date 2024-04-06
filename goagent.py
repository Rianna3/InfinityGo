import collections.abc
from copy import deepcopy
import random

import torch
import torch.optim as optim
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
# #Now import hyper
# import hyper
import numpy as np
# from keras.optimizers import SGD
from encoder import Encoder

from gorule import Agent, is_point_an_eye

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Branch:
    '''
    record the simulation rounds:
        - the move
        - the number of visits
        - the total value of visits
    '''
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0

class ZeroTreeNode:
    def __init__(self, state, value, priors, parent, last_move):
        self.state = state # 当前游戏状态
        self.value = value # 预测得分
        self.parent = parent # 当前node的parent
        self.last_move = last_move # 最后一步
        self.total_visit_count = 1
        self.branches = {}
        moves = []
        for move, p in priors.items():
            '''
            record the valid branch moves
            '''
            # print(move.point)
            # if move.point != None:
            # moves.append(move.point)
            if state.is_valid_move(move) and \
                not is_point_an_eye(state.board,move.point, state.next_player):
                self.branches[move] = Branch(p)
        self.children = {}
        
    def moves(self):
        '''
        return all the moves
        '''
        return self.branches.keys()
    
    def add_child(self, move, child_node):
        '''
        add the child_node as the children node through this move
        '''
        self.children[move] = child_node
        
    def has_child(self, move):
        '''
        check whether the move has child or not
        -> check last move
        '''
        return move in self.children
    
    def get_child(self, move):
        '''
        return all the children of the move
        '''
        return self.children[move]
    
    def record_visit(self, move, value):
        '''
        record the visit of the move
        - total visit + 1
        - visit + 1
        - total value + value
        '''
        if move is not None:
            self.total_visit_count += 1
            self.branches[move].visit_count += 1
            self.branches[move].total_value += value

    #  helper functions
    def expected_value(self, move):
        '''
        return the expected value of the move
        '''
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count
    
    def prior(self, move):
        '''
        return the prior move
        '''
        return self.branches[move].prior
    
    def visit_count(self, move):
        '''
        return the visit time of the move
        '''
        if move in self.branches:
            return self.branches[move].visit_count
        return 0
    
class ZeroAgent(Agent):
    '''
    Q + cP(square_root(N)/(1+n))
    '''
    def __init__(self, model, encoder, rounds_pre_move=20, c=10):
        self.model = model.to(device)
        self.encoder = encoder
        self.num_rounds = rounds_pre_move # n_playout
        self.c = c # c_punct
        self.temperature = 0
        
        self.collector = None
        
    def set_temperature(self, temperature):
        self.tmperature = temperature
        
    def set_collector(self, collector):
        self.collector = collector
    
    def select_move(self, game_state, temp, simulate=False):
        '''
        Select the move with the highest score
        '''        
        root = self.create_node(game_state)
        # forward process
        for i in range(self.num_rounds):
            node = root
            next_move = self.select_branch(node)
            if next_move is None:
                return None
            # continue choose the next branch until reach a branch with no children
            last_node = node
            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)
                if next_move is None:
                    return None
            # if next_move is None:
            #     print(original)
            #     print(next_move)
            
            # backpropagation process 
            # walk back up the tree and update the statistics for each parent that lead to this node
            new_state = node.state.apply_move(next_move)
            child_node = self.create_node(new_state,move=next_move,parent=node)
            move = next_move
            # The perspective switches between the black and white player,
            # need to flip the sign of the value at each step
            value = -1*child_node.value
            while node is not None: # loop until hit the root node
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value = -1*value
        
        
        if self.collector is not None:
            '''
            Passing along the decision to the experience collector
            '''
            # record 8 different states using rotations and flip
            
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([
                root.visit_count(self.encoder.decode_move_index(idx))
                for idx in range(self.encoder.num_moves())
            ])
            self.collector.record_decision(root_state_tensor,visit_counts)

        act_visits = [(act, root.visit_count(act))
                      for act in root.moves()]
        acts, visits = zip(*act_visits) 
        # print(np.log(np.array(visits)+1e-10))
        act_probs = torch.FloatTensor(1.0/temp * np.log(np.array(visits)+1e-10))
        act_probs = torch.nn.functional.softmax(act_probs,dim=0)
        if simulate:
            modified_probs = 0.2*act_probs+0.8*np.random.dirichlet(0.3*np.ones(len(act_probs)))
        else:
            modified_probs = act_probs
        normalized_probs = modified_probs / modified_probs.sum()
        move = np.random.choice(acts, 
                                p=normalized_probs)
              
        # return max_moves
        # return random.choice(max_moves)
        return move
    
                    
    def create_node(self, game_state, move=None, parent=None):
        '''
        Create a new node for this move\n
        take the previous game state and apply the current move to get a new game state\n
        Return the new state tree
        '''
        
        # 结合之前的游戏状态以及当前的决策，并放入神经网络进行预测
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])
        
        if not isinstance(model_input, torch.Tensor):
            model_input = torch.tensor(model_input,dtype=torch.float).to(device)
        priors,values = self.model(model_input)
        
        # priors, values = self.model.predict(model_input)
        priors = priors[0] # 下一步的所有先验估计
        value = values[0][0] # 新状态的预估值
        move_priors = { # 将预测的策略都放到字典中
            self.encoder.decode_move_index(idx): p
            for idx,p in enumerate(priors)
        }
        new_node = ZeroTreeNode(
            game_state, value, move_priors, parent, move
        )
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node
    
    def select_branch(self, node):
        '''
        select branch with highest score
        function: score_branch: return the score of the branch
        '''
        total_n = node.total_visit_count
        
        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            # if n/total_n > 0.1:
            #     return -1
            return q + self.c * p * np.sqrt(total_n) / (n+1)
        if len(node.moves()) == 0:
          return None
        else:
            return max(node.moves(),key=score_branch)
    
    # def train(self, experience, learning_rate, batch_size):
    #     '''
    #     Training process
    #     '''
    #     num_examples = experience.states.shape[0]
        
    #     model_input = experience.states
        
    #     # normalize the visit counts: 
    #     # original visit counts / total visit count => normalized visit counts
    #     visit_sums = np.sum(experience.visit_counts, axis=1).reshape(
    #         (num_examples,1)
    #     )
        
    #     action_target = experience.visit_counts / visit_sums
    #     value_target = experience.rewards
        
    #     self.model.compile(
    #         SGD(lr=learning_rate),
    #         loss=['categorical_crossentropy','mse']
    #     )
        
    #     self.model.fit(model_input,[action_target, value_target],
    #                    batch_size=batch_size)
    def train(self, experience, learning_rate, batch_size):
        '''
        Training process
        '''            
        self.model.train()  # Set the model to training mode
        # num_examples = experience['states'].shape[0]
        num_examples = experience.states.shape[0]
        print('The number of examples is %s'%experience.states.shape[0])

        # Convert data to PyTorch tensors
        # model_input = torch.tensor(experience['states'], dtype=torch.float).to(device)
        # action_target = torch.tensor(experience['visit_counts'], dtype=torch.float).to(device)
        # value_target = torch.tensor(experience['rewards'], dtype=torch.float).view(-1, 1).to(device)  # Ensure correct shape
        model_input = torch.tensor(experience.states, dtype=torch.float).to(device)
        action_target = torch.tensor(experience.visit_counts, dtype=torch.float).to(device)
        value_target = torch.tensor(experience.rewards, dtype=torch.float).view(-1, 1).to(device)  # Ensure correct shape


        # Normalize the visit counts
        visit_sums = action_target.sum(dim=1, keepdim=True)
        action_target /= visit_sums

        # Define the optimizer and loss functions
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        policy_loss_function = torch.nn.CrossEntropyLoss()  # Adjust if your action targets are not class indices
        value_loss_function = torch.nn.MSELoss()

        # Calculate number of batches
        num_batches = int(np.ceil(num_examples / batch_size))

        for epoch in range(10):  # Loop over the dataset multiple times if needed
            running_loss = 0.0
            for i in range(num_batches):
                # Get the mini-batch
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, num_examples)
                inputs = model_input[start_index:end_index]
                actions = action_target[start_index:end_index]
                values = value_target[start_index:end_index]

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                policy_output, value_output = self.model(inputs)

                # Compute the loss
                _, actions_indices = actions.max(dim=1)
                policy_loss = policy_loss_function(policy_output, actions_indices)
                value_loss = value_loss_function(value_output, values)
                total_loss = policy_loss + value_loss

                # Backward pass and optimize
                total_loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += total_loss.item()
                if i % 8 == 0:  # Print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

        print('Finished Training')

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        # h5file.create_group('model')
        # save_model_to_hdf5_group(self.model,h5file['model'])
        # print('The type of the model is ' ,self.model.state_dict())
        # for name, params in self.model.state_dict().items():
        #     h5file['model'].attrs[name] = params
        # print('The type of the h5file',type(h5file['model']))
        
def load_zero_encoder(h5file):    
    encoder_name = h5file['encoder'].attrs['name']
    # decode the encoder_name
    if not isinstance(encoder_name,str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    # print('The name of the encoder is:',board_height,board_width)
    # encoder = get_encoder_by_name(encoder_name,
    #                 (board_width,board_height))
    encoder = Encoder(board_width)
    return encoder

def load_zero_model(modelfile):
    model = torch.load(modelfile).to(device)
    model.eval()
    return model

# ZeroAgent.train()    
 