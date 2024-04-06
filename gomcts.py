import math
import random

from gorule import Agent, Player, RandomBot

class MCTSNode(object):
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.win_counts = {
            Player.black: 0,
            Player.white: 0
        }
        self.num_rollouts = 0
        self.unvisited_moves = game_state.legal_moves()
    
    def add_random_child(self):
        index= random.randint(0,len(self.unvisited_moves)-1)
        new_move = self.unvisited_moves.pop(index)
        new_game_state = self.game_state.apply_move(new_move)
        new_node = MCTSNode(new_game_state,self, new_move)
        self.children.append(new_node)
        return new_node
    
    def record_win(self,winner):
        self.win_counts[winner] += 1
        self.num_rollouts += 1
    
    def can_add_child(self):
        '''
        reports whether this position has any legal moves that
        haven't yet been added to the tree
        '''
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        '''
        whether the game is over at this node
        '''
        return self.game_state.is_over()
    
    def winning_frac(self, player):
        '''
        the fraction of rollouts that were won by a given player
        '''
        return float(self.win_counts[player])/float(self.num_rollouts)
    
class MCTSAgent(Agent):
    def __init__(self, num_rounds, temperature):
        Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature
        
    def select_move(self, game_state):
        root = MCTSNode(game_state)
        
        for i in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)
            # add a new child node into the tree
            if node.can_add_child():
                node = node.add_random_child()
                
            # simulate scores back up the tree
            winner = self.simulate_random_game(node.game_state)
            
            # perpagate scores back up the tree 
            while node is not None:
                node.record_win(winner)
                node = node.parent
            
        scored_moves = [
            (child.winning_frac(game_state.next_player), child.move, child.num_rollouts)
            for child in root.children
        ]
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        for s,m,n in scored_moves[:10]:
            print('%s - %.3f (%d)' % (m,s,n))
            
        # Having performed as many MCTS rounds as we have time for,
        # we now pick a move
        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_frac(game_state.next_player)
            if child_pct > best_pct:
                best_move = child_pct
                best_move = child.move
        print('Select move %s with win pct %.3f'%(best_move, best_pct))
        return best_move
    
    def select_child(self, node):
        '''
        Selecting a branch according to the upper confidence bound for 
        tree (UCT) metric to explore with the UCT formula
        '''
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = math.log(total_rollouts)
        
        best_score = -1
        best_child = None
        
        # Loop over each child
        for child in node.children:
            # calculate the uct score
            win_percentage = child.winning_frac(node.game_state.next_player)
            exploration_factor = math.sqrt(log_rollouts / child.num_rollouts)
            uct_score = win_percentage + self.temperature * exploration_factor
            
            # check if this is the largest we've seen so far
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child
    
    @staticmethod
    def simulate_random_game(game):
        bots = {
            Player.black: RandomBot(),
            Player.white: RandomBot(),
        }
        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        return game.winner()
        
   
def fmt(x):
    if x is Player.black:
        return 'B'
    if x is Player.white:
        return 'W'
    if x.is_pass:
        return 'pass'
    if x.is_resign:
        return 'resign'
    
def show_tree(node, indent='', max_depth=3):
    if max_depth < 0:
        return 
    if node is None:
        return
    if node.parent is None:
        print('%sroot'%indent)
    else:
        player = node.parent.game_state.next_player
        move = node.move
        print('%s%s %s %d %.3f' % (
            indent, fmt(player), fmt(move),
            node.num_rollouts,
            node.winning_frac(player)
        ))
    for child in sorted(node.children, key=lambda n: n.num_rollouts, reverse=True):
        show_tree(child, indent + '  ', max_depth=max_depth - 1)
                        
        
        