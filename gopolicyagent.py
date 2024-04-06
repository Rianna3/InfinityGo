import numpy as np
from keras import backend as K
from keras.optimizers import SGD

from gorule import is_point_an_eye
from encoder import Encoder


class Agent:
    def __init__(self):
        pass
    def selct_move(self,game_state):
        raise NotImplementedError()
    def diagnostics(self):
        return {}
    
def policy_gradient_loss(y_true, y_pred):
    '''
    Calculate loss function
    '''
    clip_pred = K.clip(y_pred, K.epsilon(),1-K.epsilon())
    loss = -1*y_true*K.log(clip_pred)
    return K.mean(K.sum(loss,axis=1))

def normalize(x):
    '''
    Normalization
    '''
    total = np.sum(x)
    return x/total

class PolicyAgent(Agent):
    '''
    An agent that uses a deep policy network to select moves
    '''
    def __init__(self,model,encoder):
        Agent.__init__(self)
        self._model = model
        self._encoder = encoder
        self._collector = None
        self._temperature = 0.0
        
    def predict(self, game_state):
        encoded_state = self._encoder.encoder(game_state)
        input_tensor = np.array([encoded_state])
        return self._model.predict(input_tensor)[0]

