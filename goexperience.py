import numpy as np

class ZeroExperienceCollector:
    '''
    Data collection: collecting data from a expisodes of a game or simulation
    '''
    def __init__(self):
        self.states = []
        self.visit_counts = []
        self.rewards = []
        self._current_episode_states = []
        self._current_episode_visit_counts  = []
        
    def begin_episode(self):
        '''
        resets the temporary list
        '''
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        '''
        record the decision point within an episode
        '''
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)
        
    def complete_episode(self, reward):
        '''
        called at the end of an episode
        - add the states and visit counts from the episode to the main lists
        - distributes the episode's reward to all state within the episode
        - reset the temporary lists for the next episode
        '''
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.visit_counts += self._current_episode_visit_counts
        self.rewards += [reward for _ in range(num_states)]
        
        self._current_episode_states = []
        self._current_episode_visit_counts = []
        
class ZeroExperienceBuffer:
    '''
    To store game experience data and provide a method to serialize this data into a file for later use or analysis
    '''
    def __init__(self, states, visit_counts, rewards):
        self.states = states
        self.visit_counts = visit_counts
        self.rewards = rewards
        
    def serialize(self, h5file, type='w'):
        '''
        Saving the stored experience data into an HDF5 file
        '''
        if type == 'a':
            states = np.array(h5file['experience']['states'])
            del h5file['experience']['states']

            visit_counts = np.array(h5file['experience']['visit_counts'])
            del h5file['experience']['visit_counts']

            rewards = np.array(h5file['experience']['rewards'])
            del h5file['experience']['rewards']

            states = np.concatenate([states, self.states],axis=0)
            visit_counts = np.concatenate([visit_counts, self.visit_counts],axis=0)
            rewards = np.concatenate([rewards, self.rewards],axis=0)
            
            h5file['experience'].create_dataset('states', data=states)
            h5file['experience'].create_dataset('visit_counts', data=visit_counts)
            h5file['experience'].create_dataset('rewards', data=rewards)
        else:
            h5file.create_group('experience')
            h5file['experience'].create_dataset('states', data=self.states)
            h5file['experience'].create_dataset('visit_counts', data=self.visit_counts)
            h5file['experience'].create_dataset('rewards', data=self.rewards)


def combine_experience(collectors):
    '''
    Aggregate experience data from multiple collectors into a single 'ZeroExperienceBuffer'
    '''
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_visit_counts = np.concatenate([np.array(c.visit_counts) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
    
    return ZeroExperienceBuffer(
        combined_states, combined_visit_counts, combined_rewards
    )

def load_experience(h5file):
    '''
    Load game experience data from an HDF5 file and encapsulate this data in object 'ZeroExperienceBuffer'
    '''
    return ZeroExperienceBuffer(
        states=np.array(h5file['experience']['states']),
        visit_counts=np.array(h5file['experience']['visit_counts']),
        rewards=np.array(h5file['experience']['rewards'])
    )

    
        