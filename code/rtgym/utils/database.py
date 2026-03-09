import os
import pickle
import tqdm
from ..gym import RatatouGym

def generate_database(
        gym,
        path,
        n_trials,
        trial_duration,
        ):
    """
    Generate a database of sensory responses
    @param gym: gym object
    @param path: path to save the database
    @param n_trials: number of trials
    @param trial_duration: duration of each trial
    """
    # check if path is a directory and if not, create it
    start_idx = 0
    if not os.path.exists(path):
        os.makedirs(path)
        # save gym object
        with open(os.path.join(path, 'gym.pkl'), 'wb') as f:
            pickle.dump(gym, f)
    else:
        if os.listdir(path):
            # load gym object
            gym = None
            with open(os.path.join(path, 'gym.pkl'), 'rb') as f:
                gym = pickle.load(f)
                assert gym is not None, "gym object not loaded"
                print("Loaded gym object from {}".format(os.path.join(path, 'gym.pkl')))
            # find the last trial index
            for file in os.listdir(path):
                if file.startswith('trial_'):
                    file_idx = int(file.split('_')[1].split('.')[0])
                    start_idx = max(start_idx, file_idx + 1)
            print("Start generating trials from trial_{}".format(start_idx))

    # generate trials
    for i in tqdm.tqdm(range(start_idx, n_trials)):
        _, sm_responses, mm_responses = gym.(
            duration=trial_duration,
            )
        trial = {
            'sm_responses': sm_responses,
            'mm_responses': mm_responses,
            }
        with open(os.path.join(path, 'trial_{}.pkl'.format(i)), 'wb') as f:
            pickle.dump(trial, f)
    

class GymDataloader:
    def __init__(self, path):
        """
        @param path: path to the database
        """
        self.path = path
    

    def get_gym(self):
        return self._get_gym(), self._get_trial_info()


    def _get_trial_info(self):
        # check how many trials are there
        n_trials = 0
        for file in os.listdir(self.path):
            if file.startswith('trial_'):
                n_trials += 1
        
        # check the trial durations and keys
        trial_dict = self.get_trial(0)
        trial_duration = trial_dict['sm_responses'].shape[0]

        return {
            'n_trials': n_trials,
            'trial_duration': trial_duration,
            }


    def _get_gym(self):
        with open(os.path.join(self.path, 'gym.pkl'), 'rb') as f:
            gym = pickle.load(f)
        return gym
    

    def get_trial(self, trial_idx):
        with open(os.path.join(self.path, 'trial_{}.pkl'.format(trial_idx)), 'rb') as f:
            trial = pickle.load(f)
        return trial