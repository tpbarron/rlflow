
import os
import pickle
import h5py


def create_dir_if_not_exists(record_experience_path):
    if not os.path.isdir(record_experience_path):
        os.mkdir(record_experience_path)


def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def save_h5py(path, states, actions, rewards):
    with h5py.File(path, 'w') as hf:
        hf.create_dataset("states", data=states)
        hf.create_dataset("actions", data=actions)
        hf.create_dataset("rewards", data=rewards)
