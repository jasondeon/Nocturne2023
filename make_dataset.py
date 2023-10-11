import os
import pickle
import pretty_midi as pm

from utils import *
from glob import glob
from tqdm import tqdm


def run_maestro():
    inp_dir = '../Music Speech/MIDI preprocess/MIDI/maestro'
    all_seqs = []
    files = sorted(glob(os.path.join(inp_dir, '*.midi')))
    for f in tqdm(files):
        midi_data = pm.PrettyMIDI(f)
        seq = mid2dat_anna(midi_data=midi_data)
        all_seqs.append(seq)
    with open('maestro.pickle', 'wb') as f:
        pickle.dump(all_seqs, f)


def split_pickle(filename, split=0.8):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    split_idx = int(split * len(data))
    # train split
    with open(f"{filename[:-7]}_train.pickle", "wb") as f:
        pickle.dump(data[:split_idx], f)
    # test split
    with open(f"{filename[:-7]}_test.pickle", "wb") as f:
        pickle.dump(data[split_idx:], f)


if __name__ == "__main__":
    run_maestro()
    #split_pickle("maestro.pickle")