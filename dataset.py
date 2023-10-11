import torch
import pickle
import random

from utils import *
from torch.utils.data import Dataset


class Maestro(Dataset):
    def __init__(self, pickle_file, seq_len=512):
        super().__init__()
        
        self.pickle_file = pickle_file
        self.seq_len = seq_len

        with open(pickle_file, 'rb') as f:
            self.music_seqs = pickle.load(f)
        
        # discard songs shorter than seq_len
        keep_idx = [i for i in range(len(self.music_seqs)) if len(self.music_seqs[i]) >= self.seq_len]
        self.music_seqs = [self.music_seqs[i] for i in keep_idx]
        
        # record num of songs and song lengths
        self.num_songs = len(self.music_seqs)
        print('Number of songs:', self.num_songs)
        self._weights = torch.tensor([len(self.music_seqs[i]) for i in range(self.num_songs)], dtype=torch.int)


    def augment(self, seq):
        # pitch transposition
        all_pitches = [tok.value if tok.value < 128 else tok.value-128 for tok in seq if tok.type in ('NOTE_ON', 'NOTE_OFF')]
        low = 0 - min(all_pitches) # distance between lowest note and lowest valid value
        high = 127 - max(all_pitches) # distance between highest note and highest valid value
        pitch_change = random.randint(max(low, -3), min(high, 3))
        seq = [MidiToken(tok.type, tok.value + pitch_change) if tok.type in ('NOTE_ON', 'NOTE_OFF') else tok for tok in seq]
        # time stretch
        time_stretch = random.choice([0.95, 0.975, 1.0, 1.025, 1.05])
        seq = [MidiToken(tok.type, int(min(max((((time_stretch * tok.value) + 5) // 10) * 10, 10), 1000))) if tok.type == 'TIME_SHIFT' else tok for tok in seq]
        return seq
    

    def get_batch(self, batch_sz):
        # select random excerpts weighted by song length
        song_idx = torch.multinomial(self._weights.float(), batch_sz, replacement=True)
        song_pos = [torch.randint(0, self._weights[song_idx[i]].item()-self.seq_len+1, size=(1,))[0] for i in range(batch_sz)]
        
        tgt = torch.zeros((batch_sz, self.seq_len), dtype=torch.long)
        for i in range(batch_sz):
            x = self.music_seqs[song_idx[i]][song_pos[i]:song_pos[i]+self.seq_len-2].copy()
            x = self.augment(x)
            pitches = torch.tensor([tok.value for tok in x if tok.type == "NOTE_ON"], dtype=float)
            tgt[i,0], tgt[i,1] = self.bins(torch.mean(pitches), torch.var(pitches))
            tgt[i,2:] = torch.tensor(list(map(MidiToken.key_mapping, x.copy())))
        return {"tgt": tgt}


    def bins(self, mean_pitch, var_pitch):
        # pitch bins (388,389,390,391): 0-60, 60-64, 65-70, 70+ 
        # var bins (392,393,394,395): 0-95, 95-135, 135-190, 190+
        if mean_pitch < 60:
            mean_i = 388
        elif mean_pitch < 65:
            mean_i = 389
        elif mean_pitch < 70:
            mean_i = 390
        else:
            mean_i = 391
        if var_pitch < 95:
            var_i = 392
        elif var_pitch < 135:
            var_i = 393
        elif var_pitch < 190:
            var_i = 394
        else:
            var_i = 395
        return mean_i, var_i



if __name__ == "__main__":
    data = Maestro("maestro_test.pickle", seq_len=512)
    batch = data.get_batch(1)
    print(batch["tgt"])