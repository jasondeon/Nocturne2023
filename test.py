import os
import torch

from utils import *
from tqdm import tqdm
from model import BertModel


# Conditioning variables
MEAN_PITCH = "midhigh" # ["low", "midlow", "midhigh", "high"]
VAR_PITCH = "high" # ["low", "midlow", "midhigh", "high"]
SAMPLE_LEN = 2 # in minutes
DEVICE = "cuda:0"


def single_sample():
    '''Generate single output of length=seq_len tokens'''
    checkpoint = torch.load("bert_maestro.pt", map_location=DEVICE)
    args = checkpoint["args"]
    model = BertModel(args).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    tok1 = {"low":388, "midlow":389, "midhigh":390, "high":391}[MEAN_PITCH]
    tok2 = {"low":392, "midlow":393, "midhigh":394, "high":395}[VAR_PITCH]
    out = torch.empty((1, 0), dtype=torch.long).to(DEVICE)
    out = torch.cat((out, torch.tensor([[tok1, tok2]], dtype=torch.long, device=DEVICE)), dim=1)

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(args.seq_len-2)):
            logits = model(out)[:,-1,:] # (1, V)
            probs = torch.nn.functional.softmax(logits, dim=1) # (1, V)
            tok = torch.multinomial(probs, 1) # (1, 1)
            out = torch.cat((out, tok), dim=1) # (1, L)
    out = list(map(MidiToken.tok_mapping, out[0,2:]))
    midi = dat2mid_anna(out)
    midi.write("test.mid")


def long_sample():
    '''Generate output until length=SAMPLE_LEN minutes'''
    checkpoint = torch.load("bert_maestro.pt", map_location=DEVICE)
    args = checkpoint["args"]
    model = BertModel(args).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    tok1 = {"low":388, "midlow":389, "midhigh":390, "high":391}[MEAN_PITCH]
    tok2 = {"low":392, "midlow":393, "midhigh":394, "high":395}[VAR_PITCH]
    out = torch.empty((1, 0), dtype=torch.long).to(DEVICE)
    cond_toks = torch.tensor([[tok1, tok2]], dtype=torch.long, device=DEVICE)
    curr_len = 0

    model.eval()
    with torch.no_grad():
        while curr_len < SAMPLE_LEN * 60_000:
            logits = model(torch.cat((cond_toks, out[:,-256:]), dim=1))[:,-1,:] # (1, V)
            probs = torch.nn.functional.softmax(logits, dim=1) # (1, V)
            tok = torch.multinomial(probs, 1) # (1, 1)
            out = torch.cat((out, tok), dim=1) # (1, L)
            midi_token = MidiToken.tok_mapping(tok[0,0])
            if midi_token.type == "TIME_SHIFT":
                curr_len += midi_token.value
                print(f"{curr_len/60_000:3.2f}m out of {SAMPLE_LEN:3.0f}m", end="\r")

    out = list(map(MidiToken.tok_mapping, out[0]))
    midi = dat2mid_anna(out)
    midi.write("output.mid")


def long_sample_primer(input_mid, priming_len):
    checkpoint = torch.load("bert_maestro.pt", map_location=DEVICE)
    args = checkpoint["args"]
    model = BertModel(args).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    tok1 = {"low":388, "midlow":389, "midhigh":390, "high":391}[MEAN_PITCH]
    tok2 = {"low":392, "midlow":393, "midhigh":394, "high":395}[VAR_PITCH]
    #out = torch.empty((1, 0), dtype=torch.long).to(DEVICE)
    cond_toks = torch.tensor([[tok1, tok2]], dtype=torch.long, device=DEVICE)
    
    curr_len = 0
    out = mid2dat_anna(input_mid)
    for i,tok in enumerate(out):
        if tok.type == "TIME_SHIFT":
            curr_len += tok.value
        if curr_len/60_000 >= priming_len:
            out = out[:i]
            break
    out = torch.tensor(list(map(MidiToken.key_mapping, out)), dtype=torch.long).to(DEVICE)
    out = out.unsqueeze(0)
    
    curr_len = 0
    model.eval()
    with torch.no_grad():
        while curr_len < SAMPLE_LEN * 60_000:
            logits = model(torch.cat((cond_toks, out[:,-500:]), dim=1))[:,-1,:] # (1, V)
            probs = torch.nn.functional.softmax(logits, dim=1) # (1, V)
            tok = torch.multinomial(probs, 1) # (1, 1)
            out = torch.cat((out, tok), dim=1) # (1, L)
            midi_token = MidiToken.tok_mapping(tok[0,0])
            if midi_token.type == "TIME_SHIFT":
                curr_len += midi_token.value
                print(f"{curr_len/60_000:3.2f}m out of {SAMPLE_LEN:3.0f}m", end="\r")

    out = list(map(MidiToken.tok_mapping, out[0]))
    midi = dat2mid_anna(out)
    midi.write("output.mid")


if __name__ == "__main__":
    long_sample()