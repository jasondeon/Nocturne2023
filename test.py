import torch

from utils import *
from tqdm import tqdm
from model import BertModel


# Conditioning variables
MEAN_PITCH = "midlow" # ["low", "midlow", "midhigh", "high"]
VAR_PITCH = "midlow" # ["low", "midlow", "midhigh", "high"]
DEVICE = "cuda:0"

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