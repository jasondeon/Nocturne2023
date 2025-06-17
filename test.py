import torch
import numpy as np

from utils import *
from model import BertModel

DEVICE = "cuda:0"


def bins(mean_pitch, var_pitch):
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

print("Lib loaded")

sz = 90
x = np.zeros((sz,))
tgt = torch.randint(low=0, high=396, size=(1,sz+2), device=DEVICE)

print("File loaded")

checkpoint = torch.load("bert_maestro.pt", map_location=DEVICE)
args = checkpoint["args"]
model = BertModel(args).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

print("Model loaded")

print(len(x))
model.eval()
with torch.no_grad():
    logits = model(tgt)[0,1:-1,:] # (L, V)
    probs = torch.nn.functional.softmax(logits, dim=1) # (L, V)
    log_likelihoods = torch.log(probs[np.arange(len(x)), tgt[0,2:]])
    print(log_likelihoods)
    print(torch.exp(torch.mean(log_likelihoods)))