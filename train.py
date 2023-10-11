import os
import torch
import wandb
import random
import argparse

from utils import *
from model import BertModel 
from dataset import Maestro


def train(args):
    # Loading
    train_data = Maestro(args.train_file, seq_len=args.seq_len)
    valid_data = Maestro(args.test_file, seq_len=args.seq_len)
    model = BertModel(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Checkpointing
    checkpoint = {
        'step': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }
    if not os.path.isfile(args.checkpoint_path):
        print('Checkpoint not found, writing a new checkpoint...', flush=True)
        torch.save(checkpoint, args.checkpoint_path)
    else:
        print(f'Continuing from checkpoint at {args.checkpoint_path}.')
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Num trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    # Training pass
    train_loss = 0.0 # for printing
    while True:
        model.train()
        batch = train_data.get_batch(args.batch_size)
        tgt = batch['tgt'].to(args.device)
        optimizer.zero_grad()
        out = model(tgt[:,:-1]) # (B,L)
        loss = criterion(out[:,1:,:].reshape(-1, args.vocab_size), tgt[:,2:].reshape(-1))
        loss.backward()
        train_loss += loss.detach()
        checkpoint['step'] += 1
        optimizer.step()

        # Print progress
        if checkpoint['step'] % args.print_every == 0:
            model.eval()
            valid_loss = 0.0 # for printing
            with torch.no_grad():
                for val_i in range(args.print_every):
                    batch = valid_data.get_batch(args.batch_size)
                    tgt = batch['tgt'].to(args.device)
                    out = model(tgt[:,:-1]) # (B,L)
                    valid_loss += criterion(out[:,1:,:].reshape(-1, args.vocab_size), tgt[:,2:].reshape(-1))
            train_loss = train_loss / (args.print_every)
            valid_loss = valid_loss / (args.print_every)

            
            
            wandb.log({
                'step': checkpoint['step'],
                'train_loss': train_loss.detach(),
                'valid_loss': valid_loss.detach(),
            })

            # Checkpoint
            if checkpoint['step'] % (10*args.print_every) == 0:
                checkpoint['model_state_dict'] = model.state_dict()
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                temp_path = os.path.splitext(args.checkpoint_path)[0] + "_temp.pt"
                torch.save(checkpoint, temp_path)
                os.replace(temp_path, args.checkpoint_path)
                print('Checkpoint.', flush=True)

            train_loss = 0.0
            valid_loss = 0.0


def main():
    random.seed(0)
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--embed_dim', type=int, required=True)
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--pretrained', type=int, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--print_every', type=int, required=True)
    args = parser.parse_args()

    wandb.init(
        project="nocturne_maestro",
        config={
            "seq_len": args.seq_len,
            "embed_dim": args.embed_dim,
            "layers": args.layers,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "pretrained": args.pretrained,
        }
    )

    train(args=args)


if __name__ == '__main__':
    main()