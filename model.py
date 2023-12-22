import torch
import torch.nn as nn

from transformers import AutoModel


class BertModel(nn.Module):
    def __init__(self, args):
        super(BertModel, self).__init__()
        self.pretrained = args.pretrained
        self.bert = AutoModel.from_pretrained(f"google/bert_uncased_L-{args.layers}_H-{args.embed_dim}_A-{args.embed_dim // 64}")
        if not self.pretrained:
            self.bert.apply(self.bert._init_weights)
        self.fc = nn.Linear(args.embed_dim, args.vocab_size)

    def forward(self, inp):
        mask = self.generate_square_subsequent_mask(inp.size(0), inp.size(1)).to(inp.device)
        out = self.bert(input_ids=inp, attention_mask=mask).last_hidden_state
        out = self.fc(out)
        return out

    def generate_square_subsequent_mask(self, n, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)) # (L, L)
        mask = mask.float().unsqueeze(0).repeat(n, 1, 1) # (B, L, L)
        return mask


if __name__ == "__main__":
    from argparse import Namespace
    args_dict = {
        "pretrained": 0,
        "layers": 4,
        "embed_dim": 512,
        "vocab_size": 388,
    }
    args = Namespace(**args_dict)
    model = BertModel(args)
    print(model)
    inp = torch.randint(0, 387, (3,512), dtype=torch.long)
    print(inp.shape)
    out = model(inp)