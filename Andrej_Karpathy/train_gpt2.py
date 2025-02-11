from dataclasses import dataclass
import torch 
import torch 
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

### 



class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # no really a bias, more of a mask but following the openAU/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    

    def forward(self,x):
        B,T,C=x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query,key, values for all heads in batch and move head forward to tbe the batch
        # nh is "number of heads", hs is "head size", and C (number of channels)=nh*hs
        #examplein gpt-2 n_head=12, hs=64, so nh+hs=768 channels in the transformer
        qkv=self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B, nh, T, hs
        q=q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v=v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materialized the large (t,t)) matrix for all the queries and keys)
        att=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        att=att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        att=F.softmax(att, dim=-1)
        y=att@v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y=y.transpose(1, 2).contiguous().view(B, T, C) # reassemble all head outputs side by side
        # output projection
        y=self.c_proj(y)
        return y





class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gleu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1
    def forward(self, x):
        x=self.c_fc(x)
        x=self.gleu(x)
        x=self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.mlp =MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # this is where they communicate
        x = x + self.mlp(self.ln_2(x)) # this is where they reflect on the information that they have gathered
        return x


@dataclass
class GPTConfig:
    block_size: int = 128 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1<|endofsentence|>
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embeddnig dimension

## what is a wrapper funtcion 
class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config=config
        self.transformer = nn.ModuleDict(dict(            
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),# position embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    

        # weight sharing scheme
        self.transformer.wte.weight=self.lm_head.weight
    

        # init params
        self.apply(self._init_weights)
    def _init_weights(self, module):
        std=0.02
        if hasattr(module,'NABOGPT_SCALE_INIT'):
            std *= (2*self.config.n_layer)**(-0.5)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self,idx, targets=None):
        # idx is of shape (B,T)
        B,T=idx.size()
        assert T<=self.config.block_size, "Cannot forward, model block size is exhausted."
        # forward the token and position embeddings
        pos= torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb=self.transformer.wpe(pos) # shape (T, C)
        token_emb=self.transformer.wte(idx) # shape (B, T, C)
        x=token_emb+pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x=block(x)
        # forward the final layernorm and the classifier
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)

        x=self.transformer.ln_f(x)
        logits=self.lm_head(x) # shape (B, T, vocab_size)
        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # forward the final layernorm and the classifier
        
        return logits,loss




    @classmethod
    def from_pretrained(cls,model_type):
        """ loads pretrained GPT-2 model weights from higging face"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        
        # n_layers, n_heads and n_embd are determined from model_type
        config_args={
            'gpt2': {'n_layer': 12, 'n_head': 12, 'n_embd': 768}, # 124M params
            'gpt2-medium': {'n_layer': 24, 'n_head': 16, 'n_embd': 1024}, # 350M params
            'gpt2-large': {'n_layer': 36, 'n_head': 20, 'n_embd': 1280}, # 774M params
            'gpt2-xl': {'n_layer': 48, 'n_head': 25, 'n_embd': 1600}, # 1558M params'
            }[model_type] # what is the meaning of this line
        
        config_args['vocab_size']=50257 # always 50257 for GPT model checkpoints
        config_args['block_size']=1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config_args = GPTConfig(**config_args) # meaning of this
        model = cls(config_args)
        sd=model.state_dict()
        sd_keys=sd.keys()
        sd_keys=[k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a higgingface/transfomers model
        model_hf=GPT2LMHeadModel.from_pretrained(model_type)

        # coping while ensuring all of the parameters are aligned and matched in the two models
        sd_hf=model_hf.state_dict()
        sd_hf_keys=sd_hf.keys()
        sd_hf_keys=[k for k in sd_hf_keys if not k.endswith('.attn.masked_bias')] # ignore the masked bias
        sd_hf_keys=[k for k in sd_hf_keys if not k.endswith('.attn.bias')] # ignore the bias
        transposed=['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'] # hardcoded weights which need to be transposed as the weights are from tensorflow
        # basically the openAI checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear?
        # so we tranpose these weights when we load them
        assert len(sd_hf_keys)==len(sd_keys), f"mismatched keys: {len(sd_hf_keys)} != {len(sd_keys)}"

        for k in sd_hf_keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1]==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


# ---------------------------------------------------------------
device='cpu'# 'cuda' if torch.cuda.is_available() else 'cpu'
num_return_sequences=5
max_length=30

# model=GPT.from_pretrained('gpt2') for configured model
# using non configured model


# prefix tokens
import tiktoken
# enc=tiktoken.get_encoding("gpt2")
# tokens=enc.encode("hello, i am a language model,")
# tokens=torch.tensor(tokens,dtype=torch.long) # (8, )
# tokens=tokens.unsqueeze(0).repeat(num_return_sequences,1) # (5, 8)
# x=tokens.to(device)


with open('input_tiny.txt','r') as f:
    text=f.read()
data=text[:1000] # gpt has a compression ratio of 3:1
# print(data)

enc=tiktoken.get_encoding('gpt2')


# --------------dataloader light ----------------

class DataLoadersLite:
    def __init__(self,B,T):
        self.B=B
        self.T=T


        # at init load tokens from disk and store them in memory

        with open('input_tiny.txt','r') as f:
            text=f.read()
        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)
        print(f'loaded {len(tokens)} tokens')
        print(f"1 epoch={len((self.tokens)//(B*T))} batches")

        # state
        self.current_position=0

    def next_batch(self):
        B,T=self.B, self.T
        buf=self.tokens[self.current_position:self.current_position+B*T+1]
        x=(buf[:-1]).view(B, T) # inputs
        y=(buf[1:]).view(B,T) # targets
        # advance the position in the tensor
        self.current_position+=B*T

        # if loading the next batch would be out of bounds, reset
        if self.current_position+(B*T+1)>len(self.tokens):
            self.current_position=0
        return x,y


train_loader=DataLoadersLite(B=4, T=32) # 5 batches of 32 tokens each




# token=enc.encode(data)
# # print(token[:24])
# B,T = 4,32

# buf=torch.tensor(token[:B*T+1])
# buf=buf.to(device)
# x=buf[:-1].view(B,T)
# y=buf[1:].view(B, T)
# # print(y)
# # print(x)

torch.manual_seed(1337)

model=GPT(GPTConfig())
print("didn't crash yay!")
print("new")
# model.eval()
model.to(device)

# calculate the loss
# get logits
# logits,loss=model(x,y)
# print(logits.shape)
# print(f"loss: {loss}")

optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3)
for i in range (50):
    x,y=train_loader.next_batch()
    x=x.to(device)
    y=y.to(device)
    optimizer.zero_grad()
    logits, loss=model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")


import sys;sys.exit(0)






# # generate! right now x is (B,T) where B=5, T=8
# # set the seed to 42
# torch.manual_seed(42)
# if device=="cuda":
#     torch.cuda.manual_seed(42) 

# while x.size(1)<max_length:
#     # forward the model to get the logits
#     with torch.no_grad():
#         # forward
#         logits=model(x) # (B,T, vocab_size)_
#         # focus only on the last time step- take the logits at the last position ?
#         logits=logits[:, -1, :] # becomes (B, vocab_size)
#         # apply softmax to get probabilities
#         probs=F.softmax(logits, dim=-1)
#         # do top-k sampling of 50(huggingface pipeline default)
#         # topk_probs here becomes (5,50), topk_indices is (5,50)
        
#         topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
#         # select a token from the top-k probabilities
#         idx=torch.multinomial(topk_probs, num_samples=1) # (B,1)
#         # gahter the corresponding indices
#         xcol=torch.gather(topk_indices, dim=-1, index=idx) # (B,1)
#         # append to the sequence
#         x=torch.cat((x, xcol), dim=1) # (B, T+1)

# # print the generated text
# for i in range(num_return_sequences):
#     tokens=x[i,:max_length].tolist()
#     decoded=enc.decode(tokens)
#     print(">",decoded)