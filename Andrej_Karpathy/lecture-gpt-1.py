# hi- start date
import logging
# import time
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.debug('Start of program')
import torch 
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)



with open('input.txt','r',encoding='utf-8') as f: # with statement replaces a try-catch block with a concise shorthand
    # ensures closing resources right after processing them.
    text=f.read()
print(f"length of the dataset characters {len(text)}")

# pick out the unique characters
chars=sorted(list(set(text)))
vocab_size=len(chars)
print("".join(chars))


# creating mapping from char to integers
stoi={ch:i for i,ch in enumerate(chars)} # enumerate gives index(not key) value
itos={i:ch for i,ch in enumerate(chars)}
# for index_int,val in enumerate (stoi):
#     print(f"key: {val} value: {stoi[val]} ")


# lambda takes lambda var: -----> var as input to the function 
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers 
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))



import torch 
data= torch.tensor(encode(text), dtype=torch.long)
print(data.shape)
print(data[:10])
# print(decode(data[0][:10].item


# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def tensor_decoder(ten):
    dec=decode([ten[x].item() for x in range(ten.shape[0])])
    return dec
    

# dec=[data[x].item() for x in range(10)]


block_size = 8 # also called the context size
train_data[:block_size+1] # +1 because if lets say abcd is the len is 3, d context is abc 


# context can be taken anywhere between 1 to 8 is the meaning

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")


    # print(len(data))
batch_size=4
# print(torch.randint(len(data) - block_size, (batch_size,)))
ix = torch.randint(len(data) - block_size, (batch_size,))
print([data[i:i+block_size] for i in ix])



torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # catches starting point randomly from any line
    x = torch.stack([data[i:i+block_size] for i in ix])   ## they become a row in 4x8 tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")




class BigramLanguageModel(nn.Module):
    def __init__(self,vocam_size):
        super().__init__() # inherit everything from nn class
    
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size) # token embeding table of size vocab_size*vocab_size
    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) every idx pluck correspoinding row from embedding table
        # how will the embedding table look like # B batch=4, T time is 8 and C channel is vocab size(65 in this case)
        # pluck out the rows, arrange them in b,t,c
        # logits are the score for the next in the sequence

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # see the documentation
            targets = targets.view(B*T) # alternatively can do -1
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens): # take in context and generate +1+2 upto max tokens
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)


batch_size=32
for steps in range(1000):
    xb,yb= get_batch('train')
    
    logits, loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print(loss.item())