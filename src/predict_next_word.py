import pickle
import re
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import random

pickle_in = open("plots_text.pickle","rb")
movie_plots = pickle.load(pickle_in)
movie_plots = [re.sub("[^a-z ' ]", " ", i) for i in movie_plots] 
def get_fixed_sequence(text, seq_len = 5):
  sequences = []
  words = text.split()
  if len(words) > seq_len:
    for i in range(seq_len, len(words)):
      seq_list = words[i-seq_len: i]
      sequences.append(" ".join(seq_list))
  else:
    sequences = words
  return sequences
seqs = [get_fixed_sequence(plot) for plot in movie_plots]
seqs = sum(seqs, [])
x = []
y = []
for seq in seqs:
  words = seq.split()
  x.append(" ".join(words[:-1]))
  y.append(" ".join(words[1:]))
# create integer-to-token mapping
int2token = {}
cnt = 0

for w in set(" ".join(movie_plots).split()):
  int2token[cnt] = w
  cnt+= 1

# create token-to-integer mapping
token2int = {t: i for i, t in int2token.items()}
    
def predict(net, tkn, h=None):
         
  # tensor inputs
  x = np.array([[token2int[tkn]]])
  inputs = torch.from_numpy(x)
  
  # push to GPU
  #inputs = inputs.cuda()

  # detach hidden state from history
  h = tuple([each.data for each in h])

  # get the output of the model
  out, h = net(inputs, h)

  # get the token probabilities
  p = F.softmax(out, dim=1).data

  p = p.cpu()

  p = p.numpy()
  p = p.reshape(p.shape[1],)

  # get indices of top 3 values
  top_n_idx = p.argsort()[-3:][::-1]
  # print(top_n_idx)
  # randomly select one of the three indices
  # sampled_token_index = top_n_idx[random.sample([0,1,2],1)[0]]
  # return the encoded value of the predicted char and the hidden state
  list1 = []
  for i in top_n_idx:
    list1.append(int2token[i])
  return list1, h

def sample(net, prime='it is'):
        
    # push to GPU
    #net.cuda()
    
    net.eval()

    # batch size is 1
    h = net.init_hidden(1)

    toks = prime.split()
    # predict next token
    for t in prime.split():
     
        token, h = predict(net, t, h)

      # print(token)
    
    # toks.append(token)

    # predict subsequent tokens
    # for i in range(size-1):
    #     token, h = predict(net, toks[-1], h)
    #     toks.append(token)
    result=''
    for i in token:
        i=i+' '
        result+=i
    return result 