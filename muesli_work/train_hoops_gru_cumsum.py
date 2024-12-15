import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from tensordict.utils import expand_as_right, unravel_key_list
import triton
import triton.language as tl
import math
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.utils import (
    _set_compilation_env,
)


def cumsum_fn(x, y):
    return x + y

def log_cumsum_exp_fn(x, y):
    return torch.logaddexp(x, y)
    

with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
    associative_scan_fn = torch.compile(associative_scan)

def heinsen_associative_scan_log(log_coeffs, log_values):
    log_coeffs = log_coeffs.to("cuda")
    log_values = log_values.to("cuda")
    a_star = associative_scan_fn(cumsum_fn, log_coeffs, dim=1)
    res_tensor =  log_values - a_star
    log_h0_plus_b_star = associative_scan_fn(log_cumsum_exp_fn, res_tensor, dim=1, combine_mode="generic")
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp().to("cpu")

def default(v, d):
    return v if v is not None else d

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

class MinGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.to_hidden_and_gate = nn.Linear(input_dim, hidden_dim * 2, bias = False)
        self.to_out = nn.Linear(hidden_dim, output_dim, bias = False) if output_dim else nn.Identity()

    def forward(self, input: torch.Tensor, is_init: torch.Tensor, h_0 = None):
        #we assume batch dim is always present
        missing_batch = False
        training = h_0 is None
        time_dim = 1
        if not training and len(input.shape) < 2:
            missing_batch = True
            input = input.unsqueeze(0)
            h_0 = h_0.unsqueeze(0)
            is_init = is_init.unsqueeze(0)
        
        if not training and len(input.shape) < 3:
            input = input.unsqueeze(time_dim) #add time dim
        
        batch = input.shape[0]
        seq_len = input.shape[1]


        if not training and len(is_init.shape) < 3:
            is_init = is_init.unsqueeze(time_dim) #add time dim
        
        if h_0 is None:
            h_0 = torch.zeros((batch, 1, self.hidden_dim), dtype=torch.float32, device=input.device)
            #prepend to is_init
        else:
            h_0 = torch.where(expand_as_right(is_init, h_0), 0, h_0)
            
        if not training and len(h_0.shape) < 3:
            h_0 = h_0.unsqueeze(time_dim) #add time dim

        hidden, gate = self.to_hidden_and_gate(input).chunk(2, dim = -1)
        

        #h_0 = torch.where(expand_as_right(is_init, h_0), 0, h_0) #this should probably be a part of the parallel implementation.
        if seq_len == 1:
            # handle sequential

            hidden = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(h_0, hidden, gate) if h_0 is not None else (hidden * gate)
        else:
            # parallel

            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if h_0 is not None:
                log_values = torch.cat((h_0.log(), log_values), dim = 1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            # Apply associative scan with reset logic
            #

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        h_n = out[:, -1:]

        out = self.to_out(out)

        if not training:
            out = out.squeeze(1)
            h_n = h_n.squeeze(1)

        if missing_batch:
            out = out.squeeze(0)
            h_n = h_n.squeeze(0)

        return out, h_n
    
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    # Tags are: DET - determiner; NN - noun; V - verb
    # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
# For each words-list (sentence) and tags-list in each tuple of training_data
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.gru = MinGRU(hidden_dim, hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        is_init = torch.zeros(1, len(sentence), 1, dtype=torch.bool, device=sentence.device)
        is_init[:, 0] = True
        embeds = embeds.view(1, len(sentence), -1)
        #sequential processing:
        #gru_outs = []
        #gru_hidden = torch.zeros((1, self.hidden_dim))
        
        # for is_init_sample, embed_sample in zip(is_init.unbind(1), embeds.unbind(1)):
        #     gru_out, gru_hidden = self.gru(embed_sample, is_init_sample, gru_hidden)
        #     gru_outs.append(gru_out)
        # gru_out = torch.stack(gru_outs, dim=1) #stack along time dim
        #parallel processing
        gru_out, gru_hidden = self.gru(embeds, is_init)
        tag_space = self.hidden2tag(gru_out.view(1, len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=-1)
        return tag_scores
    
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
    print("Initial actions: ", torch.argmax(tag_scores, dim=-1))
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        tag_scores = model(sentence_in).reshape(-1, len(tag_to_ix))
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)
    print("Post training: ", torch.argmax(tag_scores, dim=-1))