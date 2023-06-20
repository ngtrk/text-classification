# https://github.com/ArdalanM/nlp-benchmarks/tree/master/src/transformer


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import copy

from torch.autograd import Variable
from torch.nn import CrossEntropyLoss



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
            
        return torch.matmul(p_attn, value), p_attn
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        n_batches = query.size(0)
        
        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)
        
        return self.linears[-1](x)
    
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=.1):
        super().__init__()
        self.w1 = nn.Conv1d(d_model, d_ff, 1)
        self.w2 = nn.Conv1d(d_ff, d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = output.transpose(1, 2)
        
        return output


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
    


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        
        div_term = torch.exp(div_term)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x
    

class LayerNorm(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super().__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.epsilon) + self.b2
    
    
class EncoderLayer(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout=.1):
        super().__init__()
        self.d = dropout
        self.self_attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.n1 = LayerNorm(d_model)
        self.n2 = LayerNorm(d_model)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(self.n1(x), self.n1(x), self.n1(x), mask)
        
        attn_output = F.dropout(attn_output, self.d)
        attn_output = x + attn_output
        
        encoder_out = self.feed_forward(self.n2(attn_output))
        encoder_out = F.dropout(encoder_out, self.d)
        encoder_out = attn_output + encoder_out
        
        return encoder_out
    

class Encoders(nn.Module):
    def __init__(self, vocab_size, h, d_model, d_ff, dropout=.1, n_layer=2):
        super().__init__()
        self.pe = PositionalEncoding(d_model, max_len=5000)
        
        self.emb = Embeddings(d_model, vocab_size)
        self.layers = clones(EncoderLayer(h, d_model, d_ff, dropout), n_layer)
        self.norm = LayerNorm(d_model)
        self.drop = nn.Dropout(p=dropout)
        
    def forward(self, src, src_mask):
        x = self.drop(self.pe(self.emb(src)))
        
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    

class DecoderLayer(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout=.1):
        super().__init__()
        
        self.d = dropout
        self.decoder_attn = MultiHeadedAttention(h, d_model, dropout)
        self.encoder_attn = MultiHeadedAttention(h, d_model, dropout)
        
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.n1 = LayerNorm(d_model)
        self.n2 = LayerNorm(d_model)
        self.n3 = LayerNorm(d_model)
        
    def forward(self, x, encoder_out, src_mask, tgt_mask):
        attn_out = self.decoder_attn(self.n1(x), self.n1(x), self.n1(x), tgt_mask)
        attn_out = F.dropout(attn_out, self.d)
        attn_out += x
        
        attn_in = self.encoder_attn(self.n2(attn_out), self.n2(attn_out), self.n2(attn_out), src_mask)
        attn_in = F.dropout(attn_in, self.d)
        attn_in = attn_in + attn_out
        
        decoder_out = self.ff(self.n3(attn_in))
        decoder_out = F.dropout(decoder_out, self.d)
        decoder_out += attn_in
        return decoder_out
    

class Decoders(nn.Module):
    def __init__(self, vocab_size, h, d_model, d_ff, dropout=.1, n_layer=2):
        super().__init__()
        self.pe = PositionalEncoding(d_model, max_len=5000)
        self.emb = Embeddings(d_model, vocab_size)
        self.decoder_layers = clones(DecoderLayer(h, d_model, d_ff, dropout), n_layer)
        
        self.norm = LayerNorm(d_model)
        self.drop = nn.Dropout(p=dropout)
        
    
    def forward(self, src, encoder_out, src_mask, tgt_mask):
        x = self.drop(self.pe(self.emb(src)))
        
        for layer in self.decoder_layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        
        return self.norm(x)
    
class SequenceClsHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


class ClsHead(nn.Module):
    def __init__(self, input_dim, n_labels):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, n_labels)
        
    def forward(self, x):
        out = x[:, 0]
        out = torch.tanh(self.linear1(out))
        out = self.linear2(out)
        return out


class TransformerLM(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, h, d_model, d_ff, dropout=.1, n_layer=1):
        super().__init__()
        self.encoder = Encoders(src_vocab_size, h, d_model, d_ff, dropout, n_layer)
        self.decoder = Decoders(trg_vocab_size, h, d_model, d_ff, dropout, n_layer)
        self.classification = SequenceClsHead(d_model, trg_vocab_size)
        
        for params in self.parameters():
            if params.dim() > 1:
                nn.init.xavier_uniform_(params)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(tgt, encoder_out, src_mask, tgt_mask)
        out = self.classification(decoder_out)
        return out
    


class TransformerCls(nn.Module):
    def __init__(self, n_classes, src_vocab_size, h, d_model, d_ff, dropout=.1, n_layer=1):
        super().__init__()
        self.encoder = Encoders(src_vocab_size, h, d_model, d_ff, dropout, n_layer)
        self.classification = ClsHead(d_model, n_classes)
        
        for params in self.parameters():
            if params.dim() > 1:
                nn.init.xavier_uniform_(params)
    
    def forward(self, src, src_mask):
        encoder_out = self.encoder(src, src_mask)
        out = self.classification(encoder_out)
        return out
    
    
class Batch:
    def __init__(self, src, trg=None, pad=0, dev=None):
        self.src = src.to(dev)
        self.src_mask = src.ne(pad).unsqueeze(-2).to(dev)
        self.dev = dev
        
        if trg is not None:
            self.trg = trg.to(dev)
            
            if trg.dim() > 1:
                self.trg = trg[:, :-1].to(dev)
                self.trg_y = trg[:, 1:].to(dev)
                
                self.trg_mask = self.make_std_mask(self.trg, pad)
                self.n_tokens = (self.trg_y != pad).data.sum().to(dev)
                
    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = tgt.ne(pad).unsqueeze(1)
        next_mask = subsequent_mask(tgt.size(-1)).to(tgt_mask.dev)
        tgt_mask = tgt_mask & next_mask
        return tgt_mask
    


def subsequent_mask(size):
    attn_shape = (size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    return mask.eq(0)


def data_gen_binary_classify(batch_size, n_batches, dev, vocab_size):
    seq_len = 100
    thresh = (seq_len * vocab_size) // 2
    for i in range(n_batches):
        tx = torch.from_numpy(np.random.randint(1, vocab_size, size=(batch_size, seq_len)))
        ty = (tx.sum(-1) > thresh).type_as(tx.data)
        yield Batch(tx, ty, 0, dev)
        
        

def data_gen_sequence(vocab_size, batch, n_batches, dev):
    for i in range(n_batches):
        data = torch.randint(1, vocab_size, size=(batch, 10))
        data[:, 0] = 1
        yield Batch(data, data, 0, dev)
        

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
    return ys




class NoamOpt:
    def __init__(self, model_size, factor, warmup, opt):
        self.opt = opt
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    
    def step(self):
        self._step += 1
        rate = self.rate()
        for params in self.opt.param_groups:
            params['lr'] = rate
            
        self._rate = rate
        self.opt.step()
        
    def rate(self, step=None):
        if step is None:
            step = self._step
            
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
    


class SmoothLoss(nn.Module):
    def __init__(self, n_classes=None, padding_id=0, smoothing=0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        
        self.padding_id = padding_id
        self.conf = 1.0 - smoothing
        self.smoothing = smoothing
        self.n_classes = n_classes
        
    
    def forward(self, ty_prob, ty_true):
        assert ty_true.size(0) == ty_prob.size(0)
        
        if self.smoothing > 0:
            n_classes = self.n_classes or ty_prob.size(1)
            
            ty_true_oh = torch.zeros_like(ty_prob).fill_(self.smoothing / (n_classes - 2))
            
            ty_true_oh.scatter_(1, ty_true.data.unsqueeze(1), self.conf)
            mask = torch.nonzero(ty_true == self.padding_id)
            if mask.numel() > 0:
                ty_true_oh.index_fill_(0, mask.squeeze(), 0.0)
            
            loss = self.criterion(ty_prob, Variable(ty_true_oh, requires_grad=False))
        
        else:
            loss = F.cross_entropy(ty_prob, ty_true, ignore_index=self.padding_id, reduction='sum')

        
        return loss
    

from types import SimpleNamespace

def test_train_transformer_lm():
    opt = SimpleNamespace(n_classes=2, d_model=32, N=2, d_ff=1024, h=8, dropout=.1, vocab_size=11, batch_size=128, n_batches=1000)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = data_gen_sequence(opt.vocab_size, opt.vocab_size, opt.n_batches, dev)
    
    model = TransformerLM(opt.vocab_size, opt.vocab_size, opt.h, opt.d_model, opt.d_ff, dropout=opt.dropout, n_layer=opt.N)
    
    optimizer = NoamOpt(model.encoder.emb.d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    criterion = SmoothLoss(opt.vocab_size, padding_id=0, smoothing=0)
    
    model.to(dev)
    criterion.to(dev)
    
    n_iter = 1000
    for i in range(n_iter):
        try:
            batch = next(gen)
        except:
            gen = data_gen_sequence(opt.vocab_size, opt.vocab_size, opt.n_batches, dev)
            batch = next(gen)
            
        ty_prob = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        ty_prob_reshaped = ty_prob.contiguous().view(-1, ty_prob.size(-1))
        
        ty_true = batch.trg_y
        ty_true_reshaped = ty_true.contiguous().view(-1)
        loss = criterion(ty_prob_reshaped, ty_true_reshaped)
        loss.backward()
        optimizer.step()
        optimizer.optimizer.zero_grad()
        acc = (ty_prob_reshaped.argmax(-1) == ty_true_reshaped).float().mean()
        
        if i > 0 and i % 50 == 1:
            print(i, acc.cpu().numpy(), loss.detach().cpu().numpy() / batch.n_tokens.cpu().numpy())
    model.eval()
    max_len = 10
    
    src = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).to(dev)
    src_mask = torch.ones(1, 1, src.size(1)).to(dev)
    
    memory = model.encoders(src, src_mask)
    
    ys = torch.ones(1, 1).type_as(src.data).to(dev)
    trg_mask = subsequent_mask(ys.size(1)).type_as(src.data).to(dev)
    
    for i in range(max_len-1):
        decoder_out = model.decoder(ys, memory, src_mask, trg_mask)
        prob = model.classification(decoder_out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
    print(f'Input: {src}')
    print(f'Prediction: {ys}')
    


def test_train_transformer_cls():
    opt = SimpleNamespace(n_classes=2, d_model=32, N=2, d_ff=1024, h=8, dropout=.1, vocab_size=1000, batch_size=128, n_batches=1000)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = data_gen_sequence(opt.vocab_size, opt.vocab_size, opt.n_batches, dev)
    
    model = TransformerCls(opt.n_classes, opt.vocab_size, opt.h, opt.d_model, opt.d_ff, opt.dropout, n_layer=opt.N)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=.0001)
    criterion = CrossEntropyLoss()
    
    model.to(dev)
    criterion.to(dev)
    
    n_iter = 1000
    gen = data_gen_binary_classify(opt.batch_size, opt.n_batches, dev, opt.vocab_size)
    
    for i in range(n_iter):
        try:
            batch = next(gen)
        except:
            gen = data_gen_binary_classify(opt.batch_size, opt.n_batches, dev, opt.vocab_size)
            batch = next(gen)
            
        out = model(batch.src, batch.src_mask)
        ty_true = batch.trg
        loss = criterion(out, ty_true)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc = (out.argmax(-1) == ty_true).sum().item() / ty_true.size(0)
        
        if i > 0 and i % 50 == 1:
            print(f'iter: {i}, acc: {acc}, loss: {loss.item()}')


if __name__ == '__main__':
    test_train_transformer_lm()
    test_train_transformer_cls()


    