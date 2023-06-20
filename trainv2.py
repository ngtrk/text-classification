from net import TransformerCls, NoamOpt
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import params as pr
import tokenizer as tk
import os



def train(epoch, net, dataset, dev, msg='val/test', optimize=False, optimizer=None, criterion=None):
    net.train() if optimize else net.eval()
    
    epoch_loss = 0
    epoch_acc = 0
    dic_metrics ={'loss': 0, 'acc': 0, 'lr': 0}
    # n_classes = len(list(net.parameters())[-1])
    
    with tqdm(total=len(dataset), desc=f'Epoch {epoch} - {msg}') as pbar:
        for i, (tx, mask, ty) in enumerate(dataset):
            data = (tx, mask, ty)
            
            data = [x.to(dev) for x in data]
            
            if optimize:
                optimizer.zero_grad()
                
            net.to(dev)
            
            out = net(data[0], data[1])
            loss = criterion(out, data[2])
            
            epoch_loss += loss.item()
            epoch_acc += (data[-1] == out.argmax(-1)).sum().item() / len(out)
            
            dic_metrics['loss'] = epoch_loss / (i + 1)
            dic_metrics['acc'] = epoch_acc / (i + 1)
            
            if optimize:
                loss.backward()
                dic_metrics['lr'] = optimizer.state_dict()['param_groups'][0]['lr']
                optimizer.step()
            
            pbar.update(1)
            pbar.set_postfix(dic_metrics)


def save(net, txt_dict, path):
    dict_m = net.state_dict()
    dict_m['txt_dict'] = txt_dict
    torch.save(dict_m, path)



def collate_fn(l):
    temp = []
    for x in l:
        temp.append((x['input_ids'], x['label']))
    
    sequence, labels = zip(*temp)
    local_maxlen = max(map(len, sequence))
    
    xs = [np.pad(x, (0, local_maxlen - len(x))) for x in sequence]
    xs = np.array(xs, dtype='int32')

    tx = torch.LongTensor(xs)
    tx_mask = tx.ne(0).unsqueeze(-2)
    ty = torch.LongTensor(labels)
    return tx, tx_mask, ty



epochs = 50
weigth_decay = 0
lr = 0.0001
gamma = 0.9
snapshot_interval = 10 # save model each 10 epoch
model_folder = '/model'
data_folder = '/dataset'
ff_hidden_size = 16
n_heads = 4
n_layers = 4
attention_dim = 16
n_classes = 4
dropout = 0.2
n_tokens = 30000
batch_size = 128
seed = 1337



print('Loading dataset...')
dataset = load_dataset('ag_news')



def get_tokenizer():
    if ''.join([pr.token_filename, '.json']) not in os.listdir(pr.current_folder):
        tk.main()
        
    token = PreTrainedTokenizerFast(
        model_max_length=n_tokens,
        tokenizer_file="tokenizer-wiki.json",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
        padding_side="left")
    return token


print('Tokenizer data...')
tok = get_tokenizer()



def process_ds(examples):
    return tok(examples['text'], truncation=True)


print('Mapping data...')
tokenized_ds = dataset.map(process_ds, batched=True)

os.makedirs(model_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)

variables = {
    'train': {'var': None, 'path': f'{data_folder}/train.lmdb'},
    'test': {'var': None, 'path': f'{data_folder}/test.lmdb'},
    'params': {'var': None, 'path': f'{data_folder}/params.pkl'}
    
}

train_ds = tokenized_ds['train']
val_ds = tokenized_ds['test']

tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,  pin_memory=True)
te_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print("Creating model...")
net = TransformerCls(n_classes=n_classes,
                        src_vocab_size=n_tokens,
                        h=n_heads,
                        d_model=attention_dim,
                        d_ff=ff_hidden_size,
                        dropout=dropout,
                        n_layer=n_layers)


net.to(dev)
optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=lr, weight_decay=weigth_decay)
criterion = nn.CrossEntropyLoss()


for epoch in range(1, epochs + 1):
    train(epoch, net, tr_loader, dev, msg='training...', optimize=True, optimizer=optimizer, criterion=criterion)
    train(epoch, net, te_loader, dev, msg='testing...', criterion=criterion)
    
    if (epoch % snapshot_interval == 0) and (epoch > 0):
        path = f'{model_folder}/model_epoch_{epoch}'
        save(net, variables['params']['var'], path=path)
        

if epochs > 0:
    path = f'{model_folder}/model_epoch_{epochs}'
    save(net, variables['params']['var'], path=path)
    
    


