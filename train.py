from net import TransformerCls
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from datasets import DatasetDict
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


print('Loading dataset...')
dataset = load_dataset('ag_news')



def get_tokenizer():
    if ''.join([pr.token_filename, '.json']) not in os.listdir(pr.current_folder):
        tk.main()
        
    token = PreTrainedTokenizerFast(
        model_max_length=pr.vocab_size,
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

train_ds = dataset['train'].train_test_split(test_size=.2, seed=pr.seed)
ds_splits = DatasetDict({
    'train': train_ds['train'],
    'val': train_ds['test'],
    'test': dataset['test']
})



def process_ds(examples):
    return tok(examples['text'], truncation=True)


print('Mapping data...')
tokenized_ds = ds_splits.map(process_ds, batched=True)


os.makedirs(pr.model_folder, exist_ok=True)
os.makedirs(pr.data_folder, exist_ok=True)

variables = {
    'train': {'var': None, 'path': f'{pr.data_folder}/train.lmdb'},
    'test': {'var': None, 'path': f'{pr.data_folder}/test.lmdb'},
    'params': {'var': None, 'path': f'{pr.data_folder}/params.pkl'}
    
}


tr_loader = DataLoader(tokenized_ds['train'], batch_size=pr.batch_size, shuffle=True, collate_fn=collate_fn,  pin_memory=True)
te_loader = DataLoader(tokenized_ds['val'], batch_size=pr.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print("Creating model...")
net = TransformerCls(n_classes=pr.n_classes,
                        src_vocab_size=pr.vocab_size,
                        h=pr.n_heads,
                        d_model=pr.attention_dim,
                        d_ff=pr.ff_hidden_size,
                        dropout=pr.dropout,
                        n_layer=pr.n_layers)


net.to(dev)
optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=pr.lr, weight_decay=pr.weigth_decay)
criterion = nn.CrossEntropyLoss()


for epoch in range(1, pr.epochs + 1):
    train(epoch, net, tr_loader, dev, msg='training...', optimize=True, optimizer=optimizer, criterion=criterion)
    train(epoch, net, te_loader, dev, msg='testing...', criterion=criterion)
    
    if (epoch % pr.snapshot_interval == 0) and (epoch > 0):
        path = f'{pr.model_folder}/model_epoch_{epoch}.pth'
        save(net, variables['params']['var'], path=path)
        

if pr.epochs > 0:
    path = f'{pr.model_folder}/model_epoch_{pr.epochs}'
    save(net, variables['params']['var'], path=path)


