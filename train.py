from net import TransformerCls
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset, DatasetDict

import params
import tokenizer as tk
import os



def train(epoch, net, dataset, dev, msg='val/test', optimize=False, optimizer=None, criterion=None):
    net.train() if optimize else net.eval()
    
    epoch_loss = 0
    epoch_acc = 0
    dic_metrics ={'loss': 0, 'acc': 0, 'lr': 0}
    
    with tqdm(total=len(dataset), desc=f'Epoch {epoch} - {msg}') as pbar:
        for i, (tx, mask, ty) in enumerate(dataset):
            data = (tx, mask, ty)
            data = [x.to(dev) for x in data]
            
            if optimize:
                optimizer.zero_grad()
                
            
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



def get_tokenizer():
    param = params.get_args()
    
    
    if ''.join([param.token_filename, '.json']) not in os.listdir(param.current_folder):
        tk.main()
        
        
    token = PreTrainedTokenizerFast(
        model_max_length=param.vocab_size,
        tokenizer_file=f"{param.token_filename}.json",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
        padding_side="left")
    
    return token



def process_ds(examples):
    tok = get_tokenizer()
    return tok(examples['text'], truncation=True)



def load_data(pr):
    print('Loading dataset...')
    dataset = load_dataset('ag_news')
    n_classes = len(set(dataset['train']['label']))
    train_ds = dataset['train'].train_test_split(test_size=.2, seed=pr.seed)
    ds_splits = DatasetDict({
        'train': train_ds['train'],
        'val': train_ds['test'],
        'test': dataset['test']
    })
    return ds_splits, n_classes



if __name__ == '__main__':
    pr = params.get_args()
    
    data, n_classes = load_data(pr)
    
    print('Mapping data...')
    tokenized_train = data['train'].map(process_ds, batched=True)
    tokenized_val = data['val'].map(process_ds, batched=True)


    os.makedirs(pr.model_folder, exist_ok=True)
    os.makedirs(pr.data_folder, exist_ok=True)


    train_loader = DataLoader(tokenized_train, batch_size=pr.batch_size, shuffle=True, collate_fn=collate_fn,  pin_memory=True)
    val_loader = DataLoader(tokenized_val, batch_size=pr.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    print("Creating model...")
    net = TransformerCls(n_classes=n_classes,
                            src_vocab_size=pr.vocab_size,
                            h=pr.n_heads,
                            d_model=pr.attention_dim,
                            d_ff=pr.ff_hidden_size,
                            dropout=pr.dropout,
                            n_layer=pr.n_layers)


    net.to(dev)
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=pr.lr, weight_decay=pr.weight_decay)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(1, pr.epochs + 1):
        train(epoch, net, train_loader, dev, msg='TRAINING...', optimize=True, optimizer=optimizer, criterion=criterion)
        train(epoch, net, val_loader, dev, msg='VALIDATING...', criterion=criterion)
        
        if (epoch % pr.snapshot_interval == 0) and (epoch > 0):
            path = f'{pr.model_folder}/model_epoch_{epoch}.pth'
            torch.save(net.state_dict(), path)
            

    if pr.epochs > 0:
        path = f'{pr.model_folder}/model_epoch_{pr.epochs}.pth'
        torch.save(net.state_dict(), path)


