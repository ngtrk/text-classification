import params
import torch
from net import TransformerCls
import train
from torch.utils.data import DataLoader


pr = params.get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, n_classes = train.load_data(pr)

n = len(data['test'])

tokenized_test = data['test'].map(train.process_ds, batched=True)

test_loader = DataLoader(tokenized_test, batch_size=pr.batch_size, shuffle=False, collate_fn=train.collate_fn, pin_memory=True)

model = TransformerCls(n_classes=n_classes,
                        src_vocab_size=pr.vocab_size,
                        h=pr.n_heads,
                        d_model=pr.attention_dim,
                        d_ff=pr.ff_hidden_size,
                        dropout=pr.dropout,
                        n_layer=pr.n_layers)


path = f'{pr.model_folder}/model_epoch_{pr.epochs}.pth'

model.load_state_dict(torch.load(path, map_location=torch.device('cpu') ))
model.to(device)
model.eval()
acc = 0

print('\n Inferencing on test data...')

for (tx, mask, ty) in test_loader:
    inp = (tx, mask, ty)
    inp = [x.to(device) for x in inp]
    out = model(inp[0], inp[1])
    _, preds = torch.max(out, 1)
    preds = preds.cpu().numpy()

    for j in range(len(out)):
        if ty[j] == preds[j]:
            acc += 1

print(f'Accuracy on test dataset: {(acc / n) * 100: .4f}%')


