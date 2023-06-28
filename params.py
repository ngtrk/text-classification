import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", type=str, default="dataset/")
    parser.add_argument("--model_folder", type=str, default="model/")


    parser.add_argument("--attention_dim", type=int, default=16)
    parser.add_argument("--n_heads", type=int, default=4, help="")
    parser.add_argument("--n_layers", type=int, default=4, help="")
    
    parser.add_argument("--dropout", type=float, default=0.2, help="")
    parser.add_argument("--ff_hidden_size", type=int, default=16, help="point-wise feed forward NN")


    parser.add_argument("--lr", type=float, default=0.00015)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--vocab_size", type=int, default=30000)
 
    parser.add_argument("--batch_size", type=int, default=128, help="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--snapshot_interval", type=int, default=10, help="Save model every n epoch")

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--token_filename", type=str, default='tokenizer-wiki')
    parser.add_argument("--current_folder", type=str, default='.')
    args = parser.parse_args()
    return args

