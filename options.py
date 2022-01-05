import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector_size',type=int,default=300)
    parser.add_argument('--max_no_tokens',type=int,default=20)
    parser.add_argument('--training_size',type=int,default=10000)
    parser.add_argument('--embedding_dim',type=int,default=300)
    parser.add_argument('--hidden_dim',type=int,default=200)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--lr_rate',type=float,default=0.001)
    parser.add_argument('--batch_size',type=int,default=64)

    opt = parser.parse_args()
    return opt