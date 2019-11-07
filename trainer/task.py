import datetime
import os
import subprocess
import sys
import argparse
import collections
import torch
import numpy as np

from . import model
from . import utils


def get_args():
    """Argument parser

    Returns:
        Dictionary of parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-file',
        type=str,
        required=True,
        help='Training file local or GCS')
    parser.add_argument(
        '--bucket-name',
        type=str,
        default='credit-card-fraud-detection3',
        help='The bucket name in which the finished model is saved')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help="the number of records to read for each training step, default=128")
    parser.add_argument(
        '--learning-rate',
        default=0.01,
        type=float,
        help='learning rate for gradient descent, default=0.01')
    parser.add_argument(
        '--first-compression',
        default=18,
        type=int,
        help='The number of input nodes in the first hidden layer, default=18')
    parser.add_argument(
        '--second-compression',
        default=8,
        type=int,
        help='The number of input nodes in the second hidden layer, default=8')

    return parser.parse_args()

def train_and_evaluate(args):
    """Helper function: Trains, evaluates, and saves Autoencoder to Google Cloud.

    Args:
        args: a dictionary of arguments - see get_args() for details

    Returns:
        None
    """

    # load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = utils.preprocess(args.train_file)

    # create Pytorch Dataloaders
    xdl = utils.create_dataloader(X_train.values, args.batch_size)
    vdl = utils.create_dataloader(X_val.values, args.batch_size)
    tdl = utils.create_dataloader(X_test.values, args.batch_size)

    # create Autoencoder
    num_features = len(X_train.columns)
    autoencoder = model.compile_model(num_features, args.first_compression, args.second_compression)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters())

    # train and save Autoencoder
    model_filename = 'autoencoder.pt'
    train_model_and_save(args.num_epochs, autoencoder, loss, optimizer, xdl, vdl, model_filename)

    # Upload the saved model to the bucket.
    model_folder = datetime.datetime.now().strftime('imdb_%Y%m%d_%H%M%S')
    gcs_model_path = os.path.join('gs://', hparams.bucket_name, 'results', model_folder, 'models')

    subprocess.check_call(['gsutil', 'cp -r', model_filename, gcs_model_path], stderr=sys.stdout)
    os.remove(model_filename)


def train_model_and_save(num_epochs, autoencoder, loss, optimizer, train_dataloader,
                         val_dataloader, model_filename):
    try: c = model_loss.epoch[-1]
    except: c = 0

    model_hist = collections.namedtuple('Model', 'epoch loss val_loss')
    model_loss = model_hist(epoch=[], loss=[], val_loss=[])

    for epoch in range(num_epochs):
        losses = []
        dl = iter(train_dataloader)

        for t in range(len(dl)):
            # Forward Pass
            xt = next(dl)
            y_pred = autoencoder(torch.autograd.Variable(xt))

            l = loss(y_pred, torch.autograd.Variable(xt))
            losses.append(l)
            optimizer.zero_grad()

            l.backward()

            optimizer.step()

        val_dl = iter(val_dataloader)
        val_scores = [score(next(val_dl), autoencoder, loss) for i in range(len(val_dl))]

        model_loss.epoch.append(c + epoch)
        model_loss.loss.append(l.item())
        model_loss.val_loss.append(np.mean(val_scores))

        print(f'Epoch: {epoch}   Loss: {l.item():.4f}   Val_Loss: {np.mean(val_scores):.4f}')

        # save trained autoencoder and history locally
        torch.save({'epoch': len(model_loss.epoch),
                    'model_state_dict': autoencoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss' : loss}, model_filename)

def score(x, autoencoder, loss):
    y_pred = autoencoder(torch.autograd.Variable(x))
    x1 = torch.autograd.Variable(x)
    return loss(y_pred,x1).item()

if __name__ == '__main__':

    args = get_args()
    train_and_evaluate(args)