import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from utils import *

from torch.utils.tensorboard import SummaryWriter


def main():
    # read command line arguments
    args = parse_args()
    model_name = args.model
    dataset_name = args.dataset
    dataset_dir = args.dir
    dataset_split = args.split_scheme
    dataset_transform = args.transform
    model_layers = args.layers
    model_num_filters = args.num_filters
    model_loss = args.loss
    model_metric = args.metric
    train_epochs = args.epochs
    train_batch_size = args.batch_size
    train_lr = args.lr
    train_load_model = args.load_model

    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # setup model according to command line arguments
    model = initialize_model(in_channel=3, out_channel=3, type=model_name, layers=model_layers, num_filters=model_num_filters)
    if train_load_model:
        checkpoint_load(model, train_load_model)
    
    model = model.to(device)
    print(model)

    # setup dataset
    dataset = initialize_dataset(dataset=dataset_name, split_scheme=dataset_split, dir=dataset_dir, transform=dataset_transform, split_ratio=[1, 1], pre_transform=None)
    # dataset.process() # test dataset processing parallel
    print(dataset)

    # split dataset into train, val and test sets
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9):]

    # setup dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    # setup loss function
    loss_fn = initialize_loss(loss_type=model_loss)

    # setup metric function
    metric_fn = initialize_metric(metric_type=model_metric)

    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)

    # setup tensorboard
    logdir = '../train/logs/{}'.format(get_cur_time())
    savedir = '../train/checkpoints/{}'.format(get_cur_time())
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    writer_logs = SummaryWriter(logdir)
    
    # training loop
    for epoch in range(train_epochs):
        model.train()
        avg_loss = 0
        avg_accuracy = 0
        for batch in train_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(batch.y, pred)
            avg_loss += loss.item()
            avg_accuracy += metric_fn(batch.y, pred)
            loss.backward()
            optimizer.step()
            # print('Epoch: {:03d}, Batch: {:03d}, Loss: {:.4f}, Accuracy metric: {:4f}'.format(epoch, batch.batch[-1], loss.item(), metric_fn(batch.y, pred)))

        avg_loss /= len(train_dataloader)
        avg_accuracy /= len(train_dataloader)
        print('Epoch: {:03d}, Loss: {:.4f}, Accuracy metric: {:4f}'.format(epoch, avg_loss, avg_accuracy))

        writer_logs.add_scalar('Loss/train', avg_loss, epoch)
        writer_logs.add_scalar('Max_div/train', avg_accuracy, epoch)
        # evaluate model with validation set every 25 epochs and save checkpoint
        if epoch % 25 == 0:
            evaluate_model(model, val_dataloader, writer_logs, epoch, loss_fn, metric_fn, device, mode='val')
            checkpoint_save(model, savedir, epoch)

    # evaluate model with test set
    evaluate_model(model, test_dataloader, writer_logs, epoch, loss_fn, metric_fn, device, mode='test')

    # close tensorboard and save final model
    writer_logs.close()
    checkpoint_save(model, savedir, epoch)

if __name__ == '__main__':
    main()