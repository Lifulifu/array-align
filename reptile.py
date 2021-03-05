import cv2
import os
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from .model import Resnet
from .draw import *
from .dataset import *
from .util import *


def train_task(model, xtr, xte, ytr, yte, lr=1e-3, epochs=10, device='cuda:0'):
    new_model = Resnet().to(device)
    new_model.load_state_dict(model.state_dict())  # copy model
    xtr, xte, ytr, yte = torch.tensor(xtr).float().to(device), torch.tensor(xte).float().to(
        device), torch.tensor(ytr).float().to(device), torch.tensor(yte).float().to(device)
    optimizer = torch.optim.SGD(new_model.parameters(), lr=lr)
    loss_func = nn.SmoothL1Loss(reduction='mean')

    # train
    bar = tqdm(range(epochs))
    for i in bar:
        ypred = new_model(xtr)
        loss = loss_func(ypred, ytr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bar.set_description(f'task loss: {loss.item():.3f}')

    # test
    with torch.no_grad():
        ypred = new_model(xte)
        loss = loss_func(ypred, yte)

    return new_model, loss.item(), ypred.cpu().numpy()  # for xte


def meta_train(model, data_dir, meta_start_epoch=1, meta_epochs=1000, meta_tr_size=800, meta_lr=1e-3,
               task_epochs=10, task_tr_size=5, task_te_size=32, task_te_epochs=100, task_lr=1e-3,
               imgs_per_task=None, save_interval=5, output_dir='outputs/', device='cuda:0'):

    model.to(device)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'preds'), exist_ok=True)
    tasks = os.listdir(data_dir)
    tr_tasks, te_tasks = tasks[:meta_tr_size], tasks[meta_tr_size:]
    write_file(tr_tasks, os.path.join(output_dir, 'tr.txt'))
    write_file(te_tasks, os.path.join(output_dir, 'te.txt'))

    optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    # log file header
    if meta_start_epoch <= 1:
        write_file('epoch,tr_loss,va_loss',
                   os.path.join(output_dir, 'training_log.txt'), mode='w')
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs/'))

    for epoch in range(meta_start_epoch, meta_start_epoch+meta_epochs):
        # --- Train ---
        print(f'epoch {epoch}')
        model.train()
        # Sample task
        task = os.path.join(data_dir, np.random.choice(tr_tasks))
        x, y = np.load(os.path.join(task, 'x.npy')), np.load(os.path.join(task, 'y.npy'))
        xtr, xte, ytr, yte = x[
            :task_tr_size], x[task_tr_size:], y[:task_tr_size], y[task_tr_size:]
        new_model, meta_tr_loss, ypred = train_task(
            model, xtr, xte, ytr, yte, epochs=task_epochs, lr=task_lr, device=device)

        # Inject updates into each .grad
        for p, new_p in zip(model.parameters(), new_model.parameters()):
            if p.grad is None:  # grad is None at the first iteration
                p.grad = Variable(torch.zeros(p.size())).to(device)
            p.grad.data.add_(p.data - new_p.data)

        # Update meta-model
        optimizer.step()
        optimizer.zero_grad()

        if epoch % save_interval == 0 and xtr is not None:
            path = os.path.join(output_dir, f'preds/e{epoch}/tr/')
            os.makedirs(path, exist_ok=True)
            write_file(task, os.path.join(path, 'task.txt'))
            write_corners_xybs([xte], [yte], [ypred],
                               output_dir=path, n_samples=10)

        # --- Validate ---
        model.eval()
        task = os.path.join(data_dir, np.random.choice(te_tasks))
        x, y = np.load(os.path.join(task, 'x.npy')), np.load(os.path.join(task, 'y.npy'))
        xtr, xte, ytr, yte = x[
            :task_tr_size], x[task_tr_size:], y[:task_tr_size], y[task_tr_size:]

        _, meta_va_loss, ypred = train_task(
            model, xtr, xte, ytr, yte, epochs=task_te_epochs)

        writer.add_scalars("tr_loss", {
            'tr': meta_tr_loss,
            'va': meta_va_loss}, epoch)
        print(f'meta tr loss: {meta_tr_loss:.3f}, meta va loss: {meta_va_loss:.3f}')

        if epoch % save_interval == 0 and xtr is not None:
            path = os.path.join(output_dir, f'preds/e{epoch}/va/')
            os.makedirs(path, exist_ok=True)
            write_corners_xybs([xte], [yte], [ypred],
                               output_dir=path, n_samples=10)
            write_file(task, os.path.join(path, f'task.txt'))
            torch.save(model, os.path.join(output_dir, f'models/{epoch}.pt'))

        write_file(f'{epoch},{meta_tr_loss},{meta_va_loss}',
            os.path.join(output_dir, 'training_log.txt'), mode='a')
    writer.flush()
