import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from tqdm import tqdm
import json

from torch.utils.tensorboard import SummaryWriter

from .model import Resnet
from .draw import *
from .dataset import *
from .util import *


def train_task(model, xtr, xte, ytr, yte, lr=1e-3, epochs=10, device='cuda:0'):
    new_model = Resnet()
    new_model.load_state_dict(model.state_dict()).to(device)  # copy model
    xtr, xte, ytr, yte = xtr.float().to(device), xte.float().to(
        device), ytr.float().to(device), yte.float().to(device)
    optimizer = torch.optim.SGD(new_model.parameters(), lr=lr)
    loss_func = nn.SmoothL1Loss(reduction='mean')

    # train
    for i in range(epochs):
        ypred = new_model(xtr)
        loss = loss_func(ypred, ytr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # test
    with torch.no_grad():
        ypred = new_model(xte)
        loss = loss_func(ypred, yte)

    return new_model, loss.item(), ypred.cpu().numpy()  # for xte


def meta_train(model, meta_dataset, meta_start_epoch=1, meta_epochs=1000, meta_lr=1e-3,
               task_epochs=10, task_tr_size=5, task_te_size=32, task_lr=1e-3,
               imgs_per_task=None, save_interval=5, output_dir='outputs/', device='cuda:0'):

    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'preds'), exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    # log file header
    if meta_start_epoch <= 1:
        write_file('epoch,tr_loss,va_loss',
                   os.path.join(output_dir, 'training_log.txt'), mode='w')
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs/'))

    for epoch in tqdm(
            range(meta_start_epoch, meta_start_epoch+meta_epochs), ncols=100):
        # --- Train ---
        # Sample task
        xtr, xte, ytr, yte, task_info = meta_dataset.get_task(task_tr_size, task_te_size, imgs_per_task)
        new_model, meta_tr_loss, ypred = train_task(
            model, xtr, xte, ytr, yte, epochs=task_epochs, lr=task_lr, device=device)

        # Inject updates into each .grad
        for p, new_p in zip(model.parameters(), new_model.parameters()):
            p.grad.data.add_(p.data - new_p.data)

        # Update meta-model
        optimizer.step()
        optimizer.zero_grad()

        if epoch % save_interval == 0 and xtr is not None:
            path = os.path.join(output_dir, f'preds/e{epoch}/tr/')
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, 'task.json'), 'w') as f:
                f.write(json.dumps(task_info))
            write_corners_xybs([xte], [yte], [ypred],
                               output_dir=path, n_samples=10)

        # --- Validate ---
        xtr, xte, ytr, yte, _ = meta_dataset.get_task(task_tr_size, task_te_size)
        _, meta_va_loss, ypred = train_task(
            model, xtr, xte, ytr, yte, epochs=task_epochs)

        writer.add_scalars("tr_loss", {
            'tr': meta_tr_loss,
            'va': meta_va_loss}, epoch)

        if epoch % save_interval == 0 and xtr is not None:
            path = os.path.join(output_dir, f'preds/e{epoch}/va/')
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, 'task.json'), 'w') as f:
                f.write(json.dumps(task_info))
            write_corners_xybs([xte], [yte], [ypred],
                               output_dir=path, n_samples=10)
            torch.save(model, os.path.join(output_dir, f'models/{epoch}.pt'))

        write_file(f'{epoch},{meta_tr_loss},{meta_va_loss}',
            os.path.join(output_dir, 'training_log.txt'), mode='a')
    writer.flush()
