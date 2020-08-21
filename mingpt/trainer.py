"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import logging
import math

import numpy as np
import torch
import torch.optim as optim
from labml import tracker, monit, logger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from labml.configs import BaseConfigs, option
from mingpt.model import GPTConfig


class TrainerConfig(BaseConfigs):
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    model: GPTConfig
    optimizer: optim.AdamW


@option(TrainerConfig.optimizer)
def adamw_optimizer(config: TrainerConfig):
    model = config.model.model
    no_decay = ["bias", "LayerNorm.weight"]
    params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optim_groups = [
        {"params": params_decay, "weight_decay": config.weight_decay},
        {"params": params_nodecay, "weight_decay": 0.0},
    ]
    return optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)


@option(TrainerConfig.model)
def gpt_model(config: TrainerConfig):
    return GPTConfig()


class Trainer:

    def __init__(self, train_dataset, test_dataset, config: TrainerConfig):
        self.model = config.model.model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        if self.config.ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            print("saving %s", self.config.ckpt_path)
            torch.save(ckpt_model.state_dict(), self.config.ckpt_path)

    def run_epoch(self, split, epoch):
        model, config, optimizer = self.model, self.config, self.config.optimizer

        is_train = split == 'train'
        model.train(is_train)
        data = self.train_dataset if is_train else self.test_dataset
        loader = DataLoader(data, batch_size=config.batch_size, num_workers=config.num_workers)

        losses = []
        for it, (x, y) in monit.enum(split, loader):
            # place data on the correct device
            x = x.to(self.device)
            y = y.to(self.device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                logits, loss = model(x, y)
                loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())

            if is_train:

                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()

                # decay the learning rate based on our progress
                if config.lr_decay:
                    self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                    if self.tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.tokens - config.warmup_tokens) / float(
                            max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                # report progress
                tracker.save({f'loss.{split}': loss, f'lr': lr})

        # if not is_train:
        #     logger.info("test loss: %f", np.mean(losses))

    def train(self):
        self.tokens = 0  # counter used for learning rate decay
        tracker.set_scalar('*', True)
        for epoch in monit.loop(self.config.max_epochs):
            self.run_epoch('train', epoch)
            if self.test_dataset is not None:
                self.run_epoch('test', epoch)

            logger.log()

            self.save_checkpoint()
