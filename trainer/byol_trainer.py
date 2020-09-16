#-*- coding:utf-8 -*-
import time
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

from model import BYOLModel
from optimizer import LARS
from data import ImageNetLoader
from utils import params_util, logging_util, eval_util
from utils.data_prefetcher import data_prefetcher


class BYOLTrainer():
    def __init__(self, config):
        self.config = config
        self.time_stamp = self.config['checkpoint'].get('time_stamp',
            datetime.datetime.now().strftime('%m%d_%H-%M'))

        """device parameters"""
        self.world_size = self.config['world_size']
        self.rank = self.config['rank']
        self.gpu = self.config['local_rank']
        self.distributed = self.config['distributed']

        """get the train parameters!"""
        self.total_epochs = self.config['optimizer']['total_epochs']
        self.warmup_epochs = self.config['optimizer']['warmup_epochs']

        self.train_batch_size = self.config['data']['train_batch_size']
        self.val_batch_size = self.config['data']['val_batch_size']
        self.global_batch_size = self.world_size * self.train_batch_size

        self.num_examples = self.config['data']['num_examples']
        self.warmup_steps = self.warmup_epochs * self.num_examples // self.global_batch_size
        self.total_steps = self.total_epochs * self.num_examples // self.global_batch_size

        base_lr = self.config['optimizer']['base_lr'] / 256
        self.max_lr = base_lr * self.global_batch_size

        self.base_mm = self.config['model']['base_momentum']

        """construct the whole network"""
        self.resume_path = self.config['checkpoint']['resume_path']
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.gpu}')
            torch.cuda.set_device(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        self.construct_model()

        """save checkpoint path"""
        self.save_epoch = self.config['checkpoint']['save_epoch']
        self.ckpt_path = self.config['checkpoint']['ckpt_path'].format(
            self.time_stamp, self.config['model']['backbone']['type'], {})

        """log tools in the running phase"""
        self.steps = 0
        self.log_step = self.config['log']['log_step']
        self.logging = logging_util.get_std_logging()
        if self.rank == 0:
            self.writer = SummaryWriter(self.config['log']['log_dir'])

    def construct_model(self):
        """get data loader"""
        self.stage = self.config['stage']
        assert self.stage == 'train', ValueError(f'Invalid stage: {self.stage}, only "train" for BYOL training')
        self.data_ins = ImageNetLoader(self.config)
        self.train_loader = self.data_ins.get_loader(self.stage, self.train_batch_size)

        self.sync_bn = self.config['amp']['sync_bn']
        self.opt_level = self.config['amp']['opt_level']
        print(f"sync_bn: {self.sync_bn}")

        """build model"""
        print("init byol model!")
        net = BYOLModel(self.config)
        if self.sync_bn:
            net = apex.parallel.convert_syncbn_model(net)
        self.model = net.to(self.device)
        print("init byol model end!")

        """build optimizer"""
        print("get optimizer!")
        momentum = self.config['optimizer']['momentum']
        weight_decay = self.config['optimizer']['weight_decay']
        exclude_bias_and_bn = self.config['optimizer']['exclude_bias_and_bn']
        params = params_util.collect_params([self.model.online_network, self.model.predictor],
                                            exclude_bias_and_bn=exclude_bias_and_bn)
        self.optimizer = LARS(params, lr=self.max_lr, momentum=momentum, weight_decay=weight_decay)

        """init amp"""
        print("amp init!")
        self.model, self.optimizer = amp.initialize(
            self.model, self.optimizer, opt_level=self.opt_level)

        if self.distributed:
            self.model = DDP(self.model, delay_allreduce=True)
        print("amp init end!")

    # resume snapshots from pre-train
    def resume_model(self, model_path=None):
        if model_path is None and not self.resume_path:
            self.start_epoch = 0
            self.logging.info("--> No loaded checkpoint!")
        else:
            model_path = model_path or self.resume_path
            checkpoint = torch.load(model_path, map_location=self.device)

            self.start_epoch = checkpoint['epoch']
            self.steps = checkpoint['steps']
            self.model.load_state_dict(checkpoint['model'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            amp.load_state_dict(checkpoint['amp'])
            self.logging.info(f"--> Loaded checkpoint '{model_path}' (epoch {self.start_epoch})")

    # save snapshots
    def save_checkpoint(self, epoch):
        if epoch % self.save_epoch == 0 and self.rank == 0:
            state = {'config': self.config,
                     'epoch': epoch,
                     'steps': self.steps,
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'amp': amp.state_dict()
                    }
            torch.save(state, self.ckpt_path.format(epoch))

    def adjust_learning_rate(self, step):
        """learning rate warm up and decay"""
        max_lr = self.max_lr
        min_lr = 1e-3 * self.max_lr
        if step < self.warmup_steps:
            lr = (max_lr - min_lr) * step / self.warmup_steps + min_lr
        else:
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos((step - self.warmup_steps) * np.pi / self.total_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_mm(self, step):
        self.mm = 1 - (1 - self.base_mm) * (np.cos(np.pi * step / self.total_steps) + 1) / 2

    def forward_loss(self, preds, targets):
        bz = preds.size(0)
        preds_norm = F.normalize(preds, dim=1)
        targets_norm = F.normalize(targets, dim=1)
        loss = 2 - 2 * (preds_norm * targets_norm).sum() / bz
        return loss

    def train_epoch(self, epoch, printer=print):
        batch_time = eval_util.AverageMeter()
        data_time = eval_util.AverageMeter()
        forward_time = eval_util.AverageMeter()
        backward_time = eval_util.AverageMeter()
        log_time = eval_util.AverageMeter()
        loss_meter = eval_util.AverageMeter()

        self.model.train()

        end = time.time()
        self.data_ins.set_epoch(epoch)

        prefetcher = data_prefetcher(self.train_loader)
        images, _ = prefetcher.next()
        i = 0
        while images is not None:
            i += 1
            self.adjust_learning_rate(self.steps)
            self.adjust_mm(self.steps)
            self.steps += 1

            assert images.dim() == 5, f"Input must have 5 dims, got: {images.dim()}"
            view1 = images[:, 0, ...].contiguous()
            view2 = images[:, 1, ...].contiguous()
            # measure data loading time
            data_time.update(time.time() - end)

            # forward
            tflag = time.time()
            q, target_z = self.model(view1, view2, self.mm)
            forward_time.update(time.time() - tflag)

            tflag = time.time()
            loss = self.forward_loss(q, target_z)

            self.optimizer.zero_grad()
            if self.opt_level == 'O0':
                loss.backward()
            else:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            self.optimizer.step()
            backward_time.update(time.time() - tflag)
            loss_meter.update(loss.item(), view1.size(0))

            tflag = time.time()
            if self.steps % self.log_step == 0 and self.rank == 0:
                self.writer.add_scalar('lr', round(self.optimizer.param_groups[0]['lr'], 5), self.steps)
                self.writer.add_scalar('mm', round(self.mm, 5), self.steps)
                self.writer.add_scalar('loss', loss_meter.val, self.steps)
            log_time.update(time.time() - tflag)

            batch_time.update(time.time() - end)
            end = time.time()

            # Print log info
            if self.gpu == 0 and self.steps % self.log_step == 0:
                printer(f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t'
                        f'Step {self.steps}\t'
                        f'lr {round(self.optimizer.param_groups[0]["lr"], 5)}\t'
                        f'mm {round(self.mm, 5)}\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'Batch Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'Data Time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                        f'Forward Time {forward_time.val:.4f} ({forward_time.avg:.4f})\t'
                        f'Backward Time {backward_time.val:.4f} ({backward_time.avg:.4f})\t'
                        f'Log Time {log_time.val:.4f} ({log_time.avg:.4f})\t')

            images, _ = prefetcher.next()
