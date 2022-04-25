import os
import torch
import importlib
import torch.nn as nn
from thop import profile, clever_format
from config import system_configs
from models.py_utils.data_parallel import DataParallel

torch.manual_seed(317)
class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()
        self.model = model
        self.loss  = loss
    def forward(self, iteration, save, viz_split,
                xs, ys, **kwargs):
        preds = self.model(*xs, **kwargs)
        loss  = self.loss(iteration, save, viz_split, preds, ys, **kwargs)
        return loss
# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model
    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

class NetworkFactory(object):
    def __init__(self, db, flag=False, freeze=False, test_mode='Tv', train_mode='parallel'):
        super(NetworkFactory, self).__init__()
        if 'Feat' in system_configs.snapshot_name:
            module_file = "models.Pv-Tv_feat"
        elif 'GTR' in system_configs.snapshot_name:
            module_file = "models.Pv-Tv_gtr"
        elif 'MVTR' in system_configs.snapshot_name:
            module_file = "models.Pv-Tv_mvtr"
        elif 'IMG' in system_configs.snapshot_name:
            module_file = "models.Pv-Tv_joint"
        else:
            raise NotImplementedError
            # module_file = "models.Pv-Tv_joint"
        print("module_file: {}".format(module_file))
        nnet_module = importlib.import_module(module_file)
        self.model   = DummyModule(nnet_module.model(db, flag=flag, freeze=freeze, test_mode=test_mode, train_mode=train_mode))
        self.loss    = nnet_module.loss(db)
        self.network = Network(self.model, self.loss)
        self.network = DataParallel(self.network, chunk_sizes=system_configs.chunk_sizes)
        # self.flag    = flag

        # Count total parameters
        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("Total parameters: {}".format(total_params))

        # if 'Pv' in system_configs.snapshot_file:
        #     input_test = torch.randn(1, 3, 360, 480).cuda()
        #     input_mask = torch.randn(1, 3, 360, 480).cuda()
        #     macs, params, = profile(self.model, inputs=(input_test, input_mask), verbose=False)
        #     macs, params = clever_format([macs, params], "%.3f")
        #     print('Macs: {}'.format(macs))

        if system_configs.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
        elif system_configs.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate, 
                momentum=0.9, weight_decay=0.0001
            )
        elif system_configs.opt_algo == 'adamW':
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate,
                weight_decay=1e-4
            )
        # elif system_configs.opt_algo == 'CosineAnnealingLR':
        #     self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
        #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
        #                                                                 T_max=int(5990/8)+1,
        #                                                                 eta_min=0.00001)
        else:
            raise ValueError("unknown optimizer")

    def train(self, iteration, save, viz_split, xs, ys, **kwargs):
        xs = [x.cuda(non_blocking=True) for x in xs]
        ys = [y.cuda(non_blocking=True) for y in ys]
        self.optimizer.zero_grad()
        pv_loss_kp, tv_loss_kp = self.network(iteration, save, viz_split, xs, ys)
        pv_loss      = pv_loss_kp[0]
        pv_loss_dict = pv_loss_kp[1:]
        pv_loss      = pv_loss.mean()
        tv_loss      = tv_loss_kp[0]
        tv_loss_dict = tv_loss_kp[1:]
        tv_loss      = tv_loss.mean()
        loss = pv_loss + tv_loss
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()
        return loss, pv_loss_dict, tv_loss_dict

    def validate(self, iteration, save, viz_split, xs, ys, **kwargs):
        with torch.no_grad():
            pv_loss_kp, tv_loss_kp = self.network(iteration, save, viz_split, xs, ys)
            pv_loss      = pv_loss_kp[0]
            pv_loss_dict = pv_loss_kp[1:]
            pv_loss      = pv_loss.mean()
            tv_loss      = tv_loss_kp[0]
            tv_loss_dict = tv_loss_kp[1:]
            tv_loss      = tv_loss.mean()
            loss = pv_loss + tv_loss
            return loss, pv_loss_dict, tv_loss_dict

    def test(self, xs, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            return self.model(*xs, **kwargs)

    def set_lr(self, lr):
        print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def load_pretrained_params(self, pretrained_model):
        print("loading from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def load_params(self, iteration, is_bbox_only=False):
        if not is_bbox_only:
            cache_file = system_configs.snapshot_file.format(iteration)
            print("loading [J] model from {}".format(cache_file))
        else:
            cache_file = system_configs.box_snapshot_file.format(iteration)
            print("loading [BBox] model from {}".format(cache_file))
        with open(cache_file, "rb") as f:
            params = torch.load(f)
            model_dict = self.model.state_dict()
            if len(params) != len(model_dict):
                pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
            else:
                pretrained_dict = params
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

    def save_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f)

    def cuda(self):
        self.model.cuda()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

