import logging
from pathlib import Path
from functools import cache
from typing import Callable, TypeVar

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from . import distributed
from .engine import gather_attribute, dispatch_attribute

logger = logging.getLogger(__name__)

T = TypeVar("T")


@cache
def init_distributed():
    distributed.init_distributed(backend="ddp")


class Engine:
    def __init__(self, model: nn.Module, hp, ckpt_dir: Path):
        init_distributed()
        device = torch.device("cuda")
        self.module = model.to(device)
        self.ddp = DDP(self.module, device_ids=[distributed.local_rank()])
        self.optimizer = torch.optim.Adam(self.ddp.parameters(), lr=hp.min_lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=hp.max_lr,
            total_steps=hp.max_steps,
        )
        self._ckpt_dir = ckpt_dir
        self._global_step = 0
        self._gradient_clipping = hp.gradient_clipping

    @property
    def path(self) -> Path:
        return self._ckpt_dir

    @property
    def gradient_clipping(self):
        return self._gradient_clipping

    @property
    def global_step(self) -> int:
        return self._global_step

    def train(self):
        self.ddp.train()

    def eval(self):
        self.ddp.eval()

    def __call__(self, *args, **kwargs):
        return self.ddp(*args, **kwargs)

    def backward(self, loss):
        loss.backward()

    def step(self):
        torch.nn.utils.clip_grad_norm_(self.ddp.parameters(), self.gradient_clipping)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self._global_step += 1

    def get_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

    def get_grad_norm(self):
        total = 0.0
        for p in self.ddp.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    def freeze_(self):
        for p in self.ddp.parameters():
            if p.requires_grad:
                p.requires_grad_(False)

    def unfreeze_(self):
        for p in self.ddp.parameters():
            if not p.requires_grad:
                p.requires_grad_(True)

    def gather_attribute(self, *args, **kwargs):
        return gather_attribute(self.module, *args, **kwargs)

    def dispatch_attribute(self, *args, **kwargs):
        return dispatch_attribute(self.module, *args, **kwargs)

    def save_checkpoint(self, tag: str = "default"):
        path = self._ckpt_dir / f"{tag}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.ddp.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self._global_step,
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(
        self,
        tag: str = "default",
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ):
        path = self._ckpt_dir / f"{tag}.pt"
        if not path.exists():
            return
        state = torch.load(path, map_location="cpu")
        self.ddp.load_state_dict(state["model"])
        self._global_step = state.get("step", 0)
        if load_optimizer_states:
            self.optimizer.load_state_dict(state["optimizer"])
        if load_lr_scheduler_states:
            self.scheduler.load_state_dict(state["scheduler"])
