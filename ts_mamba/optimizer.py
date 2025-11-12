import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []

        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * step / float(self.warmup_steps)
            
            else:
                progress = (step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine_decay

            lrs.append(lr)

        return lrs
