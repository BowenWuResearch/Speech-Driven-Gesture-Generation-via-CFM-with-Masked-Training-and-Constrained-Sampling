from torch.optim.lr_scheduler import _LRScheduler


class LRSchedulerWrapper:
    """Handle restarting from a checkpoint with a learning rate scheduler.
    
    The problem is that when using cosineAnnealingLR for example,
    when assigned a a new larger total_steps, the scheduler will
    compute using this new total_steps, causing inconsistency.
    This class returns the last computed learning rate as a 
    constant from the previous run for future steps.
    However, it also provide a flag to use the new total_steps
    in load_state_dict function.
    """
    def __init__(self, scheduler: _LRScheduler, total_steps: int):
        self.scheduler = scheduler
        self.total_steps = total_steps
        
    def state_dict(self):
        return dict(
            total_steps=self.total_steps,
            scheduler_state=self.scheduler.state_dict()
        )
        
    def load_state_dict(self, state_dict, overwrite_total_steps: bool = True):
        if overwrite_total_steps:
            self.total_steps = state_dict['total_steps']
        self.scheduler.load_state_dict(state_dict['scheduler_state'])
        
    def get_last_lr(self):
        return self.scheduler.get_last_lr()
        
    def step(self, step: int):
        if step >= self.total_steps:
            # Do nothing to maintain the last computed learning rate
            pass
        else:
            self.scheduler.step()