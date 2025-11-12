class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass
    def watch(self, *args):
        pass
    def log_artifact(self, *args):
        pass
