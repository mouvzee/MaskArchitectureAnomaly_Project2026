from lightning.pytorch.callbacks import BaseFinetuning


class FreezeAllExceptClassHead(BaseFinetuning):
    def freeze_before_training(self, pl_module):
        print("Freezing backbone before training.")
        self.freeze(pl_module.network)
        self.make_trainable(pl_module.network.class_head)
        self.make_trainable(pl_module.network.mask_head)
        self.make_trainable(pl_module.network.q)
        self.make_trainable(pl_module.network.upscale)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        pass


class FreezeEncoderOnly(BaseFinetuning):
    def freeze_before_training(self, pl_module):
        print("Freezing encoder before training.")
        self.freeze(pl_module.network.encoder)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        pass

class FreezeAllExceptTopBlocks(BaseFinetuning):
    def __init__(self, num_blocks=2):
        super().__init__()
        self.num_blocks = num_blocks

    def freeze_before_training(self, pl_module):
        print(f"Freezing all except top {self.num_blocks} blocks.")
        self.freeze(pl_module.network.encoder)
        blocks = list(pl_module.network.encoder.backbone.blocks)
        for block in blocks[-self.num_blocks:]:
            self.make_trainable(block)
        self.make_trainable(pl_module.network.encoder.backbone.norm)
        self.make_trainable(pl_module.network.class_head)
        self.make_trainable(pl_module.network.mask_head)
        self.make_trainable(pl_module.network.q)
        self.make_trainable(pl_module.network.upscale)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        pass
