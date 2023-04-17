from utils.registry import MODEL_REGISTRY
from models.base_model import BaseModel

@MODEL_REGISTRY.register()
class Baseline(BaseModel):
    def __init__(self, opt, arch_opt=None):
        super(Baseline, self).__init__(opt=opt, arch_opt=arch_opt)

    # def sum(self, losses):
    #     sum_loss = 0.0
    #     for key, value in losses.items():
    #         sum_loss += value
    #     return sum_loss

    def manipulate_grad(self, losses):
        # baseline model do not need to manipulate the gradients
        loss = sum(losses.values())
        loss.backward()
        return
