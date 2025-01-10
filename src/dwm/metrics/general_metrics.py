import torch
import torchmetrics
from torchmetrics import MeanMetric
import torch.distributed


class CustomMeanMetrics(MeanMetric):
    """
    Description:
        calculate the mean value of certain metrics
    """

    def __init__(
        self, **kwargs
    ):
        super().__init__(**kwargs)

    def compute(self, **kwargs):
        self.num_samples = int(self.weight)
        return super().compute(**kwargs)
