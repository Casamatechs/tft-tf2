import libs.utils as utils

class QuantileLossCalculator(object):
    """Computes the combined quantile loss for prespecified quantiles.

    Attributes:
    quantiles: Quantiles to compute losses
    """

    def __init__(self, quantiles, output_size):
        """Initializes computer with quantiles for loss calculations.

        Args:
            quantiles: Quantiles to use for computations.
            output_size: Output size
        """
        self.quantiles = quantiles
        self.output_size = output_size

    def quantile_loss(self, a, b):
        """Returns quantile loss for specified quantiles.

        Args:
            a: Targets
            b: Predictions
        """
        quantiles_used = set(self.quantiles)

        loss = 0.
        for i, quantile in enumerate(self.quantiles):
            if quantile in quantiles_used:
                loss += utils.tensorflow_quantile_loss(
                    a[Ellipsis, self.output_size * i:self.output_size * (i + 1)],
                    b[Ellipsis, self.output_size * i:self.output_size * (i + 1)], quantile)
        return loss