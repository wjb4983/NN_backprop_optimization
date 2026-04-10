from __future__ import annotations

from nn_opt_rd.interfaces.core import Model


class MeanSquaredErrorLoss:
    def value_and_grad(
        self,
        model: Model,
        xs: list[float],
        ys: list[float],
    ) -> tuple[float, float]:
        n = max(1, len(xs))
        losses = []
        grad_sum = 0.0
        for x, y in zip(xs, ys):
            pred = model.predict(x)
            err = pred - y
            losses.append(err * err)
            grad_sum += 2.0 * err * x
        return sum(losses) / n, grad_sum / n
