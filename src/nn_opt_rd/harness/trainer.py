from __future__ import annotations

from nn_opt_rd.config.schema import TrainConfig
from nn_opt_rd.controller.interface import Controller, ControllerState
from nn_opt_rd.diagnostics.basic import summarize_step
from nn_opt_rd.harness.losses import MeanSquaredErrorLoss
from nn_opt_rd.logging.metrics import MetricLogger
from nn_opt_rd.models.linear_model import LinearModel
from nn_opt_rd.optimizers.baselines import build_optimizer
from nn_opt_rd.probe.interface import Probe
from nn_opt_rd.tasks.linear_regression import LinearRegressionTask


def run_training(config: TrainConfig) -> dict:
    task = LinearRegressionTask(seed=config.seed)
    model = LinearModel(init_weight=0.0)
    optimizer = build_optimizer(config.optimizer, config.learning_rate)
    loss_fn = MeanSquaredErrorLoss()
    logger = MetricLogger(config.output_dir)
    controller = Controller()
    controller_state = ControllerState()
    probe = Probe()

    last_loss = 0.0
    for step in range(config.steps):
        xs, ys = task.sample_batch(config.batch_size)
        loss, grad = loss_fn.value_and_grad(model, xs, ys)
        metrics = summarize_step(loss, grad, model.parameters.value)

        controller.before_step(controller_state, metrics)
        optimizer.step(model.parameters, grad)
        probe.capture(step, metrics)

        last_loss = loss
        logger.log({"step": step, **metrics})

    summary = {
        "final_loss": last_loss,
        "final_weight": model.parameters.value,
        "steps": config.steps,
        "optimizer": config.optimizer,
    }
    logger.log({"summary": summary})
    return summary
