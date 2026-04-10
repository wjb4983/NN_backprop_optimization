from __future__ import annotations

from nn_opt_rd.config.schema import TrainConfig
from nn_opt_rd.controller.interface import Controller, ControllerState
from nn_opt_rd.diagnostics.basic import RollingDiagnostics, summarize_step
from nn_opt_rd.harness.losses import MeanSquaredErrorLoss
from nn_opt_rd.logging.metrics import MetricLogger
from nn_opt_rd.models.linear_model import LinearModel
from nn_opt_rd.optimizers.baselines import build_optimizer
from nn_opt_rd.probe.interface import Probe
from nn_opt_rd.tasks.linear_regression import LinearRegressionTask


def _clip_grad(grad: float, threshold: float) -> tuple[float, bool, float]:
    if threshold <= 0.0:
        return grad, False, 0.0
    abs_grad = abs(grad)
    if abs_grad <= threshold:
        return grad, False, 0.0
    clipped_grad = threshold if grad > 0.0 else -threshold
    clip_ratio = (abs_grad - threshold) / abs_grad
    return clipped_grad, True, clip_ratio


def run_training(config: TrainConfig) -> dict:
    task = LinearRegressionTask(seed=config.seed)
    model = LinearModel(init_weight=0.0)
    optimizer = build_optimizer(config.optimizer, config.learning_rate)
    loss_fn = MeanSquaredErrorLoss()
    logger = MetricLogger(config.output_dir)
    controller = Controller(config.controller)
    controller_state = ControllerState()
    controller.bootstrap(controller_state)
    probe = Probe()
    rolling = RollingDiagnostics(window=config.controller.rolling_window)

    last_loss = 0.0
    for step in range(config.steps):
        xs, ys = task.sample_batch(config.batch_size)
        loss, grad = loss_fn.value_and_grad(model, xs, ys)

        clip_threshold = controller_state.clip_threshold if config.controller.enabled else float("inf")
        grad_for_step, clipped, clip_ratio = _clip_grad(grad, clip_threshold)

        pre_value = model.parameters.value
        optimizer.set_lr_multiplier(controller_state.lr_multiplier)
        optimizer.set_trust(controller_state.trust)
        if hasattr(optimizer, "set_momentum") and config.controller.enable_momentum_tweaks:
            optimizer.set_momentum(controller_state.momentum)

        optimizer.step(model.parameters, grad_for_step)
        update = model.parameters.value - pre_value

        step_metrics = summarize_step(
            loss=loss,
            grad=grad_for_step,
            weight=model.parameters.value,
            update=update,
            clipped=clipped,
            clip_ratio=clip_ratio,
        )
        features = rolling.update(
            loss=step_metrics["loss"],
            grad_norm=step_metrics["grad_abs"],
            update_norm=step_metrics["update_abs"],
            alignment=step_metrics["update_alignment"],
            clipped=clipped,
            clip_ratio=clip_ratio,
        )

        controller_metrics = controller.maybe_update(controller_state, step_metrics, features)
        payload = {"step": step, **step_metrics, **features, **controller_metrics}
        probe.capture(step, payload)

        last_loss = loss
        logger.log(payload)

    summary = {
        "final_loss": last_loss,
        "final_weight": model.parameters.value,
        "steps": config.steps,
        "optimizer": config.optimizer,
        "controller_enabled": int(config.controller.enabled),
        "interventions": controller_state.intervention_count,
    }
    logger.log({"summary": summary})
    return summary
