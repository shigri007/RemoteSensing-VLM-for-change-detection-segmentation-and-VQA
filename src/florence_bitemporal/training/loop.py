import math
from collections import defaultdict

import torch
import torch.nn.functional as F


def task_loss_weight(task, cfg):
    if task == "CHANGE_DETECTION":
        return cfg.LOSS_W_DET
    if task == "CHANGE_SEGMENTATION":
        return cfg.LOSS_W_SEG
    if task == "CHANGE_VQA":
        return cfg.LOSS_W_VQA
    return 1.0


def group_grad_norms(optimizer):
    output = {}
    for group in optimizer.param_groups:
        name = group.get("name", f"group_{id(group)}")
        squared = 0.0
        params_with_grad = 0
        params_total = len(group["params"])

        for param in group["params"]:
            if param.grad is not None:
                squared += float(param.grad.detach().pow(2).sum().item())
                params_with_grad += 1

        output[name] = {
            "grad_norm": math.sqrt(squared),
            "params_with_grad": params_with_grad,
            "params_total": params_total,
            "lr": group["lr"],
        }
    return output


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    epoch,
    cfg,
    device,
    accum_steps_override=None,
):
    from ..utils import log

    model.train()
    total_loss = 0.0
    num_batches = 0
    per_task_loss = defaultdict(lambda: [0.0, 0])
    accum = accum_steps_override or cfg.GRAD_ACCUM_STEPS
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    step = -1
    first_step_grad_norms = None

    def do_optimizer_step():
        nonlocal optimizer_steps
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            1.0,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    for step, batch in enumerate(loader):
        pixel_values_t1 = batch["pixel_values_t1"].to(device, dtype=torch.bfloat16)
        pixel_values_t2 = batch["pixel_values_t2"].to(device, dtype=torch.bfloat16)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        output = model(
            pixel_values_t1,
            pixel_values_t2,
            input_ids,
            attention_mask,
            labels=labels,
        )
        logits = output.logits
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.size())

        valid = (labels != -100).float()
        per_sample = (ce * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-9)

        weights = torch.tensor(
            [task_loss_weight(task, cfg) for task in batch["tasks"]],
            device=per_sample.device,
            dtype=per_sample.dtype,
        )
        loss = (per_sample * weights).mean()

        for index, task in enumerate(batch["tasks"]):
            per_task_loss[task][0] += per_sample[index].item()
            per_task_loss[task][1] += 1

        loss_for_backward = loss / accum
        loss_for_backward.backward()

        if first_step_grad_norms is None:
            first_step_grad_norms = group_grad_norms(optimizer)

        if (step + 1) % accum == 0:
            do_optimizer_step()

        total_loss += loss.item()
        num_batches += 1
        if step % 50 == 0:
            log(
                f"  Epoch {epoch} step {step}/{len(loader)}  "
                f"loss={loss.item():.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    if step >= 0 and (step + 1) % accum != 0:
        log(f"  Flushing trailing accumulation ({(step + 1) % accum} pending batches).")
        do_optimizer_step()

    log(f"  Optimizer steps this epoch: {optimizer_steps}")
    log("  Per-task mean loss:")
    for task, (total, count) in per_task_loss.items():
        log(f"    {task}: {total / max(count, 1):.4f}  (n={count})")

    if first_step_grad_norms:
        log("  Gradient flow (first backward of epoch):")
        for name, info in first_step_grad_norms.items():
            log(
                f"    {name:8s}  grad_norm={info['grad_norm']:.4e}  "
                f"params_with_grad={info['params_with_grad']}/{info['params_total']}  "
                f"lr={info['lr']:.2e}"
            )
            if info["grad_norm"] < 1e-10 and info["params_with_grad"] > 0:
                log(f"      *** WARNING: {name} group has near-zero grad. ***")

    return total_loss / max(num_batches, 1)
