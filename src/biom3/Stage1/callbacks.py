"""Stage 1 metrics history: explicit metric list, no prefix conventions.

Why this exists
---------------
Stage 1 reuses Stage3.callbacks.MetricsHistoryCallback, which harvests
`trainer.callback_metrics` by PREFIX -- keys starting with "train_" go to the
training history, keys starting with "val_" go to the validation history. Two
things fall through that filter in Stage 1:

  * Validation. Stage 1's validation_step logs `valid_loss`, `valid_loss_align`,
    `valid_loss_intra`, `valid_loss_text_mask`, `valid_loss_seq_mask` -- "valid_",
    not "val_". Nothing matched, so metrics_history.val.jsonl contained only
    {epoch, global_step} and `loss_gap` was never computed. We had no validation
    curve at all.

  * Learning rate. LearningRateMonitor IS registered and publishes to the
    loggers, but under keys like "lr-AdamW/pg1", which match neither prefix. So
    LR never reached the history arrays.

Rather than rename Stage 1's metrics to fit the convention (which would silently
change the names any --checkpoint_monitors reference) or widen the filter in
Stage 3's callback (which would affect Stage 3), this subclass names the metrics
it wants explicitly. Adding a metric means adding it to the list below; nothing
depends on how it happens to be spelled.

Only the three collection hooks are overridden. File handling, .pt snapshots,
rank filtering and crash-recovery flushing are inherited unchanged.
"""
from biom3.Stage3.callbacks import MetricsHistoryCallback

# Explicitly tracked metrics. Left-hand side is the key Lightning logs; the
# right-hand side is the name stored in the history. Anything absent from
# callback_metrics on a given step is simply skipped.
TRAIN_METRICS = {
    "train_loss": "train_loss",
    "train_loss_align": "train_loss_align",           # L_GC
    "train_loss_intra": "train_loss_intra",           # L_PFC
    "train_loss_text_mask": "train_loss_text_mask",   # L_bML
    "train_loss_seq_mask": "train_loss_seq_mask",     # L_pML
    # L_unif, logged whenever --uniformity_weight > 0 or --log_uniformity.
    # Absent keys are skipped, so this is inert on runs without it.
    "train_loss_unif_protein": "train_loss_unif_protein",
    "train_loss_unif_text": "train_loss_unif_text",
    "train_text_accuracy": "train_text_accuracy",
    "train_text_f1": "train_text_f1",
    "train_seq_accuracy": "train_seq_accuracy",
    "train_seq_f1": "train_seq_f1",
    "train_total_accuracy": "train_total_accuracy",
    "train_total_f1": "train_total_f1",
}

VAL_METRICS = {
    "valid_loss": "val_loss",
    "valid_loss_align": "val_loss_align",
    "valid_loss_intra": "val_loss_intra",
    "valid_loss_text_mask": "val_loss_text_mask",
    "valid_loss_seq_mask": "val_loss_seq_mask",
    "valid_loss_unif_protein": "val_loss_unif_protein",
    "valid_loss_unif_text": "val_loss_unif_text",
    "valid_text_accuracy": "val_text_accuracy",
    "valid_text_f1": "val_text_f1",
    "valid_seq_accuracy": "val_seq_accuracy",
    "valid_seq_f1": "val_seq_f1",
    "valid_total_accuracy": "val_total_accuracy",
    "valid_total_f1": "val_total_f1",
}

# Lightning suffixes epoch-reduced metrics with _epoch and per-step ones with
# _step. Accept the bare name first, then those, so a metric logged either way
# is captured.
_SUFFIXES = ("", "_epoch", "_step")


def _num(v):
    return v.item() if hasattr(v, "item") else v


def _collect(logged, spec):
    out = {}
    for src, dst in spec.items():
        for suf in _SUFFIXES:
            if (src + suf) in logged:
                out[dst] = _num(logged[src + suf])
                break
    return out


def _collect_lr(trainer):
    """Read the learning rate straight off the optimizers.

    Not from LearningRateMonitor's logged keys: it does populate
    callback_metrics, but under names it derives itself ("lr-AdamW",
    "lr-AdamW/pg1", ...) which vary with optimizer count, scheduler presence and
    whether param groups carry a `name`. Reading param_groups is unambiguous and
    depends on no naming convention.

    One column per group. Stage 1 has three (protein encoder 1e-4, text encoder
    1e-6, projection heads 1e-3) spanning three orders of magnitude, so a single
    number would hide what is actually happening.
    """
    out = {}
    try:
        for oi, opt in enumerate(trainer.optimizers):
            prefix = "lr" if len(trainer.optimizers) == 1 else f"lr_opt{oi}"
            for gi, pg in enumerate(opt.param_groups):
                if "lr" in pg:
                    out[f"{prefix}_pg{gi}"] = float(pg["lr"])
    except Exception:
        pass
    return out


class Stage1MetricsHistoryCallback(MetricsHistoryCallback):
    """MetricsHistoryCallback with explicit metric selection (see module docstring)."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank not in self.save_ranks:
            return
        if trainer.global_step % self.every_n_steps != 0:
            return
        logged = trainer.callback_metrics
        record = {"global_step": trainer.global_step,
                  "epoch": trainer.current_epoch, "source": "step"}
        record.update(_collect(logged, TRAIN_METRICS))
        record.update(_collect_lr(trainer))
        self.train_step_metrics.append(record)
        self._pending_train_jsonl.append(record)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank not in self.save_ranks:
            return
        epoch = trainer.current_epoch
        if self.every_n_epochs is not None and (epoch + 1) % self.every_n_epochs == 0:
            logged = trainer.callback_metrics
            record = {"global_step": trainer.global_step,
                      "epoch": epoch, "source": "epoch"}
            record.update(_collect(logged, TRAIN_METRICS))
            record.update(_collect_lr(trainer))
            self.train_step_metrics.append(record)
            self._pending_train_jsonl.append(record)
        self._flush_pending_train_jsonl()

    def on_validation_epoch_end(self, trainer, pl_module):
        logged = trainer.callback_metrics
        if self.all_ranks_val_loss:
            rec = {"global_step": trainer.global_step,
                   "epoch": trainer.current_epoch, "rank": trainer.global_rank}
            for suf in _SUFFIXES:
                if ("valid_loss" + suf) in logged:
                    rec["val_loss"] = _num(logged["valid_loss" + suf])
                    break
            self.per_rank_val_loss.append(rec)

        if trainer.global_rank not in self.save_ranks:
            return
        record = {"global_step": trainer.global_step, "epoch": trainer.current_epoch}
        record.update(_collect(logged, VAL_METRICS))

        # loss_gap = val_loss - train_loss, the overfitting signal
        val_loss = record.get("val_loss")
        train_loss = None
        for suf in ("_epoch", "", "_step"):
            if ("train_loss" + suf) in logged:
                train_loss = _num(logged["train_loss" + suf])
                break
        if val_loss is not None and train_loss is not None:
            record["loss_gap"] = val_loss - train_loss

        self.val_epoch_metrics.append(record)
        if self._val_jsonl_fh is not None:
            import json
            from biom3.Stage3.callbacks import _json_default
            self._val_jsonl_fh.write(json.dumps(record, default=_json_default) + "\n")
        self._flush_pending_train_jsonl()
        self._fsync_streams()
