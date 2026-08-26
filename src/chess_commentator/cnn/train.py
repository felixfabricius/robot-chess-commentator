def train(model, dataloader, loss_fns, loss_weights, optimizer, debug, device):
    """Run one epoch over the training split and return the losses of the last ~20% of batches.

    loss_fns: dict with keys "empty" (BCEWithLogitsLoss), "color" / "type" (CrossEntropyLoss
              with ignore_index=IGNORE_INDEX so empty rows are skipped automatically).
    loss_weights: dict with keys "empty"/"color"/"type" combining the three heads into the
                  single scalar that is back-propagated.
    debug: stop after a handful of batches (smoke run); the reported losses are then all zero.

    Only the tail of the epoch is averaged, because the early batches of an epoch are stale by
    the time it ends and would drag the number away from where the model actually is.
    """
    model.train()
    n_batches = len(dataloader)
    loss_logging_threshold = n_batches * 0.8 // 1
    n_loss_samples = 0
    empty_loss_total = 0
    color_loss_total = 0
    type_loss_total = 0
    total_loss_total = 0
    for batch, (X, metadata, is_piece, color_target, type_target) in enumerate(dataloader):
        X = X.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        # is_piece comes off the default collate as float64; BCEWithLogitsLoss needs the
        # target dtype to match the float32 logits.
        is_piece = is_piece.to(device, non_blocking=True).float()
        color_target = color_target.to(device, non_blocking=True)
        type_target = type_target.to(device, non_blocking=True)

        logit_empty, logits_color, logits_type = model(X, metadata)
        empty_loss = loss_fns["empty"](logit_empty, is_piece)
        color_loss = loss_fns["color"](logits_color, color_target)  # empty rows skipped via ignore_index
        type_loss = loss_fns["type"](logits_type, type_target)      # empty rows skipped via ignore_index
        loss = (
            loss_weights["empty"] * empty_loss
            + loss_weights["color"] * color_loss
            + loss_weights["type"] * type_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if debug and batch > 3:
            break

        if batch % 10 == 0:
            print(f"Batch {batch + 1:>4d} / {n_batches:>4d} | Loss: {loss.item():.2f}")

        if batch + 1 >= loss_logging_threshold:
            n_batch = is_piece.shape[0]
            # multiply by n_batch since reduction="mean"
            empty_loss_total += empty_loss.item() * n_batch
            color_loss_total += color_loss.item() * n_batch
            type_loss_total += type_loss.item() * n_batch
            total_loss_total += loss.item() * n_batch
            n_loss_samples += n_batch

    return {
        # can get DivByZero Error if debugging (nothing accumulated before the early break)
        "train/empty/recent_loss": empty_loss_total / n_loss_samples if not debug else 0,
        "train/color/recent_loss": color_loss_total / n_loss_samples if not debug else 0,
        "train/type/recent_loss": type_loss_total / n_loss_samples if not debug else 0,
        "train/total/recent_loss": total_loss_total / n_loss_samples if not debug else 0,
        "train/total/n_recent_loss": n_loss_samples,
    }
