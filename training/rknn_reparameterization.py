"""Small, dependency-light guard used before RKNN post-fold fine-tuning."""

import torch


@torch.no_grad()
def verify_reparameterization(student, sample_videos, tolerance=1e-3):
    """Fold the backbone and check the fold stayed numerically faithful.

    Conv+BN folding is an exact identity, so any difference is float32 reordering
    noise. That noise scales with the magnitude of the folded weights, which is
    large by construction in the layers we are folding here, so the check is on
    relative error at the backbone output rather than absolute error at the
    logits. Logits are only checked for unchanged argmax.
    """
    was_training = student.training
    student.eval()
    # TF32 keeps only 10 mantissa bits, so on Ampere+ it alone produces ~1e-2
    # relative differences here and would mask a genuine fold error. Compare in
    # true fp32; training keeps whatever setting it had.
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        frames = sample_videos.flatten(0, 1)
        before_feat = student.backbone(frames)[0].float()
        before_logits = student.forward_seq(sample_videos).float()

        student.reparameterize()

        after_feat = student.backbone(frames)[0].float()
        after_logits = student.forward_seq(sample_videos).float()
    finally:
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        if was_training:
            student.train()

    feat_scale = before_feat.abs().max().clamp_min(1e-6)
    feat_error = ((before_feat - after_feat).abs().max() / feat_scale).item()
    logit_scale = before_logits.abs().max().clamp_min(1e-6)
    logit_error = ((before_logits - after_logits).abs().max() / logit_scale).item()
    argmax_kept = torch.equal(before_logits.argmax(1), after_logits.argmax(1))

    if not torch.isfinite(after_feat).all():
        raise RuntimeError("Reparameterization produced non-finite backbone features")
    if feat_error > tolerance:
        raise RuntimeError(
            f"Reparameterization mismatch: backbone relative error {feat_error:.6g} "
            f"exceeds tolerance {tolerance:.6g}"
        )
    if not argmax_kept:
        raise RuntimeError(
            f"Reparameterization changed predictions (backbone relative error "
            f"{feat_error:.6g}, logit relative error {logit_error:.6g})"
        )
    print(
        f"Reparameterization verified: backbone rel err {feat_error:.3g}, "
        f"logit rel err {logit_error:.3g}, argmax unchanged"
    )
    return feat_error
