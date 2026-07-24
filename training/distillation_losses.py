import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalPriorityLoss(nn.Module):
    def __init__(self, anchor_idx, binary_weight=1.1, fine_weight=1.0):
        super().__init__()
        self.anchor_idx = anchor_idx
        self.w_binary = binary_weight
        self.w_fine = fine_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, logits, targets):
        num_classes = int(logits.shape[1])
        if self.anchor_idx < 0 or self.anchor_idx >= num_classes:
            raise ValueError(f"anchor_idx={self.anchor_idx} out of range for num_classes={num_classes}")

        binary_targets = (targets != self.anchor_idx).float()
        probs = F.softmax(logits, dim=1)
        prob_anchor = probs[:, self.anchor_idx]
        prob_other = 1.0 - prob_anchor + 1e-7

        loss_binary = -(
            binary_targets * torch.log(prob_other) + (1 - binary_targets) * torch.log(prob_anchor)
        ).mean()

        mask_other = targets != self.anchor_idx
        loss_fine = torch.tensor(0.0, device=logits.device)
        if mask_other.sum() > 0:
            keep_indices = [idx for idx in range(num_classes) if idx != self.anchor_idx]
            other_logits = logits[mask_other][:, keep_indices]
            other_targets = targets[mask_other].clone()
            other_targets[other_targets > self.anchor_idx] -= 1
            loss_fine = self.ce(other_logits, other_targets)

        total_loss = self.w_binary * loss_binary + self.w_fine * loss_fine
        return total_loss, loss_binary.item(), loss_fine.item()


class CompositeDistillLoss(nn.Module):
    def __init__(self, gt_loss_fn, alpha=1.0, beta=8.0, gamma=1.0, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temp = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.gt_loss_fn = gt_loss_fn

    def forward(self, s_logits, t_logits, s_feats, t_feats, labels, batch_size, num_frames, epoch_ratio):
        loss_gt, l_bin, l_fine = self.gt_loss_fn(s_logits, labels)

        p_s = F.log_softmax(s_logits / self.temp, dim=1)
        p_t = F.softmax(t_logits / self.temp, dim=1)
        loss_kd = self.kl_div(p_s, p_t) * (self.temp ** 2)

        loss_feat = torch.tensor(0.0, device=s_logits.device)
        for idx, s_feat in enumerate(s_feats):
            t_feat = t_feats[idx].detach()
            if s_feat.dim() == 4 and t_feat.dim() == 5:
                channels, height, width = s_feat.shape[1:]
                s_feat = s_feat.view(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
            if s_feat.shape[-3:] != t_feat.shape[-3:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[-3:], mode="trilinear", align_corners=False)
            sim_map = F.cosine_similarity(s_feat, t_feat, dim=1)
            loss_feat += 1.0 - sim_map.mean()

        current_alpha = self.alpha * (1.0 - 0.5 * epoch_ratio)
        total_loss = self.gamma * loss_gt + self.beta * loss_kd + current_alpha * loss_feat
        return total_loss, l_bin, l_fine, loss_kd.item(), loss_feat.item()


def build_priority_composite_distill_loss(anchor_idx, alpha=1.0, beta=8.0, gamma=1.0, temperature=4.0):
    return CompositeDistillLoss(
        gt_loss_fn=HierarchicalPriorityLoss(anchor_idx=anchor_idx),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        temperature=temperature,
    )
