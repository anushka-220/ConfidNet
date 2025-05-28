import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuredMAPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            input: [B, C] tensor with scores
            target: [B, C] tensor with +1 (positive), 0 (unknown), -1 (negative)
            mask: [B, C] boolean tensor indicating valid positions

        Returns:
            loss: scalar structured MAP ranking loss
            ranking_lai: same shape as target, with computed ranking values
        """
        num_classes = input.shape[1]
        losses = []
        rankings = []

        for c in range(num_classes):
            x = input[:, c][mask[:, c]]
            y = target[:, c][mask[:, c]].long()

            if x.numel() == 0:
                losses.append(torch.tensor(0.0, device=input.device))
                rankings.append(torch.zeros_like(target[:, c]))
                continue

            pos_mask = y == 1
            neg_mask = y == 0
            num_pos = pos_mask.sum().item()
            num_neg = neg_mask.sum().item()

            if num_pos == 0 or num_neg == 0:
                losses.append(torch.tensor(0.0, device=input.device))
                rankings.append(torch.zeros_like(target[:, c]))
                continue

            # Rank inputs by value
            sorted_scores, indices = x.sort(descending=True)
            ranks = torch.zeros_like(x, dtype=torch.float)
            ranks[indices] = torch.arange(len(x), dtype=torch.float, device=x.device)

            # Compute AP
            sorted_labels = y[indices]
            precisions = []
            num_correct = 0
            for i in range(len(sorted_labels)):
                if sorted_labels[i] == 1:
                    num_correct += 1
                    precisions.append(num_correct / (i + 1))
            ap = sum(precisions) / (num_pos + 1e-8) if precisions else torch.tensor(0.0, device=x.device)

            # Ground truth ranking score
            ranking_gt = self._generate_ranking_from_labels(y)
            score_gt = self._ranking_score(x, y, ranking_gt)

            # Loss-augmented ranking
            ranking_lai = self._generate_ranking_from_labels(y, tie_break=True)
            score_lai = self._ranking_score(x, y, ranking_lai)

            # Structured MAP loss
            loss = 1.0 - ap + score_lai - score_gt
            losses.append(loss)
            padded_ranking = torch.zeros_like(target[:, c], dtype=torch.float)
            padded_ranking[mask[:, c]] = ranking_lai.float()
            rankings.append(padded_ranking)

        loss = torch.stack(losses).mean()
        ranking_lai = torch.stack(rankings, dim=1)
        return loss, ranking_lai

    def _generate_ranking_from_labels(self, labels: torch.Tensor, tie_break=False):
        n = len(labels)
        ranking = torch.zeros(n, device=labels.device, dtype=torch.long)
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == 1 and labels[j] == 0:
                    ranking[i] += 1
                    ranking[j] -= 1
                elif labels[i] == 0 and labels[j] == 1:
                    ranking[i] -= 1
                    ranking[j] += 1
                elif tie_break:
                    if i < j:
                        ranking[i] += 1
                        ranking[j] -= 1
                    else:
                        ranking[i] -= 1
                        ranking[j] += 1
        return ranking

    def _ranking_score(self, scores, labels, ranking):
        score = 0.0
        num_pos = (labels == 1).sum().item()
        num_neg = (labels == 0).sum().item()

        for i in range(len(scores)):
            for j in range(len(scores)):
                if labels[i] == 1 and labels[j] == 0:
                    if ranking[i] > ranking[j]:
                        score += scores[i] - scores[j]
                    else:
                        score -= scores[i] - scores[j]
        return score / (num_pos * num_neg + 1e-8)
