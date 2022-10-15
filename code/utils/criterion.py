from turtle import forward
import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence

# class SiLogLoss(nn.Module):
#     def __init__(self, lambd=0.5):
#         super().__init__()
#         self.lambd = lambd

#     def forward(self, pred, target):
#         valid_mask = (target > 0).detach()
#         diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
#         loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
#                           self.lambd * torch.pow(diff_log.mean(), 2))

#         return loss


class Adabin_loss(nn.Module):
    def __init__(self):
        super(Adabin_loss, self).__init__()
        self.name = 'Adabin'
        self.binchamfer_loss = BinsChamferLoss()
        self.silog = SILogLoss_ada()

    def forward(self, bins, input, target, mask, interpolate=True):
        loss1 = self.silog(input, target, mask, interpolate)
        loss2 = self.binchamfer_loss(bins, target)
        # print("Loss 1: ", loss1)
        # print("Loss 2: ", loss2)

        loss = loss1 + 0.1*loss2
        return loss

class SILogLoss_ada(nn.Module):  
    def __init__(self):
        super(SILogLoss_ada, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:

            input = torch.squeeze(input, dim=1)
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)

        Dg = torch.var(g) + 0.5 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)

        input = torch.clamp(input, min=1e-3, max=10)
        target = torch.clamp(target, min=1e-3, max=10)
        n, c, h, w = target.shape

        if mask is not None:
            input = input[mask]
            target = target[mask]
        di = torch.log(target) - torch.log(input)
        norm = 1/target.shape[0]
        di2 = torch.pow(di, 2)
        fisrt_term = torch.sum(di2)*norm
        second_term = torch.pow(torch.sum(di), 2)*(norm**2)
        loss = fisrt_term - 0.5*second_term
        return loss

class BinsChamferLoss(nn.Module):  
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss