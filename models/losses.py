import torch

def control_point_l1_loss(pred_control_points,
                          gt_control_points,
                          device="cpu"):
    """
      Computes the l1 loss between the predicted control points and the
      groundtruth control points on the gripper.
    """
    # print('control_point_l1_loss', pred_control_points.shape,
    #      gt_control_points.shape)
    error = torch.sum(torch.abs(pred_control_points - gt_control_points), -1)
    error = torch.mean(error, -1)
    return torch.mean(error)


def classification_loss(pred_logit,
                        gt,
                        device="cpu"):
    """
      Computes the cross entropy loss. Returns cross entropy loss .
    """
    classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_logit, gt)
    return classification_loss


def kl_divergence(mu, log_sigma, device="cpu"):
    """
      Computes the kl divergence for batch of mu and log_sigma.
    """
    return torch.mean(
        -.5 * torch.sum(1. + log_sigma - mu**2 - torch.exp(log_sigma), dim=-1))
