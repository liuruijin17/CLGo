import torch
import torch.nn.functional as F
from torch import nn

from .misc import (NestedTensor, nested_tensor_from_tensor_list,
                   accuracy, get_world_size, interpolate,
                   is_dist_avail_and_initialized)
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, seq_len):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # threshold = 15 / 720.
        # self.threshold = nn.Threshold(threshold**2, 0.)
        empty_weight = torch.ones(self.num_classes + 1)
        # print('empty_weight: {}'.format(empty_weight))
        empty_weight[-1] = self.eos_coef
        # print('empty_weight: {}'.format(empty_weight))
        # exit()
        self.seq_len = seq_len

        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, target_flags, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        # print('src_logits: {}'.format(src_logits))
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([tgt[:, 0][J].long() for tgt, (_, J) in zip (targets, indices)])
        # print('target_classes_o: {}'.format(target_classes_o))
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        # print('target_classes: {}'.format(target_classes))
        target_classes[idx] = target_classes_o
        # print('target_classes: {}'.format(target_classes))
        # print('src_logits.transpose(1, 2).shape: {}'.format(src_logits.transpose(1, 2).shape))
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        # exit()

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, target_flags):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([tgt.shape[0] for tgt in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, target_flags):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # self.seq_len = (targets[0].shape[1] - 5) // 5
        self.seq_len = (targets[0].shape[1] - 5) // 8

        idx = self._get_src_permutation_idx(indices)
        src_lowers = outputs['pred_boxes'][:, :, 0][idx]
        src_uppers = outputs['pred_boxes'][:, :, 1][idx]
        src_gflatlowers = outputs['pred_boxes'][:, :, 2][idx]
        src_gflatuppers = outputs['pred_boxes'][:, :, 3][idx]
        src_polys       = outputs['pred_boxes'][:, :, 4:4+4][idx]
        src_2poly       = outputs['pred_boxes'][:, :, 4+4:4+4+4][idx]
        space3d_targets = [tgt[:, 3 + self.seq_len * 2:5 + self.seq_len * 5] for tgt in targets]
        target_flags  = torch.cat([tgt[i] for tgt, (_, i) in zip(target_flags, indices)], dim=0)
        # valid_xs = target_xs >= 0  #  Must comment for apollosim
        valid_xs = target_flags > 0
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32)) ** 0.5
        weights = weights / torch.max(weights)
        target_gflatlowers = torch.cat([tgt[:, 0][i] for tgt, (_, i) in zip(space3d_targets, indices)], dim=0)
        target_gflatuppers = torch.cat([tgt[:, 1][i] for tgt, (_, i) in zip(space3d_targets, indices)], dim=0)
        target_gflatpoints = torch.cat([tgt[:, 2:][i] for tgt, (_, i) in zip(space3d_targets, indices)], dim=0)
        target_gflatxs     = target_gflatpoints[:, :self.seq_len]
        gflatys            = target_gflatpoints[:, self.seq_len:self.seq_len*2].transpose(1, 0)
        target_gflatzs     = target_gflatpoints[:, self.seq_len*2:]
        pred_gflatxs       = src_polys[:, 0] * gflatys**3 + src_polys[:, 1] * gflatys**2 + src_polys[:, 2] * gflatys + src_polys[:, 3]
        pred_gflatxs       = pred_gflatxs * weights
        pred_gflatxs       = pred_gflatxs.transpose(1, 0)
        target_gflatxs     = target_gflatxs.transpose(1, 0) * weights
        target_gflatxs     = target_gflatxs.transpose(1, 0)
        pred_gflatzs       = src_2poly[:, 0] * gflatys ** 3 + src_2poly[:, 1] * gflatys ** 2 + src_2poly[:, 2] * gflatys + src_2poly[:, 3]
        pred_gflatzs       = pred_gflatzs * weights
        pred_gflatzs       = pred_gflatzs.transpose(1, 0)
        target_gflatzs     = target_gflatzs.transpose(1, 0) * weights
        target_gflatzs     = target_gflatzs.transpose(1, 0)
        loss_gflatlowers   = F.l1_loss(src_gflatlowers, target_gflatlowers, reduction='none')
        loss_gflatuppers   = F.l1_loss(src_gflatuppers, target_gflatuppers, reduction='none')
        loss_gflatpolys    = F.l1_loss(pred_gflatxs[valid_xs], target_gflatxs[valid_xs], reduction='none')
        loss_gflatzsys     = F.l1_loss(pred_gflatzs[valid_xs], target_gflatzs[valid_xs], reduction='none')

        losses = {}
        losses['loss_gflatlowers'] = loss_gflatlowers.sum() / num_boxes
        losses['loss_gflatuppers'] = loss_gflatuppers.sum() / num_boxes
        losses['loss_gflatpolys']  = loss_gflatpolys.sum() / num_boxes
        losses['loss_gflatzsys']   = loss_gflatzsys.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, target_flags):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, target_flags, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, target_flags, **kwargs)

    def forward(self, outputs, targets, targets_flag):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, targets_flag)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(tgt.shape[0] for tgt in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, targets_flag))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, targets_flag)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, targets_flag, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, indices