# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MDETR 
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import ipdb
st = ipdb.set_trace

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 moe_fusion: bool = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Softmax(-1)
        self.moe_fusion = moe_fusion
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets, positive_map):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():

            if len(outputs["pred_logits"])==4:
                #moe
                bs, num_queries = outputs["pred_logits"].shape[1:3]

                # gate = outputs['max_gate']
                # positive_maps = torch.stack(torch.chunk(positive_map, chunks=4, dim=0), dim=0)
                # batch_idx = torch.arange(gate.size(0), device=gate.device)  # [0,1,...,B-1]
                # positive_map = positive_maps[gate, batch_idx] 
                # logits = outputs["pred_logits"][gate, batch_idx]
                # out_prob = self.norm(logits.flatten(0, 1))
                # cost_class = -torch.matmul(out_prob, positive_map.T)

                cost_class_list = []
                positive_map = torch.chunk(positive_map, chunks=4, dim=0)
                for idx, pred_logit in enumerate(outputs["pred_logits"]):
                    out_prob = self.norm(pred_logit.flatten(0, 1))
                    cost_class = -torch.matmul(out_prob, positive_map[idx].T)
                    cost_class_list.append(cost_class)
                
                #cost_class is mean of the list
                cost_class = torch.mean(torch.stack(cost_class_list), dim=0)
                
            else:

                bs, num_queries = outputs["pred_logits"].shape[:2]
                # We flatten to compute the cost matrices in a batch
                out_prob = self.norm(outputs["pred_logits"].flatten(0, 1))
                cost_class = -torch.matmul(out_prob, positive_map.T)

            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
            # assert len(tgt_bbox) == len(positive_map)

            # Compute the soft-cross entropy between the predicted token alignment and the GT one for each box
            # cost_class = -torch.matmul(out_prob, positive_map.T)
            
            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            # print('cost_bbox mean ', cost_bbox.abs().mean())
            # print('cost_class mean ', cost_class.abs().mean())

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou,
                            moe_fusion=args.moe_fusion)
