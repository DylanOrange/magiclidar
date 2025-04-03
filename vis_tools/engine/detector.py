import torch
import argparse
import sys
sys.path.append("/data/dylu/project/butd_detr")
from models import build_bdetr_model
from datasets import build_dataset
import util.misc as utils
from datasets.data_prefetcher import targets_to

from ..utils.common import rescale_bboxes
from ..utils.model_dataset import get_args_parser

class Detector:
    def __init__(self):
        self.device = 'cuda'
        self.init_args()
        self.build_model()
        self.build_dataset()
        self.model.eval()

    def init_args(self):
        parser = argparse.ArgumentParser('Deformable', parents=[get_args_parser()], allow_abbrev=False )
        # args = parser.parse_args()
        args, unknown = parser.parse_known_args()

        args.output_dir = "exps/pretrain_talk2event"
        args.dataset_config = "configs/pretrain.json"
        args.batch_size = 2
        args.lr = 1e-5
        args.lr_backbone = 1e-6
        args.text_encoder_lr = 6e-6
        args.weight_decay = 1e-4
        args.large_scale = True
        args.save_freq = 1
        args.eval_skip = 1
        args.ema
        args.combine_datasets_val = ["talk2event"]
        args.resume = "exps/pretrain_talk2event/checkpoint0019.pth"
        args.eval

        self.config = args

    def build_model(self):
        model, _, _ = \
            build_bdetr_model(self.config)
        model.to(self.device)

        checkpoint = torch.load(self.config.resume, map_location='cpu')
        model.load_state_dict(checkpoint["model_ema"], strict=False)
        model.eval()
        self.model = model

    def build_dataset(self):
        dataset = build_dataset(self.config.combine_datasets_val[0], "val", self.config)
        self.dataset = dataset

    def infrence(self, index):
        (samples, targets) = utils.collate_fn([self.dataset.__getitem__(index)])
        samples = samples.to(self.device)
        targets = targets_to(targets, self.device)
        captions = [t["caption"] for t in targets]
        positive_map = torch.cat(
            [t["positive_map"] for t in targets])

        memory_cache = None
        butd_boxes = None
        butd_masks = None
        butd_classes = None
        if self.config.butd:
            butd_boxes = torch.stack([t['butd_boxes'] for t in targets], dim=0)
            butd_masks = torch.stack([t['butd_masks'] for t in targets], dim=0)
            butd_classes = torch.stack([t['butd_classes'] for t in targets], dim=0)
        memory_cache = self.model(
            samples,
            captions,
            encode_and_save=True,
            butd_boxes=butd_boxes,
            butd_classes=butd_classes,
            butd_masks=butd_masks
        )
        outputs = self.model(
            samples, captions, encode_and_save=False,
            memory_cache=memory_cache,
            butd_boxes=butd_boxes,
            butd_classes=butd_classes,
            butd_masks=butd_masks
        )
        temp = self.dataset.dataset[index]
        image_path = temp["image_path"]
        caption = captions[0]
        probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
        keep = (probas > 0.1).cpu()

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], samples.tensors.shape[2:])

        return outputs, image_path, caption, targets[0]['boxes'].detach().cpu().numpy(), bboxes_scaled.detach().cpu().numpy()

if __name__ == "__main__":
    detector = Detector()
    detector.infrence(0)