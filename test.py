import torch
from torch.utils.data import DataLoader
import argparse
import sys
from tqdm import tqdm 
sys.path.append("/data/dylu/project/butd_detr")
from models import build_bdetr_model
from datasets import build_dataset
import util.misc as utils
from datasets.data_prefetcher import targets_to
from vis_tools.utils.common import rescale_bboxes
from vis_tools.utils.model_dataset import get_args_parser
from datasets.data_prefetcher import data_prefetcher

class Tester:
    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.device = 'cuda'
        self.init_args()
        self.build_model()
        self.build_dataloader()
        self.model.eval()

    def init_args(self):
        parser = argparse.ArgumentParser('Deformable', parents=[get_args_parser()], allow_abbrev=False )
        # args = parser.parse_args()
        args, unknown = parser.parse_known_args()

        args.output_dir = "exps/exps/all_appearance_event_data"
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
        args.resume = "exps/all_appearance_event_data/checkpoint0017.pth"
        args.eval
        
        self.config = args
        args.event_config = 'models/event/backbone.yaml'
        args.event_checkpoint = 'data/flexevent.ckpt'

    def build_model(self):
        model, _, _ = \
            build_bdetr_model(self.config)
        model.to(self.device)

        checkpoint = torch.load(self.config.resume, map_location='cpu')
        model.load_state_dict(checkpoint["model_ema"], strict=False)
        model.eval()
        self.model = model

    def build_dataloader(self):
        dataset = build_dataset(self.config.combine_datasets_val[0], "test", self.config)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=utils.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def post_process(self, outputs, targets, image_size):
        probas = 1 - outputs['pred_logits'].softmax(-1)[:, :, -1].cpu()
        keep = probas.argmax(dim=-1)
        expanded_idx = keep.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4)
        pred_boxes = torch.gather(outputs['pred_boxes'].cpu(), 1, expanded_idx).squeeze(1)
        gt_bboxes = torch.cat([item['boxes'] for item in targets], dim=0)

        bboxes_scaled = rescale_bboxes(pred_boxes, image_size)
        gt_bboxes_scaled = rescale_bboxes(gt_bboxes.cpu(), image_size)
        return (bboxes_scaled, gt_bboxes_scaled)

    def test(self):
        prefetcher = data_prefetcher(self.dataloader, self.device, prefetch=True)
        num_steps = int(len(self.dataloader))
        for i in tqdm(range(num_steps)):
            samples, event_samples, targets = prefetcher.next()
            samples = samples.to(self.device)
            event_samples = event_samples.to(self.device)
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
                event_samples,
                captions,
                encode_and_save=True,
                butd_boxes=butd_boxes,
                butd_classes=butd_classes,
                butd_masks=butd_masks
            )
            outputs = self.model(
                samples, event_samples, captions, encode_and_save=False,
                memory_cache=memory_cache,
                butd_boxes=butd_boxes,
                butd_classes=butd_classes,
                butd_masks=butd_masks
            )
            output_dict = self.post_process(outputs, targets, image_size=samples.tensors.shape[2:])
        return outputs
    
if __name__ == "__main__":
    tester = Tester(batch_size=2)
    tester.test()
