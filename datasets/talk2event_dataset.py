import os
from torch.utils.data import Dataset
import json
import torch
import numpy as np
import cv2
import pickle
import json
from PIL import Image
# from utils import strefer_utils, pc_utils
# from tqdm import tqdm
from transformers import RobertaTokenizerFast
import datasets.transforms as T

import random
import kornia.augmentation as K
import kornia.augmentation.container as container

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

CLASSES = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train')
SRC_PATH = "/dataset/shared/magiclidar/"  # TODO: 修改成自己的路径
PIXEL_MEAN = [123.675, 116.280, 103.530]
PIXEL_STD = [58.395, 57.120, 57.375]

class Talk2EventDataset(Dataset):
    def __init__(self, args, image_set="train") -> None:
        super().__init__()

        print("Initializing Talk2EventDataset")
        self.args = args

        self.datasize = [480,640]
        #path for meta data
        meta_data_path = os.path.join(SRC_PATH, 'meta_data_v10_250423_appearance_added', 'meta_data_v10_250423_appearance_added', image_set)

        #sequence list
        self.dataset = []

        sequence_list = os.listdir(meta_data_path)
        # sequence_list = [sequence_list[4]]
        missing_attr_count = 0
        missing_attr_items = []
        for sequence in sorted(sequence_list):

            #load meta data
            meta_data = json.load(open(os.path.join(meta_data_path, sequence)))

            for data_item in meta_data:

                for idx in range(len(data_item['captions'])):
                    item = {}

                    item['id'] = data_item['id']
                    item['image_path'] = os.path.join(SRC_PATH, data_item['image_path'].replace('.jpg', '.png'))
                    item['event_path'] = os.path.join(SRC_PATH, data_item['event_path'])
                    item['bbox'] = data_item['bbox']
                    item['class'] = data_item['class']
                    item['other_num_objects'] = data_item['number of other objects']

                    caption = data_item['captions'][idx].lower()
                    attributes = data_item['attributes'][idx]['appearance'][0].lower()

                    caption = " ".join(caption.replace(",", " ,").replace(".", " .").split())  + ". not mentioned"
                    attributes = " ".join(attributes.replace(",", " ,").replace(".", " .").split())

                    _, _, matched_phrase = find_fuzzy_span(caption, attributes)

                    if matched_phrase is not None:     
                        item['caption'] = data_item['captions'][idx]
                        item['attributes'] = data_item['attributes'][idx]              
                        self.dataset.append(item)
                    else:
                        missing_attr_count += 1
                        item['caption'] = data_item['captions']
                        item['issue_caption'] = idx
                        item['attributes'] = data_item['attributes']  
                        missing_attr_items.append(item)

        # with open("/data/dylu/project/butd_detr/ls_attributes.json", "w", encoding="utf-8") as f:
        #     json.dump(missing_attr_items, f, indent=4)

        print(f"load {len(self.dataset)} data from {image_set} split")
        print(f"missing {missing_attr_count} attributes")

        #读取RoBERTa的tokenizer, 用于处理文本数据
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        # self.transforms = make_kornia_transform(image_set, cautious=True)
        self.transforms = make_coco_transforms(image_set, cautious=True)

        #image normalizer
        pixel_mean = np.array(PIXEL_MEAN).reshape(3, 1, 1).astype(np.float32)
        pixel_std = np.array(PIXEL_STD).reshape(3, 1, 1).astype(np.float32)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def __getitem__(self, index):
        data = self.dataset[index]

        #load image
        image_path = data["image_path"]
        image = Image.open(image_path).convert("RGB")
        #image to numpy
        # image = np.array(image, dtype=np.float32)  # H,W,C
        # image = torch.from_numpy(image).permute(2, 0, 1)# C,H,W
        # image = cv2.imread(image_path)#480,640,3,uint8
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)#3,480,640
        # image = torch.from_numpy(image)
        H,W = self.datasize

        # #load event
        event_path = data["event_path"]
        event_data = np.load(event_path)
        event = torch.from_numpy(event_data['events'].astype(np.float32))

        #load bbox
        bbox = data["bbox"]

        #change bbox to x, y, x_end, y_end
        x_end = (bbox['x'] + bbox['w'])
        y_end = (bbox['y'] + bbox['h'])
        gt_box = torch.as_tensor([
            bbox['x'], bbox['y'], x_end, y_end], dtype=torch.float32)
        gt_box[0::2].clamp_(0, W)
        gt_box[1::2].clamp_(0, H)
        gt_box = gt_box.unsqueeze(0)

        #load bbox class
        bbox_class = data["class"]
        bbox_class = CLASSES.index(bbox_class)
        bbox_class = torch.as_tensor(bbox_class, dtype=torch.int64)
        bbox_class = bbox_class.unsqueeze(0)

        #load caption
        caption = data["caption"].lower()
        caption = " ".join(caption.replace(",", " ,").replace(".", " .").split())  + ". not mentioned"

        #load attributes
        attributes = data["attributes"]['appearance'][0].lower()
        attributes = " ".join(attributes.replace(",", " ,").replace(".", " .").split())

        #get tokens_positive
        tokens_positive_list, tokens_positive = [], []
        start, end, _ = find_fuzzy_span(caption, attributes)
        if start<0:
            start = 0
        tokens_positive.append((start,end)) 
        # tokens_positive.append(end)
        tokens_positive_list.append(tokens_positive)

        tokenized = self.tokenizer(caption, return_tensors="pt")
        positive_map = create_positive_map(tokenized, tokens_positive_list)

        target = {}
        # target["event"] = event
        target["boxes"] = gt_box
        target["labels"] = bbox_class
        target["caption"] = caption
        target["tokens_positive"] = tokens_positive_list
        target["positive_map"] = positive_map
        target["orig_size"] = torch.as_tensor([int(H), int(W)])
        target["size"] = torch.as_tensor([int(H), int(W)])
        target["category"] = data['class']
        target["other_num_objects"] = data["other_num_objects"]

        if self.transforms is not None:
            image, event, target = self.transforms(image, event, target)

        input = {}
        input["image"] = image
        input["event"] = event

        return input, target

    def __len__(self):
        return len(self.dataset)


def save_pkl(file, output_path):
    output = open(output_path, "wb")
    pickle.dump(file, output)
    output.close()

def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


from rapidfuzz import fuzz, process

import re

def find_fuzzy_span(caption, label):
    caption_words = re.findall(r'\w+(?:-\w+)?', caption)  # include hyphenated words
    best_score = 0
    best_phrase = ""
    window_size = len(label.split())  # window size is the length of the label
    # Slide a window over the caption to compare n-grams
    for i in range(len(caption_words) - window_size + 1):
        for j in range(window_size-2, window_size + 2):
            span_words = caption_words[i:i+j]
            phrase = " ".join(span_words)
            score = fuzz.ratio(label, phrase)
            if score > best_score:
                best_score = score
                best_phrase = phrase
    
    if best_score > 70:  # threshold can be tuned
        start = caption.find(best_phrase)
        end = start + len(best_phrase)
        return start, end, best_phrase
    else:
        return None, None, None

def make_coco_transforms(image_set, cautious=False):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        horizontal = [] if cautious else [T.RandomHorizontalFlip()]
        return T.Compose(
            horizontal
            + [
            # T.RandomSelect(
            #     T.RandomResize(scales, max_size=1333),
            #     T.Compose([
            #         T.RandomResize([400, 500, 600]),
            #         T.RandomSizeCrop(384, 1333, respect_boxes=cautious),
            #         T.RandomResize(scales, max_size=1333),
            #     ])
            # ),
            normalize,
        ])

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            normalize,
        ])
        
    if image_set == 'train100':
        return T.Compose([
            normalize,
        ])
    

    raise ValueError(f'unknown {image_set}')