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

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

CLASSES = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train')
SRC_PATH = "/dataset/dylu/data/talk2event/"  # TODO: 修改成自己的路径
PIXEL_MEAN = [123.675, 116.280, 103.530]
PIXEL_STD = [58.395, 57.120, 57.375]

class Talk2EventDataset(Dataset):
    def __init__(self, args, image_set="train") -> None:
        super().__init__()

        print("Initializing Talk2EventDataset")
        self.args = args

        self.datasize = [480,640]
        #path for meta data
        meta_data_path = os.path.join(SRC_PATH, 'meta_data_v7', "test")

        #sequence list
        self.dataset = []

        sequence_list = os.listdir(meta_data_path)
        # sequence_list = [sequence_list[4]]
        missing_attr_count = 0
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

                    caption = data_item['captions'][idx].lower()
                    attributes = data_item['attributes']['appearance'][0].lower()

                    caption = " ".join(caption.replace(",", " ,").replace(".", " .").split())  + ". not mentioned"
                    attributes = " ".join(attributes.replace(",", " ,").replace(".", " .").split())

                    _, _, matched_phrase = find_fuzzy_span(caption, attributes)

                    if matched_phrase is not None:     
                        item['caption'] = data_item['captions'][idx]
                        item['attributes'] = data_item['attributes']               
                        self.dataset.append(item)
                    else:
                        missing_attr_count += 1

        print(f"load {len(self.dataset)} data from test split")
        print(f"missing {missing_attr_count} attributes")

        # MARK: 读取RoBERTa的tokenizer, 用于处理文本数据
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
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
        # image = cv2.imread(image_path)#480,640,3,uint8
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)#3,480,640
        # image = torch.from_numpy(image)
        H,W = self.datasize

        # #load event
        # event_path = data["event_path"]
        # event_data = np.load(event_path)
        # event = event_data['events'].astype(np.float32)#20,480,640,float32

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
        target["boxes"] = gt_box
        target["labels"] = bbox_class
        target["caption"] = caption
        target["tokens_positive"] = tokens_positive_list
        target["positive_map"] = positive_map
        target["orig_size"] = torch.as_tensor([int(H), int(W)])
        target["size"] = torch.as_tensor([int(H), int(W)])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _get_token_positive_map(self, description, attributes, max_lang_num):
        """Return correspondence of boxes to tokens.
        找到目标物体名称在文本中的位置 (tokens_positive)。
        使用 Tokenizer 把文本转换为 Token (tokenized)。
        建立物体和 Token 的关联 (positive_map)，用于多模态任务。

        # 示例输入
         0      1       2       3      4     5      6     7    8
        <s>    the   chair   next    to    the   door    .   </s> # 每个 Token 位置

        # 返回结果
        tokens_positive

        positive_map
        目标索引 |  Token 0 | Token 1 | Token 2 (chair) | Token 3 | Token 4 | ...
        ------------------------------------------------------------
        目标 0   |    0    |    0    |      1        |    0    |    0    | ...
        目标 1   |    0    |    0    |      0        |    0    |    0    | ...
        目标 2   |    0    |    0    |      0        |    0    |    0    | ...
        ...

        """
        # Token start-end span in characters
        description = " ".join(description.replace(",", " ,").replace(".", " .").split())
        # caption = " " + caption + " "
        tokens_positive = np.zeros((self.max_objects, 2))

        # # 用 spaCy 提取目标物体名称
        # doc = self.nlp(caption)  # 使用 spaCy 解析句子，返回 doc 结构，其中 token 代表单词的 NLP 结构信息。
        # cat_names = []
        # for token in doc:
        #     if token.dep_ == "nsubj":  # nsubj 表示 名词性主语，通常是描述物体的核心词汇（如 "chair"、"table"）。
        #         cat_names.append(token.text)  # e.g. "chair"、"table"、"desk"
        #         break
        # if len(cat_names) <= 0:  # 如果找不到主语 (nsubj)，则寻找谓语 (ROOT)
        #     # ROOT 通常是句子的谓语动词（如 "is"、"stands"），可能连接了描述目标物体的名词。
        #     for token in doc:
        #         if token.dep_ == "ROOT":
        #             cat_names.append(token.text)
        #             break

        # 找到 cat_name 在文本中的位置
        # 先尝试 严格匹配 " cat_name "（确保完整单词）。
        # 若未找到，则 尝试匹配 " cat_name"（单词可能是句尾）。
        # 若仍未找到，则 尝试匹配 "cat_name"（可能没有前导空格）。
        # print(caption)
        # print(cat_names)
        label = attributes['appearance'][0].lower()
        start, end, matched_phrase = find_fuzzy_span(description, label)
        if start<0:
            start = 0
        tokens_positive[0][0] = start
        tokens_positive[0][1] = end
        
        print(f"label is: {label}")
        print(f"Matched phrase: {matched_phrase}")
        print(f"description: {description}")
        # for c, cat_name in enumerate(cat_names):
        #     start_span = caption.find(" " + cat_name + " ")
        #     len_ = len(cat_name)
        #     if start_span < 0:
        #         start_span = caption.find(" " + cat_name)
        #         len_ = len(caption[start_span + 1 :].split()[0])
        #     if start_span < 0:
        #         start_span = caption.find(cat_name)
        #         orig_start_span = start_span
        #         while caption[start_span - 1] != " ":
        #             start_span -= 1
        #         len_ = len(cat_name) + orig_start_span - start_span
        #         while caption[len_ + start_span] != " ":
        #             len_ += 1
        #     end_span = start_span + len_
        #     assert start_span > -1, caption  # 确保 cat_name 在 caption 中确实存在。
        #     assert end_span > 0, caption  # 确保终止索引有效。
        #     tokens_positive[c][0] = start_span  # 记录该目标物体的 token 位置。
        #     tokens_positive[c][1] = end_span

        # Positive map (for soft token prediction)
        # 把原始文本转换为 Token，每个单词都会被映射到 Token 索引
        tokenized = self.tokenizer.batch_encode_plus([description], padding="longest", return_tensors="pt")
        # 构造 positive_map，max_lang_num 语言描述的最大 Token 数, max_objects 为最多支持多少个物体
        positive_map = np.zeros((self.max_objects, max_lang_num))
        # 找到 Token 和字符索引的对应关系。
        gt_map = get_positive_map(tokenized, tokens_positive[: 1], max_lang_num)
        positive_map[: 1] = gt_map

        return tokens_positive, positive_map

    # def evaluate(self, predict_boxes, output_path=""):
    #     pred_boxes = []
    #     target_boxes = []
    #     eval_results = []
    #     idx = 0
    #     for pred_box in tqdm(predict_boxes):
    #         data = self.dataset[idx]
    #         scene_id = data["sequence_name"]  # data["scene_id"]
    #         # point_cloud_info = data["point_cloud"]
    #         # image_info = data["image"]
    #         point_cloud_name = data["lidar_path_proj"]  # point_cloud_info["point_cloud_name"]
    #         image_name = data["image_path"]  # image_info["image_name"]

    #         target = data["ground_info"]["bbox_3d"]  # point_cloud_info["bbox"]

    #         if output_path:
    #             ex_matrix = data["calibration"]["ex_matrix"]
    #             in_matrix = data["calibration"]["in_matrix"]
    #             language = data["language"]["description"]
    #             point_cloud_path = os.path.join(SRC_PATH, "points_rgbd", scene_id, f"{point_cloud_name}.npy")
    #             img_name = os.path.join(SRC_PATH, "image", scene_id, f"{image_name}.jpg")
    #             out_data = dict()
    #             out_data["gt_box"] = target
    #             out_data["pred_box"] = pred_box
    #             out_data["point_cloud_path"] = point_cloud_path
    #             out_data["image_path"] = img_name
    #             out_data["gt_corner2d"] = strefer_utils.batch_compute_box_3d([np.array(target)], ex_matrix, in_matrix)
    #             out_data["pred_corner2d"] = strefer_utils.batch_compute_box_3d([np.array(pred_box)], ex_matrix, in_matrix)
    #             out_data["language"] = language
    #             out_data["iou"] = pc_utils.cal_iou3d(pred_box, target)
    #             eval_results.append(out_data)

    #         target_boxes.append(target)
    #         idx += 1

    #     target_boxes = np.array(target_boxes)
    #     pred_boxes = predict_boxes

    #     if output_path:
    #         save_pkl(eval_results, output_path)
    #     acc25, acc50, miou = pc_utils.cal_accuracy(pred_boxes, target_boxes)
    #     return acc25, acc50, miou

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


def get_positive_map(tokenized, tokens_positive, max_lang_num):
    """Construct a map of box-token associations.

    # 过程解释
    "The red chair is next to the table."
    ['<s>', 'The', 'red', 'chair', 'is', 'next', 'to', 'the', 'table', '.', '</s>']
    "The" -> 1, "red" -> 2, "chair" -> 3, "is" -> 4, ..., "table" -> 8
    如果 tokens_positive = [[10, 15]]（即 "chair" 的字符索引范围），那么：
        beg_pos = 3
        end_pos = 3

    # 示例输出
    positive_map =
        [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ...],  # "chair" 第一行："chair" 映射到 Token 3。
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, ...]]  # "table"  第二行："table" 映射到 Token 8。
    # positive map 是用在 loss 里面的
    """
    positive_map = torch.zeros((len(tokens_positive), max_lang_num), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
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
        positive_map[j, beg_pos : end_pos + 1].fill_(1)

    # 归一化（Soft Mapping） 确保每行（每个物体）Token 之和为 1，这样可以用于 Soft Token Prediction。
    positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-12)
    return positive_map.numpy()

from rapidfuzz import fuzz, process

import re

def find_fuzzy_span(caption, label, window_size=5):
    caption_words = re.findall(r'\w+(?:-\w+)?', caption)  # include hyphenated words
    best_score = 0
    best_phrase = ""
    window_size = len(label.split())  # window size is the length of the label
    # Slide a window over the caption to compare n-grams
    for i in range(len(caption_words) - window_size + 1):
        for j in range(window_size-1, window_size + 2):
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
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 1333, respect_boxes=cautious),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
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