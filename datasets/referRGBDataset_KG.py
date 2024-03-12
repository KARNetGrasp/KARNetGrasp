from utils.grasp import asGraspRectangle, asGrasp
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Tuple
from pathlib import Path
import torch
import torchvision.transforms.functional as F
import os
from bisect import bisect_left
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageEnhance
from .enhance_fun import crop, bright_contrast_color, coarse_erasing, ereasingWord, FLIP, random_rotate
import numpy as np
from transformers import RobertaTokenizer, BertTokenizer

import random
from skimage.draw import polygon, disk

import matplotlib.pyplot as plt
import seaborn as sns
import copy
import json
from loguru import logger
import nltk
import math

import time
import cv2
cv2.setNumThreads(0)


def apply_mask(image, mask, color=[0, 1, 0], alpha=0.4):

    if isinstance(image, Image.Image) or isinstance(image, torch.Tensor):
        image = np.array(image)
    if isinstance(mask, Image.Image) or isinstance(mask, torch.Tensor):
        mask = np.array(mask)

    if 1. in mask:
        mask = mask.astype(np.float32) * 255.
    mask = mask.astype(np.uint8)
    result = copy.deepcopy(image)
    for c in range(3):
        result[:, :, c] = np.where(mask == 255,
                                  result[:, :, c] *
                                  (1 - alpha) + alpha*color[c]* 255,
                                  result[:, :, c])
    

    ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.drawContours(result, contours, 0, (255, 255, 255), 1)

    return Image.fromarray(result)
    

class RefGraspBase(torch.utils.data.Dataset): 
    def __init__(self, output_size=300, include_depth=False, include_rgb=True, use_mask=True, use_bbox=True, weak_aug=False, 
                 weak_aug_epoch=-1, strong_aug=False, strong_aug_epoch=-1, meanstd=True, KG=False, proto_num=3):
        self.output_size = output_size
        
        self.weak_aug = weak_aug
        self.weak_aug_epoch = weak_aug_epoch
        self.strong_aug = strong_aug
        self.strong_aug_epoch = strong_aug_epoch
        

        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.use_mask = use_mask
        self.use_bbox = use_bbox
        self.meanstd = meanstd
        self.sentence_raw = None
        self.KG = KG
        self.prot_num = proto_num

        self.expression_files = []

        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')    
        if self.KG:
            word_embedding_path = f"ExtractKG/RN101_{self.prot_num}_word_embedding.npy" 
            self.word_emb_dict = np.load(word_embedding_path, allow_pickle=True).item()

        adj_f = open('ExtractKG/adj_list.txt', 'r')
        self.adj_list = [jj.strip("\n") for jj in adj_f.readlines()]
        entity_f = open('ExtractKG/entity_list.txt', 'r')
        self.entity_list = [nn.strip("\n") for nn in entity_f.readlines()]
        self.epoch = -1


    def extract_KG(self, sentence):
        sentence = sentence.strip(".")
        words = sentence.split(" ")
        words = [word.replace("_", " ") for word in words]

        wrod_embedding = torch.zeros(self.prot_num, 512)
        for word in words:
            if word == "glue" or word == "stick":
                word = "glue stick"
            
            if word == "can":
                word = 'food can'


            if word in self.word_emb_dict.keys():
                    
                if word in self.word_emb_dict.keys():
                    wrod_embedding = self.word_emb_dict[word]
                break

        return wrod_embedding

    def get_rgb(self, idx):
        raise NotImplementedError()

    def get_depth(self, idx):
        raise NotImplementedError()
    
    def get_refmask(self, idx):
        raise NotImplementedError()
    
    def get_refbbox(self, idx):
        raise NotImplementedError()
    
    def get_refgrasp(self, idx):
        raise NotImplementedError()
    
    def get_token(self, idx, maskimg=0):
        raise NotImplementedError()
    
    def get_sentence(self, idx):
        raise NotImplementedError()
    
    def show_sentence(self):
        return self.sentence_raw
    
    def _crop2square(self, rgb_img, depth, refgrasp, refbbox, refmask):
        if refbbox is not None:
            min_x, min_y, max_x, max_y = refbbox
        else:
            rect = []
            for gr in refgrasp:
                rect.append(asGraspRectangle(gr))
            rect = np.array(rect)
            rect_T = np.swapaxes(rect, 1, 2)
            min_x, min_y, max_x, max_y = rect_T[:,0,:].min(), rect_T[:,1,:].min(), rect_T[:,0,:].max(), rect_T[:,1,:].max()
        if rgb_img is not None:
            W, H = rgb_img.size
        else:
            W, H = depth.size
        min_size = min(W, H)

        if (W - min_size) / 2 > min_x or (H - min_size) /2 > min_y: 
            if rgb_img is not None:
                rgb_img = rgb_img.crop((0, 0, min_size, min_size))
            if depth is not None:
                depth = depth.crop((0, 0, min_size, min_size))
            if refmask is not None:
                refmask = refmask.crop((0, 0, min_size, min_size))

        elif W - (W - min_size) /2 < max_x or H - (H - min_size) /2 < max_y: 
            if rgb_img is not None:
                rgb_img = rgb_img.crop((W-min_size, H-min_size, W-1, H-1))
            if depth is not None:
                depth = depth.crop((W-min_size, H-min_size, W-1, H-1))
            refgrasp = np.array(refgrasp) - np.array([W-min_size, H-min_size, 0, W-min_size, H-min_size])
            
            if refbbox is not None:
                refbbox = np.array(refbbox) - np.array([W-min_size, H-min_size, W-min_size, H-min_size])
            
            if refmask is not None:
                refmask = refmask.crop((W-min_size, H-min_size, W-1, H-1))
        else: 
            if rgb_img is not None:
                rgb_img = rgb_img.crop(((W - min_size) /2, (H-min_size)/2, W - (W - min_size) /2, H - (H - min_size) /2))
            if depth is not None:
                depth = depth.crop(((W - min_size) /2, (H-min_size)/2, W - (W - min_size) /2, H - (H - min_size) /2))

            refgrasp = np.array(refgrasp) - np.array([(W - min_size) /2, (H-min_size)/2, 0, (W - min_size) /2, (H-min_size)/2])
            
            if refbbox is not None:
                refbbox = np.array(refbbox) - np.array([(W - min_size) /2, (H-min_size)/2, (W - min_size) /2, (H-min_size)/2])
            
            if refmask is not None:
                refmask = refmask.crop(((W - min_size) /2, (H-min_size)/2, W - (W - min_size) /2, H - (H - min_size) /2))
        
        return rgb_img, depth, refgrasp, refbbox, refmask

    def normalize(self, rgb_img, depth, refgrasp, refbbox, refmask):
        
        rgb_img, depth, refgrasp, refbbox, refmask = self._crop2square(rgb_img, depth, refgrasp, refbbox, refmask)
        
        return rgb_img, depth, refgrasp, refbbox, refmask
    
    def resize(self, rgb_img, depth_img, refgrasp, refbbox, refmask):
        if rgb_img is not None:
            W_ori, H_ori = rgb_img.size
        else:
            W_ori, H_ori = depth_img.size
        if rgb_img is not None:
            rgb_img = rgb_img.resize((self.output_size, self.output_size))
        if depth_img is not None:
            depth_img = depth_img.resize((self.output_size, self.output_size))
        if refmask is not None:
            refmask = refmask.resize((self.output_size, self.output_size))

        scale_H = self.output_size / (H_ori*1.0 )
        scale_W = self.output_size / (W_ori*1.0 )

        if refbbox is not None:
            refbbox = np.array(refbbox) * np.array([scale_W, scale_H, scale_W, scale_H])

        refgrasp = np.array(refgrasp) * np.array([scale_W, scale_H, 1, scale_W, scale_H])

        return rgb_img, depth_img, refgrasp, refbbox, refmask
    
    def enhance(self, rgb_img, depth_img, refmask, refbbox, refgrasp, sentence=None):
       raise NotImplementedError()
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_refgrasp_img(self, refgrasps, shape=(480, 640)):
        pos_out = np.zeros(shape)
        ang_out = np.zeros(shape)
        width_out = np.zeros(shape) 

        for i, gr in enumerate(refgrasps):
            gr_rec = asGraspRectangle(gr) 
            x1, y1, theta, x2, y2 = gr
            rr, cc = polygon(gr_rec[:,1], gr_rec[:,0], shape) 
            ang_out[rr, cc] = np.deg2rad(theta)
            length = abs(x2 - x1)
            width_out[rr, cc] = length
            pos_out[rr, cc] = 1

        pos_out = self.GaussianPos(refgrasps, shape)
        return pos_out, ang_out, width_out 

    
    def GaussianPos(self, refgrasps, shape):
        def gaussian(x, y, x0, y0, r):
            g = np.exp(-((x-x0)**2 + (y-y0)**2)/(2*r**2))
            return g
        pos_out = np.zeros(shape)
        for i, gr in enumerate(refgrasps):
            x1, y1, theta, x2, y2 = gr
            cx, cy = round((x1 + x2) /2), round((y1 + y2) / 2 )
            radius = round(min(abs(x2-x1), abs(y2 - y1)) / 2)

            rr, cc = disk((cy, cx), radius, shape=shape)

            for i in range(len(rr)):
                g = gaussian(cc[i], rr[i], cx, cy, radius)
                if pos_out[rr[i], cc[i]] != 0:
                    pos_out[rr[i], cc[i]] = g if g > pos_out[rr[i], cc[i]] else pos_out[rr[i], cc[i]]
                else:
                    pos_out[rr[i], cc[i]] = gaussian(cc[i], rr[i], cx, cy, radius)

        return pos_out


    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
        
    def __getitem__(self, idx):

        if self.include_depth:
            depth_img = self.get_depth(idx)
        else:
            depth_img = None

        if self.include_rgb:
            rgb_img = self.get_rgb(idx)
        else:
            rgb_img = None
        if self.use_mask:
            refmask = self.get_refmask(idx)

        else:
            refmask = None
        if self.use_bbox:
            refbbox = self.get_refbbox(idx)

        else:
            refbbox = None
        

        refgrasp = self.get_refgrasp(idx)

        rgb_img_o, depth_img_o, refmask_o, refbbox_o, refgrasp_o = rgb_img, depth_img, refmask, refbbox, refgrasp

        sentence_raw = self.get_sentence(idx)

        
        rgb_img, depth_img, refmask, refbbox, refgrasp, sentence_aug = self.enhance(rgb_img, depth_img, refmask, refbbox, refgrasp, sentence_raw)


        if self.KG:
            rand_num = random.uniform(0, 1)
            if rand_num < 0.3: 
                kg_word_embeddings = torch.zeros(self.prot_num, 512)
            else:
                kg_word_embeddings = self.extract_KG(sentence_raw)

        else:
            kg_word_embeddings = torch.zeros(self.prot_num, 512)


        if refgrasp.shape[0] == 0:
            rgb_img, depth_img, refmask, refbbox, refgrasp = rgb_img_o, depth_img_o, refmask_o, refbbox_o, refgrasp_o

        rgb_img, depth_img, refgrasp, refbbox, refmask = self.normalize(rgb_img, depth_img, refgrasp, refbbox, refmask)

        rgb_img, depth_img, refgrasp, refbbox, refmask = self.resize(rgb_img, depth_img, refgrasp, refbbox, refmask)

        word_embeddings, word_attention_mask, sentence_aug = self.get_token(sentence_aug)

        self.sentence_raw = sentence_aug.replace("_", " ")

        if self.include_rgb:
            rgb_img = np.array(rgb_img)
            rgb_img = rgb_img.transpose(2, 0, 1)

        if  self.include_depth:
            depth_img = np.array(depth_img)

        if refmask is not None:
            refmask = np.array(refmask)

        

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)

        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)
            if not isinstance(x, torch.FloatTensor):
                x = x.float()
            if self.meanstd:
                x = x.div_(255.).sub_(self.mean).div_(self.std)
        
        pos_img, ang_img, width_img = self.get_refgrasp_img(refgrasp, shape=(rgb_img.shape[1], rgb_img.shape[2]))

        pos = self.numpy_to_torch(pos_img) 
        cos = self.numpy_to_torch(np.cos(2*ang_img)) 

        sin = self.numpy_to_torch(np.sin(2*ang_img)) 

        if self.use_mask:
            refmask = self.numpy_to_torch(refmask)

        if self.use_bbox:
            refbbox = torch.from_numpy(refbbox)

        refgrasp = torch.from_numpy(refgrasp)
        

        width_img = np.clip(width_img, 0.0, 150.0)/150.0 
        width = self.numpy_to_torch(width_img)
        t1 = time.time()


        return x, word_embeddings, word_attention_mask, pos, cos, sin, width, refgrasp, refbbox, refmask, kg_word_embeddings, idx
    

    
    def __len__(self):
        return len(self.expression_files) 
    

    def collate(self, data):

        x, word_embeddings, word_attention_mask, pos_img, cos_img, sin_img, width_img, refgrasps, refbbox, refmask, kg_word_embeddings, idx = zip(*data)


        x = torch.stack(x, dim=0)

        pos_img = torch.stack(pos_img, dim=0)
        cos_img = torch.stack(cos_img, dim=0)
        sin_img = torch.stack(sin_img, dim=0)
        width_img = torch.stack(width_img, dim=0)

        refgrasps = refgrasps
        
        if None not in refbbox:
            refbbox = torch.stack(refbbox, dim=0)
        else:
            refbbox = None
        
        if None not in refmask:
            refmask = torch.stack(refmask, dim=0)
        else:
            refmask = None
        
        word_embeddings = torch.stack(word_embeddings, dim=0)
        word_attention_mask = torch.stack(word_attention_mask, dim=0)

        
        kg_word_embeddings = torch.stack(kg_word_embeddings, dim=0).detach()
        return x, word_embeddings, word_attention_mask, pos_img, cos_img, sin_img, width_img, refgrasps, refbbox, refmask, kg_word_embeddings, idx
    



class RefOCIDGrasp(RefGraspBase):
    def __init__(self, data, split, max_tokens=20, tokenizer_name="bert", output_size=300, include_depth=False, include_rgb=True, use_mask=True, use_bbox=True, weak_aug=False, 
                 weak_aug_epoch=0, strong_aug=False, strong_aug_epoch=0, meanstd=True, KG=False, proto_num=3):
        super().__init__(output_size, include_depth, include_rgb, use_mask, use_bbox, weak_aug, 
                 weak_aug_epoch, strong_aug, strong_aug_epoch, meanstd, KG, proto_num)
        logger.info(f"loading {split} set......")
        self.data = data
        self.split = split
        self.max_tokens = max_tokens
        if split is not "train":
            self.weak_aug=False
            self.weak_aug_epoch = -1
            self.strong_aug = False
            self.strong_aug_epoch = -1
            self.KG = False

        logger.info(f'Data Augmentation!! weak augmentation({self.weak_aug}) from epoch {self.weak_aug_epoch}, strong augmentation({self.strong_aug}) from epoch {self.strong_aug_epoch}')

        with open(os.path.join(data, 'ref-OCID-Grasp', f"{split}_expressions.json"), "r") as f:
            self.examples = json.load(f)

        self.tokenizer_name = tokenizer_name

        if self.tokenizer_name == 'robert':
            self.tokenizer = RobertaTokenizer.from_pretrained('modelZoo/roberta-base')
        elif self.tokenizer_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('modelZoo/bert-base-uncased')
        elif self.tokenizer_name == 'clip':
            from models.tokenizer import tokenize
            self.tokenizer = tokenize
        else:
            print("tokenizer error!")
            exit()


        self.rgb_paths, self.sentences, self.bboxes, self.clutter_splits, self.ref_graspRectangles, self.class_names = [], [], [], [], [], []

        os.makedirs(os.path.join(self.data, "refer-cache"), exist_ok=True)
        cached_pt = os.path.join(self.data, "refer-cache", f"{self.split}-grasp-dataset.pt") 
        
        if os.path.exists(cached_pt):
           
            logger.info("loadding dataset from cache ...") 
            self.rgb_paths, self.sentences, self.bboxes, self.ref_graspRectangles, self.clutter_splits, self.instanceIDScene, self.class_names = torch.load(str(cached_pt))
            
            assert (
                {"train": 178647, "val": 12606, "test": 18804}.get(self.split, -1)
                == len(self.rgb_paths)
                == len(self.sentences)
                == len(self.bboxes)
                == len(self.ref_graspRectangles) 
                == len(self.clutter_splits)
                == len(self.examples)
                == len(self.instanceIDScene)
            ), "Error on load from cache!"
            logger.info("Done!") 
        else:
            self.rgb_paths, self.sentences, self.bboxes, self.ref_graspRectangles , self.clutter_splits, self.instanceIDScene, self.class_names = self._process_dataset()
            torch.save([self.rgb_paths, self.sentences, self.bboxes, self.ref_graspRectangles , self.clutter_splits, self.instanceIDScene, self.class_names], str(cached_pt))
        self.expression_files = list(zip(self.rgb_paths, self.sentences, self.bboxes, self.ref_graspRectangles, self.class_names))

    def get_rgb(self, idx):
        image_path = self.rgb_paths[idx]
        img = cv2.imread(image_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        return img
    
    def get_depth(self, idx):
        depth_path = self.rgb_paths[idx].replace('rgb', 'depth')
        depth = cv2.imread(depth_path, cv2.CV_16UC1)
        depth = Image.fromarray(depth)
        
        return depth
    
    def get_refmask(self, idx):

        mask_label_path = self.rgb_paths[idx].replace('rgb', 'label')

        np_mask = np.array(Image.open(mask_label_path).convert("L"))
        ref_mask = np.zeros(np_mask.shape)
        ref_mask[np_mask == self.instanceIDScene[idx].numpy()] = 1
        ref_mask = Image.fromarray(ref_mask.astype(np.uint8), mode="P")

        return ref_mask
    
    
    
    def get_refbbox(self, idx):
        ref_bbox = np.array(self.bboxes[idx])
        return ref_bbox

    def get_refgrasp(self, idx):
        ref_graspRectangle = self.ref_graspRectangles[idx]
        ref_graspbbox = asGrasp(ref_graspRectangle)

        return np.array(ref_graspbbox)


    def get_token(self, idx, maskimg=0):
        if isinstance(idx, str):
            sentence_raw = idx
        else:
            sentence_raw = self.sentences[idx]
        attention_mask = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens
        if self.tokenizer_name == 'bert':
            input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
        elif self.tokenizer_name == 'clip':
            input_ids = self.tokenizer(sentence_raw, self.max_tokens, True).squeeze(0)
        else:
            raise NameError
        input_ids = input_ids[:self.max_tokens]
        padded_input_ids[:len(input_ids)] = input_ids
        attention_mask[:len(input_ids)] = [1]*len(input_ids)
        word_embeddings = torch.tensor(padded_input_ids)
        word_attention_mask = torch.tensor(attention_mask) 
        if maskimg:
            pass

        return word_embeddings, word_attention_mask, sentence_raw  
    
    def get_sentence(self, idx):
        return self.sentences[idx]
    
    def get_class_name(self, idx):
        return self.class_names[idx]

    def drawGrasp(self, idx=0, fig=None, graspboxes=None, name='vis_refgrasp'):
        if fig is None or graspboxes is None:
            fig = self.get_rgb(idx)
            graspboxes = self.get_refgrasp(idx)
        else:
            graspboxes = list(graspboxes)
            if isinstance(fig, np.ndarray):             
                fig = Image.fromarray(fig.astype(np.uint8))
        fig = copy.deepcopy(fig)
        print("draw grasp")
        draw = ImageDraw.Draw(fig)
        if not isinstance(graspboxes, list):
            graspboxes = [graspboxes]
        for grasp in graspboxes:
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = asGraspRectangle(grasp)
            draw.line((x1,y1,x2,y2), fill='red', width=2)
            draw.line((x2,y2,x3,y3), fill='blue', width=2)
            draw.line((x3,y3,x4,y4), fill='red', width=2)
            draw.line((x4,y4,x1,y1), fill='blue', width=2)
        if name is not None:
            fig.save(name+'.png')
        return fig
        

    def drawBbox(self, idx=0, fig=None, bbox=None, name='vis_refBbox'):
        if fig is None or bbox is None:
            fig = self.get_rgb(idx)
            bbox = self.get_refbbox(idx)
        else:
            if isinstance(fig, np.ndarray):             
                fig = Image.fromarray(fig.astype(np.uint8))
        print("draw bbox")
        fig = copy.deepcopy(fig)
        draw = ImageDraw.Draw(fig)
        bx1, by1, bx2, by2 = bbox
        draw.rectangle([bx1, by1, bx2, by2], outline=(255, 0, 0))
        if name is not None:
            fig.save(name+'.png')
        return fig
    
    def drawMask(self, idx=0, fig=None, mask=None, name=None):
        print("draw mask")
        if fig is None or mask is None:
            fig = self.get_rgb(idx)
            mask = self.get_refmask(idx)
        else:
            if isinstance(fig, np.ndarray):             
                fig = Image.fromarray(fig.astype(np.uint8))

        img_vis = apply_mask(fig, mask)

        if name is not None:
            img_vis.save(name + '.png')
        return img_vis

    def _process_dataset(self):
        def get_split(take_id: int) -> int:
            assert 0 <= take_id <= 20, "Bounds violation - `take_id` must be in [0, 20]!"
            return bisect_left([9, 16, 20], take_id)

        rgb_paths, language, bboxes, clutter_splits, instanceIDScenes, ref_graspRectangles, class_names = [], [], [], [], [], [], []
        for _idx, key in tqdm(enumerate(self.examples), total=len(self.examples), leave=False):
            example = self.examples[key]


            rgb_path, lang = os.path.join(self.data , "OCID_grasp" , example["scene_path"]), example["sentence"]
            rgb_name = example["scene_path"].split('/')[-1]
            instanceIDScene = example["scene_instance_id"]
            clutter_split = get_split(example["take_id"])
            class_name = example['class']
            anno_path = os.path.join(self.data, "OCID_grasp" ,example["sequence_path"], 'Annotations', rgb_name[:-4] + '.txt')
            allgraspRectangle_array = self._readGraspfromTxt(anno_path)
            
            ref_graspRectangle = self._filter_graspRectangle_from_maskref(allgraspRectangle_array, rgb_path, instanceIDScene)

            bbox = torch.tensor(json.loads(example["bbox"]), dtype=torch.int)
            assert bbox[0] < bbox[2] <= 640 and bbox[1] < bbox[3] <= 480, "Invalid Bounding Box Size!"


            rgb_paths.append(rgb_path)
            language.append(lang)
            bboxes.append(bbox)
            clutter_splits.append(clutter_split)
            instanceIDScenes.append(instanceIDScene)
            ref_graspRectangles.append(ref_graspRectangle)
            class_names.append(class_name)

        j = 0
        for grasp in ref_graspRectangles:
            if len(grasp) == 0:
                j+=1
        print("zero :", j)
        return rgb_paths, language, torch.stack(bboxes), ref_graspRectangles, torch.tensor(clutter_splits, dtype=torch.int), torch.tensor(instanceIDScenes, dtype=torch.int), class_names
        
    def _readGraspfromTxt(self, path):  
        with open(path, "r") as f:
            points_list = []
            grasp_boxes_list = []
            for count, line in enumerate(f):
                line = line.rstrip()
                [x, y] = line.split(' ')

                x = float(x)
                y = float(y)

                pt = (x, y)
                points_list.append(pt)

                if len(points_list) == 4:
                    grasp_boxes_list.append(points_list)
                    points_list = []

        grasp_box_arry = np.asarray(grasp_boxes_list)
        return grasp_box_arry 
    
    def _filter_graspRectangle_from_maskref(self, all_graspsRectangle, rgb_path, instanceIDScene):
        ref_graspsRectangle = []

        refmasks = rgb_path.replace('rgb', 'label')
        np_mask = np.array(Image.open(refmasks).convert("L"))
        ref_mask = np.zeros(np_mask.shape)
        ref_mask[np_mask == instanceIDScene] = 1
        for grasp in all_graspsRectangle: 
            
            [x1, y1], [x2, y2], [x3, y3], [x4, y4] = grasp
            cx, cy = (x1 + x2 + x3 + x4) / 4.0, (y1 + y2 + y3 + y4) / 4.0
            if ref_mask[round(cy), round(cx)]:
                ref_graspsRectangle.append(np.array(grasp))
        if len(ref_graspsRectangle) == 0:
            print("Goals that don't exist:", rgb_path, instanceIDScene)
        return ref_graspsRectangle

    
    
    def enhance(self, rgb_img, depth_img, refmask, refbbox, refgrasp, sentence=None):
        if self.epoch > 10:
            return rgb_img, depth_img, refmask, refbbox, refgrasp, sentence
        coin = random.choice([not (self.strong_aug and self.epoch >= self.strong_aug_epoch), self.weak_aug and self.epoch >= self.weak_aug_epoch])
        if self.weak_aug and self.epoch >= self.weak_aug_epoch and coin:
            rgb_img, depth_img, refmask, refbbox, refgrasp = crop(rgb_img, depth_img, refmask, refbbox, refgrasp, probability=0.5) 
            rgb_img = bright_contrast_color(rgb_img, probability=0.5)
            rgb_img, depth_img, refmask, refbbox, refgrasp = random_rotate(rgb_img, depth_img, refmask, refbbox, refgrasp, probability=0.5)
        if self.strong_aug and self.epoch >= self.strong_aug_epoch and not coin:
            rgb_img = coarse_erasing(rgb_img, probability = 0.3, min_h_ratio=0.001, max_h_ratio=0.02, w_aspect_ratio=0.6, eras_num=20)
            rgb_img, depth_img, sentence, refmask, refbbox, refgrasp = FLIP(rgb_img, depth_img, sentence, refmask, refbbox, refgrasp, probability=0.3)
            sentence = ereasingWord(sentence, entity_list=self.entity_list, adj_list=self.adj_list, probability=0.3, max_era=2)

        refgrasp = np.array(refgrasp)

        return rgb_img, depth_img, refmask, refbbox, refgrasp, sentence


