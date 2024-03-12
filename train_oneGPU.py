import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.loss import compute_loss
from loguru import logger
import torch
from torch.utils.data import DataLoader
from args import parse_args
from datasets.referRGBDataset_KG import RefOCIDGrasp

import time
import torch.optim as optim

import os
import shutil
from utils.evaluation import caculate_seg_iou, count_grasp_correct
from utils.post_process import post_process_output
from tqdm import tqdm
import numpy as np
from utils.draw import draw_img_grasp

from utils.evaluation import detect_grasps

from utils import utils_

import wandb

import datetime

def train_one_epoch(args, epoch, network, device, train_data, optimizer, lr_scheduler):
    network.train()
    metric_logger = utils_.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    batch_idx = 0

    for img, word_embeddings, word_attention_mask, pos_img, cos_img, sin_img, width_img, refgrasps, refbbox, refmask, kg_word_embedding, idx in metric_logger.log_every(train_data, args.print_freq, header=header):
        batch_idx += 1

        img = img.to(device)
        word_embeddings = word_embeddings.to(device)
        word_attention_mask = word_attention_mask.to(device)
        pos_img = pos_img.to(device)
        cos_img = cos_img.to(device)
        sin_img = sin_img.to(device)
        width_img = width_img.to(device)
        refmask = refmask.to(device)
        kg_word_embedding = kg_word_embedding.to(device)

        pred_seg, pos, cos, sin, width = network(img, word_embeddings, kg_word_embedding)

        loss = compute_loss(pred_seg, refmask, [pos, cos, sin, width], [pos_img, cos_img, sin_img, width_img])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        metric_logger.update(loss=loss)
        wandb.log({'loss':loss,'epoch': epoch})


        if batch_idx % args.print_freq == 0:

            MIoU_list, I_list, U_list = caculate_seg_iou(pred_seg, refmask)
            bs_miou = np.mean(np.array(MIoU_list))
            grasp_num, max_iou_list = count_grasp_correct([pos, cos, sin, width], refgrasps)
            bs_gacc =  grasp_num / len(refgrasps)
            bs_giou = np.mean(max_iou_list)

            metric_logger.update(gacc=bs_gacc, giou=bs_giou, miou=bs_miou)

            wandb.log({'bs_gacc':bs_gacc, 'bs_giou':bs_giou, 'bs_miou':bs_miou})

            if args.visualize and batch_idx // args.print_freq == 1:
                pos_gt, angle_gt, width_gt = post_process_output(pos_img, cos_img, sin_img, width_img)
                draw_img_grasp(img[0], refgrasps[0], refmask[0], [pos_gt[0], angle_gt[0], width_gt[0]], f'train/train_{epoch}_gt')

                pos_pre, angle_pre, width_pre = post_process_output(pos, cos, sin, width)

                pred_mask = np.array(pred_seg.detach().cpu().numpy() > 0.35)

                pre_grasp = detect_grasps(pos_pre[0], angle_pre[0], width_pre[0])[0]

                draw_img_grasp(img[0], pre_grasp, pred_mask[0], [pos_pre[0], angle_pre[0], width_pre[0]], f'train/train_{epoch}_pre')


        del pred_seg, pos, cos, sin, width
        del img, word_embeddings, word_attention_mask, pos_img, cos_img, sin_img, width_img, refgrasps, refbbox, refmask, idx
        del loss
    return metric_logger

@torch.no_grad()
def validate(args, network, device, val_data, epoch):
    network.eval()

    
    logger.info("validating ...")
    MIoU_list = []
    loss_list = []
    total_giou = 0
    total_cor_num = 0
    total = 0

    for img, word_embeddings, word_attention_mask, pos_img, cos_img, sin_img, width_img, refgrasps, refbbox, refmask, kg_word_embedding, idx in val_data:
        img = img.to(device)
        word_embeddings = word_embeddings.to(device)
        word_attention_mask = word_attention_mask.to(device)
        pos_img = pos_img.to(device)
        cos_img = cos_img.to(device)
        sin_img = sin_img.to(device)
        width_img = width_img.to(device)
        refmask = refmask.to(device)
        kg_word_embedding = kg_word_embedding.to(device)


        pred_seg, pos, cos, sin, width = network(img, word_embeddings, kg_word_embedding)

        loss = compute_loss(pred_seg, refmask, [pos, cos, sin, width], [pos_img, cos_img, sin_img, width_img])
        loss_list.append(loss.item())
        
        _MIoU_list, I_list, U_list = caculate_seg_iou(pred_seg, refmask)
        MIoU_list += _MIoU_list
        flag, maxiou_list = count_grasp_correct([pos, cos, sin, width], refgrasps)
        total_cor_num += flag
        total += len(refgrasps)

        total_giou += np.array(maxiou_list).sum()

        if args.visualize:
            for i, maxiou in range(maxiou_list):
                if maxiou >= 0.65:
                    pos_gt, angle_gt, width_gt = post_process_output(pos_img, cos_img, sin_img, width_img)

                    draw_img_grasp(img[i], refgrasps[i], refmask[i], [pos_gt[i], angle_gt[i], width_gt[i]], f'{epoch}_valid_{idx[i]}_gt')

                    pos_pre, angle_pre, width_pre = post_process_output(pos, cos, sin, width)

                    pred_mask = np.array(pred_seg.detach().cpu().numpy() > 0.35)

                    pre_grasp = detect_grasps(pos_pre[i], angle_pre[i], width_pre[i])[0]

                    draw_img_grasp(img[i], pre_grasp, pred_mask[i], [pos_pre[i], angle_pre[i], width_pre[i]], f'{epoch}_valid_{idx[i]}_pre_{maxiou:.3f}')
    
    miou = np.mean(np.array(MIoU_list))
    mloss = np.mean(np.array(loss_list))
    giou = total_cor_num / total
    gmiou = total_giou/total
    wandb.log({'miou':miou, 'giou':giou, 'gmiou':gmiou, 'epoch': epoch})
    logger.info("MIoU: {:.4f}, GraspAcc: {:.4f}, gmiou: {:.4f}, total grasp num: {}, Mloss: {:.4f}".format(miou, giou, gmiou, total_cor_num, mloss))
    return miou, giou



def run():
    utils_.setup_seed(1024)
    args = parse_args()
    model_name = args.model_name
    wandb.init(config=args,
                project="KGgrasp",
                mode='offline',
                dir="wandbDir",
                reinit=True)
    

    if args.log_name == '':
        log_name = 'train'
    else:
        log_name = args.log_name
    logger.add(f'terminal_log/{args.model_name}/{log_name}.log')

    device = torch.device(f"cuda:{args.device}")
    
    if model_name.lower() == 'kggrasp':
        from models.clipKGgrasp import KGgrasp
        MODEL = KGgrasp
        logger.info('Loading Network...')
        net = MODEL(args).to(device)
        logger.info(net)
        
        logger.info(f"Using knowledge to enhance expression? ({args.use_kg})")
        clip_params = []
        other_params = []

        for name, para in net.named_parameters():
            if para.requires_grad:
                if "encoder" in name and 'fusion' not in name:
                    clip_params += [para]
                else:
                    other_params += [para]

        params = [
        {"params": clip_params, "lr": args.clip_lr},
        {"params": other_params, "lr": args.lr, "weight_decay":args.weight_decay},
        ]

    elif model_name.lower() == 'cgsnet':
        from lib.cgsnet import CGSnet
        MODEL = CGSnet
        logger.info('Loading Network...')
        net = MODEL(args).to(device)
        logger.info(net)
        
        logger.info(f"Using knowledge to enhance expression? ({args.use_kg})")
        clip_params = []
        other_params = []

        for name, para in net.named_parameters():
            if para.requires_grad:
                if "encoder" in name and 'fusion' not in name:
                    clip_params += [para]
                else:
                    other_params += [para]

        params = [
        {"params": clip_params, "lr": args.clip_lr},
        {"params": other_params, "lr": args.lr, "weight_decay":args.weight_decay},
        ]

    elif model_name.lower() == 'cris':
        from compare_models.CRIS import CRIS
        MODEL = CRIS
        logger.info('Loading Network...')
        net = MODEL(args).to(device)
        logger.info(net)
        
        logger.info(f"Using knowledge to enhance expression? ({args.use_kg})")
        clip_params = []
        other_params = []

        for name, para in net.named_parameters():
            if para.requires_grad:
                if "encoder" in name and 'fusion' not in name:
                    clip_params += [para]
                else:
                    other_params += [para]

        params = [
        {"params": clip_params, "lr": args.clip_lr},
        {"params": other_params, "lr": args.lr, "weight_decay":args.weight_decay},
        ]
    elif model_name.lower() =='tfgrasp':
        from compare_models.tfGrasp import SwinTransformerSys
        MODEL = SwinTransformerSys
        logger.info('Loading Network...')
        net = MODEL(args).to(device)
        logger.info(net)
        
        logger.info(f"Using knowledge to enhance expression? ({args.use_kg})")

        other_params = []

        for name, para in net.named_parameters():
            if para.requires_grad:
                other_params += [para]

        params = [
        {"params": other_params, "lr": args.lr, "weight_decay":args.weight_decay},
        ]
    elif model_name.lower() == 'grcnn':
        from compare_models.GRConvNet import GRCNN
        MODEL = GRCNN
        logger.info('Loading Network...')
        net = MODEL(args).to(device)
        logger.info(net)
        
        logger.info(f"Using knowledge to enhance expression? ({args.use_kg})")

        other_params = []

        for name, para in net.named_parameters():
            if para.requires_grad:
                other_params += [para]

        params = [
        {"params": other_params, "lr": args.lr, "weight_decay":args.weight_decay},
        ]

    elif  model_name.lower() == 'ggcnn2':
        from compare_models.ggcnn2 import GGCNN2
        MODEL = GGCNN2
        logger.info('Loading Network...')
        net = MODEL(args).to(device)
        logger.info(net)
        
        logger.info(f"Using knowledge to enhance expression? ({args.use_kg})")

        other_params = []

        for name, para in net.named_parameters():
            if para.requires_grad:
                other_params += [para]

        params = [
        {"params": other_params, "lr": args.lr, "weight_decay":args.weight_decay},
        ]
    else:
        raise NameError
        
    
    train_dataset = RefOCIDGrasp('./data', "train", tokenizer_name=args.tokenizer, include_rgb=args.use_rgb, 
                                 include_depth=args.use_depth, use_bbox=args.use_box, use_mask=args.use_mask, 
                                 output_size=args.img_size, max_tokens=args.txt_length, 
                                 weak_aug=args.weak_aug, weak_aug_epoch=args.weak_aug_epoch, 
                                 strong_aug=args.strong_aug, strong_aug_epoch=args.strong_aug_epoch, KG=args.use_kg)
    
    val_dataset = RefOCIDGrasp('./data', "val", tokenizer_name=args.tokenizer, include_rgb=args.use_rgb, 
                               include_depth=args.use_depth, use_bbox=args.use_box, use_mask=args.use_mask, 
                               output_size=args.img_size, max_tokens=args.txt_length, KG=args.use_kg)


    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=train_dataset.collate, drop_last=True)
    val_data = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=train_dataset.collate)



    if args.resume:
        logger.info(f"loading {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])


    logger.info(f"trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)/1024/1024:.2f}")
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay) 


    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)
    
    logger.info('Done')

    best_iou = 0.0
    save_flag = False

    if args.resume:
        print(f"resum from {args.resume}")
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
        best_iou = checkpoint['best_grasp_iou']

        print('resume completely!!!')
        
    else:
        resume_epoch = -999

    start_time = time.time()
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        logger.info('Begining Epoch {:02d}'.format(epoch))
        train_dataset.set_epoch(epoch)

        train_results = train_one_epoch(args, epoch, net, device, train_data, optimizer, lr_scheduler)
        miou, giou = validate(args, net, device, val_data, epoch)

        if giou >= best_iou:
            best_iou = giou
            save_flag = True

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        lastname = os.path.join(args.output_dir, f"{model_name}_last_model_onegpu.pth")
        dict_to_save = {'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'args': args,
                'best_grasp_iou': best_iou,
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch
                }
        torch.save(dict_to_save, lastname)
        
        if save_flag:
            save_flag = False
            bestname = os.path.join(args.output_dir, f"{model_name}_best_model_onegpu.pth")
            shutil.copyfile(lastname, bestname)

        if epoch >= 10:
            epochname = os.path.join(args.output_dir, f"{model_name}_{epoch}_model_onegpu.pth")
            shutil.copyfile(lastname, epochname)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Total time of training for {args.log_name}: {total_time_str}")
    wandb.finish()

if __name__ == '__main__':
    run()
