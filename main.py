# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from datasets.coco_eval import CocoEvaluator


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    
    parser.add_argument('--inf', default=0, type=bool) #new
    parser.add_argument('--num_classes', default=None, type=int,
                        help='#classes in your dataset, which can override the value hard-coded in file models/detr.py') #new
    parser.add_argument('--finetune', default=None, type=int) #new
    parser.add_argument('--testDataInput', default=None, type=str) #new
    parser.add_argument('--testDataOutput', default=None, type=str) #new    
    
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser
#new
import torchvision.transforms as T

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def getLabels(img_ann, img_len):
  print("in get labels")
  img_range = list(range(0, img_len))
  labels = {i: [] for i in range(img_len)}
  label_names = ['N/A', 'Fire', "Smoke"]
  print(labels)
  for i in img_ann:
    # print(i["image_id"])
    # print(labels[i["image_id"]])
    # print(label_names["category_id"])
    temp_list = labels[i["image_id"]]
    temp_list.append(label_names[i["category_id"]])
    labels[i["image_id"]] = temp_list
    # print(labels)
    # labels[i["image_id"]].append(label_names[i["category_id"]])

  print(labels)
  return labels 

def inference():
    print("inference")
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print("starting")
    model2, criterion2, postprocessors2 = build_model(args)
    model2.to(device)
    print("model created")
  
    checkpoint = '/content/drive/MyDrive/checkpoints5/checkpoint.pth'
    # checkpoint = '/content/drive/MyDrive/finetune_deformable/r50_deformable_detr_single_scale-checkpoint.pth'
    # checkpoint = '/content/FineTuning-deformable-detr/my_check/checkpoint.pth'

    checkpoint = torch.load(checkpoint, map_location='cpu')
    # checkpoint = torch.load(args.resume, map_location='cpu')
    # print(checkpoint)
    model2.load_state_dict(checkpoint['model'], strict=False)
    print("model loaded")
    if torch.cuda.is_available():
        model2.cuda()
    model2.eval()
    # inference code 
    # find img url
  # model(img)

    from PIL import Image, ImageFont, ImageDraw, ImageEnhance
    import requests
    import io

  # url = 'https://drive.google.com/file/d/1tYPbfnm2Ybw1YTk3VAS24NqFRELJhTj5/view?usp=sharing'
  
  # url = 'data:image/jpeg;base64,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'

  # im = Image.open(requests.get(url, stream=True).raw)
  
    import json
  
    # Opening JSON file
    f = open('/content/FineTuning-deformable-detr/data/coco/annotations/instances_test.json')
      
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
      
    # Iterating through the json
    # list
    img_data = []
    img_ann = []
    for i in data['images']:
        img_data.append(i)

    for i in data['annotations']:
        img_ann.append(i)


    print("img_data", len(img_data),img_data[0])
    print("img_ann", len(img_ann),img_ann)

    #extracting ground truth data
    class_ids={1:'fire',2:'smoke'}
    img_names = []
    img_bbox = []
    for i in data['images']:
        img_names.append(i['file_name'])
    
    count=data['annotations'][0]['image_id']
    prev_data=data['annotations'][0]
    content = ""
    for i in data['annotations']:
        if count!=i['image_id']:
            print(count)
            print(i["image_id"])
            file_name=img_names[prev_data['image_id']].replace('.jpg','')
            sourcePath = "/content/mAP/input/ground-truth/"
            f1=open(sourcePath+file_name+".txt","w+")
            f1.write(content)
            f1.close()
            content=""
            count=i['image_id']
            print('done----------')
        content = content + class_ids[i["category_id"]] + " "
        str1=" "
        i["bbox"]=[str(j) for j in i["bbox"]]
        bbox_str = str1.join(i['bbox'])
        content = content + bbox_str +"\n"
        prev_data=i


    # Closing file
    f.close()
    all_results = []
    total_temp_labels = []
    # import glob
    # file_names = []
    # for f in glob.glob(args.testDataInput +  "/*"):
    #     file_names.append(f)

    img_labels = getLabels(img_ann, len(img_ann))
    sourcePath = "/content/FineTuning-deformable-detr/data/coco/test/"
    counter = 0
    # for f in file_names:
    for xyz in img_data:
        print(counter)
        if counter == 30:
          break
        # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        f = sourcePath +  xyz["file_name"]
        url = f
        # url = "/content/FineTuning-deformable-detr/data/coco/test/1499545303_-00960_jpg.rf.760e838e8d2d50d33534f9f5f809b936.jpg"
        # im = Image.open(requests.get(url, stream=True).raw) # url
        im = Image.open(url) # local

        img = transform(im).unsqueeze(0)
        img=img.cuda()

        outputs = model2(img)
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        # print(out_logits)
        print(out_logits.shape)

        import torch.nn as nn
        m = nn.Softmax(dim=1)
        # input = torch.randn(2, 3)
        prob = m(out_logits)
        # print(prob)
        print(prob.shape)
        tensor_shape = out_logits.shape
        # prob = out_logits.softmax(tensor_shape)
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        
        from util import box_ops
        
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        keep = scores[0] > 0.2
        boxes = boxes[0, keep]
        labels = labels[0, keep]

        # and from relative [0, 1] to absolute [0, height] coordinates
        im_h,im_w = im.size
        #print('im_h,im_w',im_h,im_w)
        target_sizes =torch.tensor([[im_w,im_h]])
        target_sizes =target_sizes.cuda()
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        # print(time.time()-t0)
        #plot_results
        # source_img = Image.open(requests.get(url, stream=True).raw).convert("RGBA")
        source_img = Image.open(url).convert("RGBA")
            
        #fnt = ImageFont.truetype("/content/content/Deformable-DETR/font/Aaargh.ttf", 18)
        draw = ImageDraw.Draw(source_img)
        #print ('label' , labels.tolist())
        label_list =  labels.tolist()
        #print("Boxes",boxes,boxes.tolist())
        i=0

        # label_names = ['N/A', 'table', 'smoke', 'non-smoke']
        colors = ['red', 'blue', 'green', 'yellow', 'black']
        label_names = [
            'N/A', 'fire', "smoke"
        ]
        print("THE LABELS:")
        temp_labels = []
        detection_result=''
        for xmin, ymin, xmax, ymax in boxes[0].tolist():
            # print(i, label_list[i])
            # draw.rectangle(((xmin, ymin), (xmax, ymax)), outline =colors[label_list[i]-1])
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=colors[(label_list[i]-1)%4] )
            # print('--------')
            # print('i= ', i)
            # print('label is = ', label_list[i]-1)
            # print(label_names[label_list[i]-1])
            if ymin-18 >=0 :
                ymin = ymin-18
            # draw.text((xmin, ymin), label_names[label_list[i]-1], anchor = 'md', fill=colors[label_list[i]-1])
            # draw.text((xmin, ymin), label_names[label_list[i]-1], anchor = 'md', fill= "white" )
            print(keep)
            print( label_names[label_list[i]] , scores[0][i].item())
            tempText = label_names[label_list[i]] + " " + str(scores[0][i].item())
            draw.text((xmin, ymin), tempText , anchor = 'md', fill= "black" )
            temp_labels.append(label_names[label_list[i]])
            detection_result=detection_result+tempText+' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax)+'\n'
            i+=1
        file_name=xyz['file_name'].replace('.jpg','')
        sourceP = "/content/mAP/input/detection-results/"
        f1=open(sourceP+file_name+".txt","w+")
        f1.write(detection_result)
        f1.close()
        import matplotlib.pyplot as plt
        import os

        
        print(temp_labels)
        print(img_labels[counter])  
        print(os.path.basename(f))      
        total_temp_labels.append(temp_labels)

        base_filename = os.path.basename(f)
        title, ext = os.path.splitext(base_filename)
        final_filepath = os.path.join(args.testDataOutput, title  + ext)

        source_img.save(final_filepath, "png")

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        all_results.append(results)
        counter += 1 
    accuracy = 0
    for i in range(len(total_temp_labels)):
      if(img_labels[i] == total_temp_labels[i]):
        print("yes")
        accuracy += 1
      else:
        print("real: ", img_labels[i] , "predicted: " ,total_temp_labels[i])

    print("accuracy is", (accuracy/len(total_temp_labels))*100)
    # print("Outputs",all_results)
    print("ending")
    # plt.imshow(source_img)

  
def inferenceNew():
    print("inference")
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print("starting")
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    checkpoint = '/content/FineTuning-deformable-detr/exps/r50_deformable_detr_single_scale/checkpoint.pth'
    checkpoint = torch.load(checkpoint, map_location='cpu')
    # checkpoint = torch.load(args.resume, map_location='cpu')
    # print(checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)
    print("model loaded")
    if torch.cuda.is_available():
        model.cuda()
    model.eval()


    dataset_test = build_dataset(image_set='test', args=args)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                  pin_memory=True)
    print(len(data_loader_test))

    base_ds = get_coco_api_from_dataset(dataset_test)
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    print(base_ds, iou_types)
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    
    for samples, targets in data_loader_test:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

def fineTune(checkpoint):
    del checkpoint["model"]["class_embed.0.weight"]
    del checkpoint["model"]["class_embed.0.bias"]
    del checkpoint["model"]["class_embed.1.weight"]
    del checkpoint["model"]["class_embed.1.bias"]
    del checkpoint["model"]["class_embed.2.weight"]
    del checkpoint["model"]["class_embed.2.bias"]
    del checkpoint["model"]["class_embed.3.weight"]
    del checkpoint["model"]["class_embed.3.bias"]
    del checkpoint["model"]["class_embed.4.weight"]
    del checkpoint["model"]["class_embed.4.bias"]
    del checkpoint["model"]["class_embed.5.weight"]
    del checkpoint["model"]["class_embed.5.bias"]
    return checkpoint

def main(args):
    if args.inf == 1:
      inference()
    else:
      utils.init_distributed_mode(args)
      print("git:\n  {}\n".format(utils.get_sha()))

      if args.frozen_weights is not None:
          assert args.masks, "Frozen training is meant for segmentation only"
      print(args)

      device = torch.device(args.device)

      # fix the seed for reproducibility
      seed = args.seed + utils.get_rank()
      torch.manual_seed(seed)
      np.random.seed(seed)
      random.seed(seed)
      #train
      model, criterion, postprocessors = build_model(args)
      model.to(device)

      model_without_ddp = model
      n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print('number of params:', n_parameters)

      dataset_train = build_dataset(image_set='train', args=args)
      dataset_val = build_dataset(image_set='val', args=args)

      if args.distributed:
          if args.cache_mode:
              sampler_train = samplers.NodeDistributedSampler(dataset_train)
              sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
          else:
              sampler_train = samplers.DistributedSampler(dataset_train)
              sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
      else:
          sampler_train = torch.utils.data.RandomSampler(dataset_train)
          sampler_val = torch.utils.data.SequentialSampler(dataset_val)

      batch_sampler_train = torch.utils.data.BatchSampler(
          sampler_train, args.batch_size, drop_last=True)

      data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                    pin_memory=True)
      data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                  pin_memory=True)

      # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
      def match_name_keywords(n, name_keywords):
          out = False
          for b in name_keywords:
              if b in n:
                  out = True
                  break
          return out

      for n, p in model_without_ddp.named_parameters():
          print(n)

      param_dicts = [
          {
              "params":
                  [p for n, p in model_without_ddp.named_parameters()
                  if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
              "lr": args.lr,
          },
          {
              "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
              "lr": args.lr_backbone,
          },
          {
              "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
              "lr": args.lr * args.lr_linear_proj_mult,
          }
      ]
      if args.sgd:
          optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                      weight_decay=args.weight_decay)
      else:
          optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                        weight_decay=args.weight_decay)
      lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

      if args.distributed:
          model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
          model_without_ddp = model.module

      if args.dataset_file == "coco_panoptic":
          # We also evaluate AP during panoptic training, on original coco DS
          coco_val = datasets.coco.build("val", args)
          base_ds = get_coco_api_from_dataset(coco_val)
      else:
          base_ds = get_coco_api_from_dataset(dataset_val)

      if args.frozen_weights is not None:
          checkpoint = torch.load(args.frozen_weights, map_location='cpu')
          model_without_ddp.detr.load_state_dict(checkpoint['model'])

      output_dir = Path(args.output_dir)
      if args.resume:
          if args.resume.startswith('https'):
              checkpoint = torch.hub.load_state_dict_from_url(
                  args.resume, map_location='cpu', check_hash=True)
          else:
              checkpoint = torch.load(args.resume, map_location='cpu') # imp
              if args.finetune==1:
                checkpoint=fineTune(checkpoint)
          missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
          unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
          if len(missing_keys) > 0:
              print('Missing Keys: {}'.format(missing_keys))
          if len(unexpected_keys) > 0:
              print('Unexpected Keys: {}'.format(unexpected_keys))
          if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint and args.finetune==None:
              import copy
              p_groups = copy.deepcopy(optimizer.param_groups)
              optimizer.load_state_dict(checkpoint['optimizer'])
              for pg, pg_old in zip(optimizer.param_groups, p_groups):
                  pg['lr'] = pg_old['lr']
                  pg['initial_lr'] = pg_old['initial_lr']
              print(optimizer.param_groups)
              lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
              # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
              args.override_resumed_lr_drop = True
              if args.override_resumed_lr_drop:
                  print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                  lr_scheduler.step_size = args.lr_drop
                  lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
              lr_scheduler.step(lr_scheduler.last_epoch)
              args.start_epoch = checkpoint['epoch'] + 1
          # check the resumed model
          if not args.eval:
              test_stats, coco_evaluator = evaluate(
                  model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
              )
      
      if args.eval:
          test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                                data_loader_val, base_ds, device, args.output_dir)
          if args.output_dir:
              utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
          return

      print("Start training")
      start_time = time.time()
      for epoch in range(0, 1):
          if args.distributed:
              sampler_train.set_epoch(epoch)
          train_stats = train_one_epoch(
              model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
          lr_scheduler.step()
          if args.output_dir:
              checkpoint_paths = [output_dir / 'checkpoint.pth']
              # extra checkpoint before LR drop and every 5 epochs
              if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                  checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
              for checkpoint_path in checkpoint_paths:
                  utils.save_on_master({
                      'model': model_without_ddp.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'lr_scheduler': lr_scheduler.state_dict(),
                      'epoch': epoch,
                      'args': args,
                  }, checkpoint_path)

          test_stats, coco_evaluator = evaluate(
              model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
          )

          log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                       **{f'test_{k}': v for k, v in test_stats.items()},
                       'epoch': epoch,
                       'n_parameters': n_parameters}

          if args.output_dir and utils.is_main_process():
              with (output_dir / "log.txt").open("a") as f:
                  f.write(json.dumps(log_stats) + "\n")

              # for evaluation logs
              if coco_evaluator is not None:
                  (output_dir / 'eval').mkdir(exist_ok=True)
                  if "bbox" in coco_evaluator.coco_eval:
                      filenames = ['latest.pth']
                      if epoch % 50 == 0:
                          filenames.append(f'{epoch:03}.pth')
                      for name in filenames:
                          torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                     output_dir / "eval" / name)
      # #--- saving model-----
      # print("saving Model--------------------------")
      # checkpoint_path = output_dir/f'whole_model.pth'
      # torch.save(model, checkpoint_path)
      # print("model saved--------------------")
      
      total_time = time.time() - start_time
      total_time_str = str(datetime.timedelta(seconds=int(total_time)))
      print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
