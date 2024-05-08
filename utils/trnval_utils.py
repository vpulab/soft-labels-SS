"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import os
import torch
import json
import time
import numpy as np
from config import cfg
from utils.misc import fast_hist, fmt_scale
from utils.misc import AverageMeter, eval_metrics
from utils.misc import metrics_per_image

from runx.logx import logx


def flip_tensor(x, dim):
    """
    Flip Tensor along a dimension
    """
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
                   else torch.arange(x.size(i) - 1, -1, -1).long()
                   for i in range(x.dim()))]


def resize_tensor(inputs, target_size):
    inputs = torch.nn.functional.interpolate(
        inputs, size=target_size, mode='bilinear',
        align_corners=cfg.MODEL.ALIGN_CORNERS)
    return inputs


def calc_err_mask(pred, gtruth, num_classes, classid):
    """
    calculate class-specific error masks
    """
    # Class-specific error mask
    class_mask = (gtruth >= 0) & (gtruth == classid)
    fp = (pred == classid) & ~class_mask & (gtruth != cfg.DATASET.IGNORE_LABEL)
    fn = (pred != classid) & class_mask
    err_mask = fp | fn

    return err_mask.astype(int)


def calc_err_mask_all(pred, gtruth, num_classes):
    """
    calculate class-agnostic error masks
    """
    # Class-specific error mask
    mask = (gtruth >= 0) & (gtruth != cfg.DATASET.IGNORE_LABEL)
    err_mask = mask & (pred != gtruth)

    return err_mask.astype(int)


def eval_minibatch(data, net, criterion, val_loss, calc_metrics, args, val_idx, dumper):
    """
    Evaluate a single minibatch of images.
     * calculate metrics
     * dump images

    There are two primary multi-scale inference types:
      1. 'MSCALE', or in-model multi-scale: where the multi-scale iteration loop is
         handled within the model itself (see networks/mscale.py -> nscale_forward())
      2. 'multi_scale_inference', where we use Averaging to combine scales
    """
    torch.cuda.empty_cache()

    scales = [args.default_scale]
    if args.multi_scale_inference:
        scales.extend([float(x) for x in args.extra_scales.split(',')])
        if val_idx == 0:
            logx.msg(f'Using multi-scale inference (AVGPOOL) with scales {scales}')

    # input    = torch.Size([1, 3, h, w])
    # gt_image = torch.Size([1, h, w])
    images, gt_image, img_names, scale_float = data
    assert len(images.size()) == 4 and len(gt_image.size()) == 3
    assert images.size()[2:] == gt_image.size()[1:]
    gt_image = gt_image.long()
    batch_pixel_size = images.size(0) * images.size(2) * images.size(3)

    input_size = [1024, 2048]# images.size(2), images.size(3) #[1024, 2048]#
    scale_mul = 1
    
    if args.do_flip:
        # By ending with flip=0, we insure that the images that are dumped
        # out correspond to the unflipped versions. A bit hacky.
        flips = [1, 0]
    else:
        flips = [0]

    with torch.no_grad():
        output = 0.0

        for flip in flips:
            for scale in scales:
                torch.cuda.empty_cache()
                #if input_size[0]*scale > 2048 or input_size[1]*scale> 2048:
                #    scale_mul = min(2048./input_size[0]/scale, 2048./input_size[1]/scale)
                #print([round(sz * scale*scale_mul) for sz in input_size])
                if flip == 1:
                    inputs = flip_tensor(images, 3)
                else:
                    inputs = images

                infer_size = [round(sz * scale*scale_mul) for sz in input_size]

                if scale*scale_mul != 1.0:
                    inputs = resize_tensor(inputs, infer_size)

                inputs = {'images': inputs, 'gts': gt_image}
                inputs = {k: v.cuda() for k, v in inputs.items()}

                # Expected Model outputs:
                #   required:
                #     'pred'  the network prediction, shape (1, 19, h, w)
                #
                #   optional:
                #     'pred_*' - multi-scale predictions from mscale model
                #     'attn_*' - multi-scale attentions from mscale model
                start = time.time()
                output_dict ={k: v.detach().cpu() for k,v in net(inputs).items()} 
                end = time.time()
                seconds = end - start
                #print("Time taken : {0} seconds".format(seconds))
                _pred = output_dict['pred'].detach().cpu()

                # save AVGPOOL style multi-scale output for visualizing
                if not cfg.MODEL.MSCALE:
                    scale_name = fmt_scale('pred', scale*scale_mul)
                    output_dict[scale_name] = _pred

                # resize tensor down to 1.0x scale in order to combine
                # with other scales of prediction
                if input_size != _pred.size()[2:]:
                    _pred = resize_tensor(_pred, input_size).detach().cpu()
                if flip == 1:
                    output = output + flip_tensor(_pred, 3).detach().cpu()
                else:
                    output = output + _pred

    output = output / len(scales) / len(flips)
    assert_msg = 'output_size {} gt_cuda size {}'
    gt_cuda = gt_image.cuda()
    assert_msg = assert_msg.format(
        output.size()[2:], gt_cuda.size()[1:])
    assert output.size()[2:] == gt_cuda.size()[1:], assert_msg
    assert output.size()[1] == cfg.DATASET.NUM_CLASSES, assert_msg

    # Update loss and scoring datastructure
    if calc_metrics:
        val_loss.update(criterion(output.cuda(), gt_image.cuda()).item(),
                        batch_pixel_size)

    output_data = torch.nn.functional.softmax(output.cuda(), dim=1).cpu().data
    max_probs, predictions = output_data.max(1)

    # Assemble assets to visualize
    assets = {}
    for item in output_dict:
        if 'attn_' in item:
            assets[item] = output_dict[item]
        if 'pred_' in item:
            smax = torch.nn.functional.softmax(output_dict[item].cuda(), dim=1)
            _, pred = smax.data.max(1)
            assets[item] = pred.cpu().numpy()
    """
    gt_image[(gt_image == 255)] = 19
    oh = torch.nn.functional.one_hot(gt_image.cuda(), num_classes=20)[:,:,:,:-1].squeeze().permute(2,0,1)
    aux = output_data.squeeze().cuda()
    assets["Loss"] = (-(oh*torch.log(aux)).sum(0))[None,:].permute(1,2,0).detach().cpu().numpy()
    assets["Loss"] = (assets["Loss"]*255/1.28).astype(np.uint8)
    
    data ={img_names[0]:{"out_ent": -(aux*torch.log(aux)).mean(1).detach().cpu().numpy(), "Loss": -(oh*torch.log(aux)).mean(1).detach().cpu().numpy(),
     "GT":(oh.sum(1)/(1024*2048)).detach().cpu().numpy(), "Out":(aux.sum(1)/(1024*2048)).detach().cpu().numpy()} }
    
    dumper.write(str(data))
    
    gt_image[(gt_image == 19)] = 255
    """
    predictions = predictions.numpy()
    assets['predictions'] = predictions
    assets['prob_mask'] = max_probs
    
    if calc_metrics:
        assets['err_mask'] = calc_err_mask_all(predictions,
                                               gt_image.numpy(),
                                               cfg.DATASET.NUM_CLASSES)

    _iou_acc = fast_hist(predictions.flatten(),
                         gt_image.numpy().flatten(),
                         cfg.DATASET.NUM_CLASSES)

    return assets, _iou_acc


def validate_topn(val_loader, net, criterion, optim, epoch, args):
    """
    Find worse case failures ...

    Only single GPU for now

    First pass = calculate TP, FP, FN pixels per image per class
      Take these stats and determine the top20 images to dump per class
    Second pass = dump all those selected images
    """
    assert args.bs_val == 1

    ######################################################################
    # First pass
    ######################################################################
    logx.msg('First pass')
    image_metrics = {}

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0

    for val_idx, data in enumerate(val_loader):

        # Run network
        assets, _iou_acc = \
            run_minibatch(data, net, criterion, val_loss, True, args, val_idx)

        # per-class metrics
        input_images, labels, img_names, _ = data

        fp, fn = metrics_per_image(_iou_acc)
        img_name = img_names[0]
        image_metrics[img_name] = (fp, fn)

        iou_acc += _iou_acc

        if val_idx % 20 == 0:
            logx.msg(f'validating[Iter: {val_idx + 1} / {len(val_loader)}]')

        if val_idx > 5 and args.test_mode:
            break

    eval_metrics(iou_acc, args, net, optim, val_loss, epoch)

    ######################################################################
    # Find top 20 worst failures from a pixel count perspective
    ######################################################################
    from collections import defaultdict
    worst_images = defaultdict(dict)
    class_to_images = defaultdict(dict)
    for classid in range(cfg.DATASET.NUM_CLASSES):
        tbl = {}
        for img_name in image_metrics.keys():
            fp, fn = image_metrics[img_name]
            fp = fp[classid]
            fn = fn[classid]
            tbl[img_name] = fp + fn
        worst = sorted(tbl, key=tbl.get, reverse=True)
        for img_name in worst[:args.dump_topn]:
            fail_pixels = tbl[img_name]
            worst_images[img_name][classid] = fail_pixels
            class_to_images[classid][img_name] = fail_pixels
    msg = str(worst_images)
    logx.msg(msg)

    # write out per-gpu jsons
    # barrier
    # make single table

    ######################################################################
    # 2nd pass
    ######################################################################
    logx.msg('Second pass')
    attn_map = None

    for val_idx, data in enumerate(val_loader):
        in_image, gt_image, img_names, _ = data

        # Only process images that were identified in first pass
        if not args.dump_topn_all and img_names[0] not in worst_images:
            continue

        with torch.no_grad():
            inputs = in_image.cuda()
            inputs = {'images': inputs, 'gts': gt_image}

            if cfg.MODEL.MSCALE:
                output, attn_map = net(inputs)
            else:
                output = net(inputs)

        output = torch.nn.functional.softmax(output, dim=1)
        prob_mask, predictions = output.data.max(1)
        predictions = predictions.cpu()

        # this has shape [bs, h, w]
        img_name = img_names[0]
        for classid in worst_images[img_name].keys():

            err_mask = calc_err_mask(predictions.numpy(),
                                     gt_image.numpy(),
                                     cfg.DATASET.NUM_CLASSES,
                                     classid)

            class_name = cfg.DATASET_INST.trainid_to_name[classid]
            error_pixels = worst_images[img_name][classid]
            logx.msg(f'{img_name} {class_name}: {error_pixels}')
            img_names = [img_name + f'_{class_name}']

            to_dump = {'gt_images': gt_image,
                       'input_images': in_image,
                       'predictions': predictions.numpy(),
                       'err_mask': err_mask,
                       'prob_mask': prob_mask,
                       'img_names': img_names}

            if attn_map is not None:
                to_dump['attn_maps'] = attn_map

            # FIXME!
            # do_dump_images([to_dump])

    html_fn = os.path.join(args.result_dir, 'best_images',
                           'topn_failures.html')
    from utils.results_page import ResultsPage
    ip = ResultsPage('topn failures', html_fn)
    for classid in class_to_images:
        class_name = cfg.DATASET_INST.trainid_to_name[classid]
        img_dict = class_to_images[classid]
        for img_name in sorted(img_dict, key=img_dict.get, reverse=True):
            fail_pixels = class_to_images[classid][img_name]
            img_cls = f'{img_name}_{class_name}'
            pred_fn = f'{img_cls}_prediction.png'
            gt_fn = f'{img_cls}_gt.png'
            inp_fn = f'{img_cls}_input.png'
            err_fn = f'{img_cls}_err_mask.png'
            prob_fn = f'{img_cls}_prob_mask.png'
            img_label_pairs = [(pred_fn, 'pred'),
                               (gt_fn, 'gt'),
                               (inp_fn, 'input'),
                               (err_fn, 'errors'),
                               (prob_fn, 'prob')]
            ip.add_table(img_label_pairs,
                         table_heading=f'{class_name}-{fail_pixels}')
    ip.write_page()

    return val_loss.avg
