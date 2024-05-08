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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from runx.logx import logx
from config2 import cfg
from loss.rmi import RMILoss
from cvxopt import matrix, spdiag, solvers

def get_loss(args):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    return: criterion, criterion_val
    """

    if args.rmi_loss:
        criterion = RMILoss(
            num_classes=cfg.DATASET.NUM_CLASSES,
            ignore_index=cfg.DATASET.IGNORE_LABEL).cuda()
    elif args.img_wt_loss:
        criterion = ImageBasedCrossEntropyLoss2d(
            classes=cfg.DATASET.NUM_CLASSES,
            ignore_index=cfg.DATASET.IGNORE_LABEL,
            upper_bound=args.wt_bound, fp16=args.fp16).cuda()
    elif args.jointwtborder:
        criterion = ImgWtLossSoftNLL(
            classes=cfg.DATASET.NUM_CLASSES,
            ignore_index=cfg.DATASET.IGNORE_LABEL,
            upper_bound=args.wt_bound).cuda()
    else:
        criterion = LOWLoss().cuda()
        #CrossEntropyLoss2d(ignore_index=cfg.DATASET.IGNORE_LABEL, cfg.Lambda, cfg.H).cuda()

    criterion_val = CrossEntropyLoss2d(
        weight=None, ignore_index=cfg.DATASET.IGNORE_LABEL).cuda()
    return criterion, criterion_val


class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, ignore_index=cfg.DATASET.IGNORE_LABEL,
                 norm=False, upper_bound=1.0, fp16=False):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logx.msg("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(weight, reduction='mean',
                                   ignore_index=ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING
        self.fp16 = fp16

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        bins = torch.histc(target, bins=self.num_classes, min=0.0,
                           max=self.num_classes)
        hist_norm = bins.float() / bins.sum()
        if self.norm:
            hist = ((bins != 0).float() * self.upper_bound *
                    (1 / hist_norm)) + 1.0
        else:
            hist = ((bins != 0).float() * self.upper_bound *
                    (1. - hist_norm)) + 1.0
        return hist

    def forward(self, inputs, targets, do_rmi=None):

        if self.batch_weights:
            weights = self.calculate_weights(targets)
            self.nll_loss.weight = weights

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(targets)
                if self.fp16:
                    weights = weights.half()
                self.nll_loss.weight = weights

            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0), dim=1),
                                  targets[i].unsqueeze(0),)
        return loss

def compute_weights(lossgrad, lamb, classes=1.0):

    device = lossgrad.get_device()
    lossgrad = lossgrad.data.cpu().numpy()

    # Compute Optimal sample Weights
    aux = -(lossgrad**2+lamb)
    sz = len(lossgrad)
    P = 2*matrix(lamb*np.identity(sz))
    q = matrix(aux.astype(np.double))
    A = spdiag(matrix(-1.0, (1,sz)))
    b = matrix(0.0, (sz,1))
    Aeq = matrix(classes.astype(np.double), (1,sz))
    beq = matrix(1.0*np.sum(classes))# sz)
    solvers.options['show_progress'] = False
    solvers.options['maxiters'] = 20
    solvers.options['abstol'] = 1e-4
    solvers.options['reltol'] = 1e-4
    solvers.options['feastol'] = 1e-4
    sol = solvers.qp(P, q, A, b, Aeq, beq)
    w = np.array(sol['x'])
    
    return torch.squeeze(torch.tensor(w, dtype=torch.float, device=device))

class LOWLoss(torch.nn.Module):
    def __init__(self, lamb=0.1, weight=None, ignore_index=cfg.DATASET.IGNORE_LABEL,
                 reduction='mean'):
        super(LOWLoss, self).__init__()
        self.lamb = 0.1#lamb # higher lamb means more smoothness -> weights closer to 1
        

    def forward(self, logits, targets, do_rmi=None):
        # Compute loss gradient norm
        class_pixels = targets.transpose(0,1).flatten(start_dim=1).sum(1).cpu().numpy()

        output_d = logits.detach()
        p = F.softmax(output_d.requires_grad_(True), dim=1)
        loss_d = -torch.mul(targets, torch.log(p)).transpose(0,1).flatten(start_dim=1).mean(1)
        loss_d.backward(torch.ones_like(loss_d))
        grad = output_d.grad.transpose(0,1).flatten(start_dim=1).detach()
        loss_d = loss_d.detach()
        lossgrad = torch.norm(grad, 2, 1)

        # Computed weighted loss
        p = F.softmax(logits, dim=1)
        loss = -torch.mul(targets, torch.log(p))
        try:
            weights = compute_weights(lossgrad, self.lamb,class_pixels)
            #print(weights)
            loss = torch.mul(loss, weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)).sum()/targets.sum()
            return loss, [weights.detach().cpu().numpy(), 1] 
        except Exception as e:
            print(e)
            loss = loss.sum()/targets.sum()
        return loss, [[0], 1] 
        

class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, ignore_index=cfg.DATASET.IGNORE_LABEL,
                 reduction='mean', l= 0.1, h=1.3):
        super(CrossEntropyLoss2d, self).__init__()
        logx.msg("Using Cross Entropy Loss")
        self.weight = weight
        print(cfg.DATASET.IGNORE_LABEL)
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction,
                                   ignore_index=ignore_index)
        self.filter = False
        self.l = l
        self.h = h

    def forward(self, inputs, targets, do_rmi=None):

        if targets.shape == inputs.shape:
            #filters = targets > 0 
            #targets = targets[filters]
            e = targets.clone()
            e[targets == 0] = 1  
            #entropy_gt = -torch.mul(e, torch.log(e)).sum(1)

            p = F.softmax(inputs, dim=1)#[filters]
            
            

            #loss = ((targets - p).square()/targets).sum(1)
            if self.filter:
                loss = torch.mul(targets, torch.log(e/p)).sum(1)#.sum(1)
                v = torch.tensor(1)
                v = ((loss < self.l)* (loss > 0)).int().detach()
                loss = loss * (1.*v)
                total = (targets.sum(1) > 0).int().detach()
                
                n_included = v.sum()
                
                return loss.sum()/n_included.sum() ,[n_included, total.sum()] 
            else:
                loss = torch.mul(targets, torch.log(e/p))
                return loss.sum()/targets.sum() ,[1, 1] 
            #per_class_loss =[[loss[i,j,:,:].clone().mean().detach().cpu().item() for j in range(19)] for i in range(len(loss))] 
            #loss_per_pixel = loss.sum(1)
            #quantiles = {"0": loss_per_pixel[entropy_gt==0].clone().mean().detach().cpu().item()}
            #quantiles.update({str(i): loss_per_pixel[(entropy_gt>i)*(entropy_gt<i+0.1)].clone().mean().detach().cpu().item() for i in [0.2, 0.3,0.5,0.55,0.6, 0.65,1,1.2,1.3]})
            #quantiles.update({"high entropy": loss_per_pixel[(entropy_gt>0.75)].clone().mean().detach().cpu().item()}) 
            #quantiles.update({"per class loss": per_class_loss})
            #final_loss = loss[entropy_gt==0].sum()/(entropy_gt==0).sum()
            #return loss.sum()/targets.sum(), quantiles#KL 
            #return - quantiles[1], [-i for i in  quantiles]
            #quantiles #,[-i for i in  quantiles] #CE 
            #p = torch.nn.functional.threshold(F.softmax(inputs, dim=1), 1e-10, 0)
            return torch.mul(targets, torch.log(targets/(p + 1e-30)+ 1e-30)).sum()/targets.sum() #+ self.nll_loss(F.log_softmax(inputs, dim=1), c)
        
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
    def start(self):
        self.filter = True
    def update(self):
        self.l *= self.h
        
    def restart(self):
        self.l = 0.1
        #self.filter = False
    def stop(self):
        self.filter = False

def customsoftmax(inp, multihotmask):
    """
    Custom Softmax
    """
    soft = F.softmax(inp)
    # This takes the mask * softmax ( sums it up hence summing up the classes
    # in border then takes of summed up version vs no summed version
    return torch.log(
        torch.max(soft,
                  (multihotmask * (soft * multihotmask).sum(1, keepdim=True)))
    )


class ImgWtLossSoftNLL(nn.Module):
    """
    Relax Loss
    """
    def __init__(self, classes, ignore_index=cfg.DATASET.IGNORE_LABEL, weights=None,
                 upper_bound=1.0, norm=False):
        super(ImgWtLossSoftNLL, self).__init__()
        self.weights = weights
        self.num_classes = classes
        self.ignore_index = ignore_index
        self.upper_bound = upper_bound
        self.norm = norm
        self.batch_weights = cfg.BATCH_WEIGHTING
        self.fp16 = False

    def calculate_weights(self, target):
        """
        Calculate weights of the classes based on training crop
        """
        if len(target.shape) == 3:
            hist = np.sum(target, axis=(1, 2)) * 1.0 / target.sum()
        else:
            hist = np.sum(target, axis=(0, 2, 3)) * 1.0 / target.sum()
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist[:-1]

    def custom_nll(self, inputs, target, class_weights, border_weights, mask):
        """
        NLL Relaxed Loss Implementation
        """
        if cfg.REDUCE_BORDER_EPOCH != -1 and \
           cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH:
            border_weights = 1 / border_weights
            target[target > 1] = 1

        wts = class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        if self.fp16:
            smax = customsoftmax(inputs, target[:, :-1, :, :].half())
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].half() *
                            wts * smax).sum(1)) * (1. - mask.half())
        else:
            smax = customsoftmax(inputs, target[:, :-1, :, :].float())
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].float() *
                            wts * smax).sum(1)) * (1. - mask.float())

        loss = loss_matrix.sum()

        # +1 to prevent division by 0
        loss = loss / (target.shape[0] * target.shape[2] * target.shape[3] -
                       mask.sum().item() + 1)
        return loss

    def forward(self, inputs, target):
        if self.fp16:
            weights = target[:, :-1, :, :].sum(1).half()
        else:
            weights = target[:, :-1, :, :].sum(1).float()
        ignore_mask = (weights == 0)
        weights[ignore_mask] = 1

        loss = 0
        target_cpu = target.data.cpu().numpy()

        if self.batch_weights:
            class_weights = self.calculate_weights(target_cpu)

        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                class_weights = self.calculate_weights(target_cpu[i])
            nll_loss = self.custom_nll(
                inputs[i].unsqueeze(0),
                target[i].unsqueeze(0),
                class_weights=torch.Tensor(class_weights).cuda(),
                border_weights=weights, mask=ignore_mask[i])
            loss = loss + nll_loss

        return loss


class MultiChannelBCEWithLogits(nn.Module):
    def __init__(self, size_average=False, reduce=True, use_beta=True, divide_by_N=True,
                 ignore_label=cfg.DATASET.IGNORE_LABEL,
                 sum_by_non_zero_weights=False):
        super(MultiChannelBCEWithLogits, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.use_beta = use_beta
        self.divide_by_N = divide_by_N
        self.ignore_label = ignore_label
        self._first_log = True
        self.sum_by_non_zero_weights = sum_by_non_zero_weights
        print('self.use_beta: ', use_beta)
        print('self.divide_by_N: ', divide_by_N)
        print('self.sum_by_non_zero_weights', self.sum_by_non_zero_weights)

    def _assertNoGrad(self, variable):
        assert not variable.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

    def forward_simple(self, input, target, return_raw_cost=False):
        self._assertNoGrad(target)
        batch_size = target.shape[0]

        # compute class agnostic beta
        # class agnostic counting
        class_agn_img = target.max(dim=1, keepdim=True)[0].view(batch_size, -1)

        count_pos = (class_agn_img == 1.0).sum(dim=1).float()
        count_neg = (class_agn_img == 0.0).sum(dim=1).float()

        count_all = count_pos + count_neg
        beta = count_neg / (count_all + 1e-8)
        beta = beta.unsqueeze(1)
        
        target = target.contiguous().view(batch_size, -1)
        input = input.view(batch_size, -1)

        mask = torch.ones_like(target).masked_fill(target == self.ignore_label, 0)
        target = target.masked_fill(target == self.ignore_label, 0)

        if not self.use_beta:
            weights = 1.
        else:
            weights = 1. - beta + (2. * beta - 1.) * target

        weights = weights * mask

        if return_raw_cost:
            cost = F.binary_cross_entropy_with_logits(input, target,
                                                      weight=weights,
                                                      size_average=False,
                                                      reduce=False)
            return cost

        if not self.sum_by_non_zero_weights:
            cost = F.binary_cross_entropy_with_logits(input, target,
                                                      weight=weights,
                                                      size_average=self.size_average,
                                                      reduce=self.reduce)
        else:
            cost = F.binary_cross_entropy_with_logits(input, target,
                                                      weight=weights,
                                                      size_average=False,
                                                      reduce=False)
            cost = cost.sum() / (torch.nonzero(weights).size(0) + 1e-8)

        if not self.divide_by_N:
            return cost
        else:
            return cost / batch_size

    def forward(self, inputs, targets, inputs_weights):
        #losses = []
        losses = 0.0
        for _input, _target, _weight in zip(inputs, targets, inputs_weights):
            if _weight != 0.0:
                loss = _weight * self.forward_simple(_input, _target)
                #losses.append(loss)
                losses += loss

        return losses


class EdgeWeightedCrossEntropyLoss2d(nn.Module):

    def __init__(self, classes, weight=None, size_average=False,
                 ignore_index=cfg.DATASET.IGNORE_LABEL,
                 norm=False, upper_bound=1.0):
        super(EdgeWeightedCrossEntropyLoss2d, self).__init__()
        logx.msg("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average,ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets, edges):

        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()
            
            out = self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),
                                               targets[i].unsqueeze(0))
            out = torch.mul(edges[i].unsqueeze(0), out)
            loss += out.sum() / (800 * 800)
        return loss



