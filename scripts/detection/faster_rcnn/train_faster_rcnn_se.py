"""Train Faster-RCNN end to end."""
import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, "/home/caiqi/v-qcaii/research/cvpr2018/faster_rcnn_pascal")
sys.path.insert(0, "./")

import os

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import logging
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform, FasterRCNNDefaultSETransform, \
    FasterRCNNDefaultSEMultiTeacherTransform
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.metrics.accuracy import Accuracy
import itertools


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster-RCNN networks e2e.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--use_vgg', type=int, default=0,
                        help="whether use vgg as base network")
    parser.add_argument('--continue_conf_thres', type=int, default=0,
                        help="whether use continue conf threshold")
    parser.add_argument('--convert_params', type=int, default=1,
                        help="whether or not convert params, it is necessary to finetuing on se models")
    parser.add_argument('--distance_based_inside_graph', type=int, default=0,
                        help="whether use inside graph distance")
    parser.add_argument('--similarity_negative_weight', type=float, default=0,
                        help="weight for similarity negative weight, tastes better used with similarity_mask_with_equal_label")
    parser.add_argument('--dense_regions', type=int, default=0,
                        help="whether use dense regions for se loss")
    parser.add_argument('--post_softmax', type=int, default=0,
                        help="whether or not use post softmax")
    parser.add_argument('--similarity_mask_with_equal_label', type=int, default=0,
                        help="enforce similarity mask to be those that have same label")
    parser.add_argument('--linear_conf', type=int, default=0,
                        help="whether or not use linear increasing confidence")
    parser.add_argument('--conf_decay_epoch', type=int, default=20,
                        help="confidence decay epoch, default 20")
    parser.add_argument('--max_conf_thres', type=float, default=1.0,
                        help="max confidence threshold")
    parser.add_argument('--hybrid', type=int, default=0,
                        help="whether or not use hybrid the network")
    parser.add_argument('--early_stop', type=float, default=1.0,
                        help="stop when the ap is lower than ")
    parser.add_argument('--similarity_weight', type=float, default=0.0,
                        help="add weight to similarity matrix")
    parser.add_argument('--inside_graph_loss_weight', type=float, default=0.0,
                        help="inside graph loss")
    parser.add_argument('--similarity_feature', type=str, default="prob",
                        help="similarity feature to compute the relationship between regions, prob or visual")
    parser.add_argument('--similarity_metric', type=str, default="cosine",
                        help="similarity metric between regions, cosine, dot , l1 or l2")
    parser.add_argument('--similarity_distance_metric', type=str, default="l1",
                        help="similarity metric between regions l1 or l2")
    parser.add_argument('--output_feature', type=int, default=1,
                        help="whether output feature for training")
    parser.add_argument('--dataset', type=str, default='cityscape',
                        help='Training dataset. Now support voc and coco.')
    parser.add_argument('--merge_teacher_after_each_epoch', type=int, default=0,
                        help='whether merge multi teacher after each epoch')
    parser.add_argument('--nms_fusion', type=int, default=0,
                        help='whether or not use nms fusion to replace normal nms')
    parser.add_argument('--weight_decay_loss', type=float, default=0,
                        help='weight decay loss for diffirent studnets')
    parser.add_argument('--with_teacher', type=int, default=0,
                        help='use teacher model, i.e., use se to update the teacher model')
    parser.add_argument('--se_alpha', type=float, default=0.99,
                        help='alpha for se')
    parser.add_argument('--teacher_agree', type=int, default=0.99,
                        help='teacher must agree on the mask')
    parser.add_argument('--teacher_aug', type=int, default=0,
                        help='use augmentation for teacher data')
    parser.add_argument('--num_teacher', type=int, default=1,
                        help='number of students/teacher')
    parser.add_argument('--validate_resume', type=int, default=0,
                        help='validate resumed model')
    parser.add_argument('--use_se', type=int, default=1,
                        help='use se model')
    parser.add_argument('--augmentation', type=int, default=0,
                        help='use augmentation while training')
    parser.add_argument('--se_rpn_loss', type=float, default=0,
                        help='loss weight for rpn')
    parser.add_argument('--student_se_loss', type=int, default=0,
                        help='if true, compute the se loss with respect to both students, i.e., the teacher is only used to generate the rois')
    parser.add_argument('--se_rcnn_loss', type=float, default=1.0,
                        help='loss weight for rcnn')
    parser.add_argument('--base_conf_thres', type=float, default=0.97,
                        help='confidence threshold for se loss')
    parser.add_argument('--fixed_conf_thres', type=int, default=0,
                        help='use fixed confidence threshold')
    parser.add_argument('--pretrained_base', type=int, default=1,
                        help='use imagenet pretrained model')
    parser.add_argument('--allow_missing', type=int, default=0,
                        help='use imagenet pretrained model')
    parser.add_argument('--ignore_extra', type=int, default=0,
                        help='whether allow extra')
    parser.add_argument('--pretrained_base_path', type=str, default="~/.mxnet/models/",
                        help='imagenet pretrained model path')
    parser.add_argument('--train_patterns', type=str, default='.*dense|.*rpn|.*down(2|3|4)_conv|.*layers(2|3|4)_conv',
                        help='training patterns')
    parser.add_argument('--classes', type=str, default="person,rider,car,truck,bus,train,motorcycle,bicycle",
                        help='classes, seperate by,')
    parser.add_argument('--train_root', type=str, default='/home/caiqi/v-qcaii/research/cvpr2018/data/foggy_cityscape',
                        help='Training dataset. Now support voc and coco.')
    parser.add_argument('--val_root', type=str, default='/home/caiqi/v-qcaii/research/cvpr2018/data/foggy_cityscape',
                        help='Training dataset. Now support voc and coco.')
    parser.add_argument('--target_root', type=str, default='/home/caiqi/v-qcaii/research/cvpr2018/data/foggy_cityscape',
                        help='Training dataset. Now support voc and coco.')
    parser.add_argument('--train_split', type=str, default='cityscape_train.txt',
                        help='Training dataset. Now support voc and coco.')
    parser.add_argument('--val_split', type=str, default='cityscape_val_foggy.txt',
                        help='Training dataset. Now support voc and coco.')
    parser.add_argument('--target_split', type=str, default='cityscape_train_foggy.txt',
                        help='Training dataset. Now support voc and coco.')
    parser.add_argument('--min_dataset_size', type=int, default=-1,
                        help='Training dataset. Now support voc and coco.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=0, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=12,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str,
                        default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./faster_rcnn_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001 for voc single gpu training.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='8,10',
                        help='epoches at which learning rate decays. default is 14,20 for voc.')
    parser.add_argument('--lr-warmup', type=float, default=100,
                        help='warmup iterations to adjust learning rate, default is 0 for voc.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4 for voc')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='output/',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Print helpful debugging info once set.')

    args = parser.parse_args()
    if args.dataset == 'voc':
        args.epochs = int(args.epochs) if args.epochs else 20
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '14,20'
        args.lr = float(args.lr) if args.lr else 0.001
        args.lr_warmup = args.lr_warmup if args.lr_warmup else -1
        args.wd = float(args.wd) if args.wd else 5e-4
    elif args.dataset == 'coco':
        args.epochs = int(args.epochs) if args.epochs else 26
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '17,23'
        args.lr = float(args.lr) if args.lr else 0.00125
        args.lr_warmup = args.lr_warmup if args.lr_warmup else 8000
        args.wd = float(args.wd) if args.wd else 1e-4
        num_gpus = len(args.gpus.split(','))
        if num_gpus == 1:
            args.lr_warmup = -1
        else:
            args.lr *= num_gpus
            args.lr_warmup /= num_gpus
    return args


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')

    def update(self, labels, preds):
        # label: [rpn_label, rpn_weight]
        # preds: [rpn_cls_logits]
        rpn_label, rpn_weight = labels
        rpn_cls_logits = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_weight)

        # cls_logits (b, c, h, w) red_label (b, 1, h, w)
        # pred_label = mx.nd.argmax(rpn_cls_logits, axis=1, keepdims=True)
        pred_label = mx.nd.sigmoid(rpn_cls_logits) >= 0.5
        # label (b, 1, h, w)
        num_acc = mx.nd.sum((pred_label == rpn_label) * rpn_weight)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')

    def update(self, labels, preds):
        # label = [rpn_bbox_target, rpn_bbox_weight]
        # pred = [rpn_bbox_reg]
        rpn_bbox_target, rpn_bbox_weight = labels
        rpn_bbox_reg = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(rpn_bbox_weight * mx.nd.smooth_l1(rpn_bbox_reg - rpn_bbox_target, scalar=3))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')

    def update(self, labels, preds):
        # label = [rcnn_label]
        # pred = [rcnn_cls]
        rcnn_label = labels[0]
        rcnn_cls = preds[0]

        # calculate num_acc
        pred_label = mx.nd.argmax(rcnn_cls, axis=-1)
        num_acc = mx.nd.sum(pred_label == rcnn_label)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += rcnn_label.size


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')

    def update(self, labels, preds):
        # label = [rcnn_bbox_target, rcnn_bbox_weight]
        # pred = [rcnn_reg]
        rcnn_bbox_target, rcnn_bbox_weight = labels
        rcnn_bbox_reg = preds[0]

        # calculate num_inst
        num_inst = mx.nd.sum(rcnn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(rcnn_bbox_weight * mx.nd.smooth_l1(rcnn_bbox_reg - rcnn_bbox_target, scalar=1))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


class ScalarMetric(mx.metric.EvalMetric):
    def __init__(self, name):
        super(ScalarMetric, self).__init__(name)

    def update(self, labels, preds=None):
        self.sum_metric += labels[0].asscalar()
        self.num_inst += 1


class se_optimizer(object):
    def __init__(self, students, teacher, ctx, alpha=0.999):
        self.ctx_len = len(ctx)
        self.student_params = students.collect_params()
        self.teacher_params = teacher.collect_params()
        self.alpha = alpha

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for s_param, t_param in zip(self.student_params, self.teacher_params):
            for j in range(self.ctx_len):
                self.teacher_params[t_param]._data[j][:] = self.teacher_params[t_param]._data[j][
                                                           :] * self.alpha + one_minus_alpha * self.student_params[
                                                                                                   s_param]._data[j][:]


def average_weight(net1, net2):
    keys1 = net1.collect_params().keys()
    keys2 = net2.collect_params().keys()
    keys1 = list(keys1)
    keys2 = list(keys2)
    keys1 = sorted(keys1)
    keys2 = sorted(keys2)
    params1 = net1.collect_params()
    params2 = net2.collect_params()
    average_params = []
    for k1, k2 in zip(keys1, keys2):
        average_params.append((params1[k1]._data[0] + params2[k2]._data[0]) / 2)
    return average_params


def set_params(net, params):
    keys = net.collect_params().keys()
    keys = list(keys)
    keys = sorted(keys)
    params_to = net.collect_params()
    for k, key in enumerate(keys):
        for i, ctx in enumerate(params_to[key]._ctx_list):
            params_to[key]._data[i][:] = params[k].as_in_context(ctx)[:]


def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval')])
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        target_dataset = None
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == "cityscape":
        train_dataset = gdata.CityScapeDetection(root=args.train_root,
                                                 splits=args.train_split, min_dataset_size=args.min_dataset_size)
        val_dataset = gdata.CityScapeDetection(root=args.val_root,
                                               splits=args.val_split)
        if args.target_split != "":

            target_dataset = gdata.CityScapeDetection(root=args.target_root,
                                                      splits=args.target_split, min_dataset_size=args.min_dataset_size)
        else:
            target_dataset = None
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == "sim10k" or dataset.lower() == "kitti":
        train_dataset = gdata.SIM10kDetection(root=args.train_root,
                                              splits=args.train_split, min_dataset_size=args.min_dataset_size)
        val_dataset = gdata.SIM10kDetection(root=args.val_root,
                                            splits=args.val_split)
        if args.target_split != "":

            target_dataset = gdata.SIM10kDetection(root=args.target_root,
                                                   splits=args.target_split, min_dataset_size=args.min_dataset_size)
        else:
            target_dataset = None
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = gdata.COCODetection(splits='instances_train2017', use_crowd=False)
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(val_dataset, args.save_prefix + '_eval', cleanup=True)
        target_dataset = None
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, target_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, target_dataset, batch_size, num_workers, args):
    """Get dataloader."""
    train_bfn = batchify.Tuple(*[batchify.Append() for _ in range(5)])
    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform(
            FasterRCNNDefaultTrainTransform(net.short, net.max_size, net, augmentation=args.augmentation)),
        batch_size, True, batchify_fn=train_bfn, last_batch='rollover', num_workers=num_workers)

    if target_dataset is None:
        target_loader = None
    else:
        target_bfn = batchify.Tuple(*[batchify.Append() for _ in range(2)])
        target_loader = mx.gluon.data.DataLoader(
            target_dataset.transform(
                FasterRCNNDefaultSEMultiTeacherTransform(net.short, net.max_size, net, teacher_num=args.num_teacher,
                                                         teacher_aug=args.teacher_aug)),
            batch_size, True, batchify_fn=target_bfn, last_batch='rollover', num_workers=num_workers)

    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(FasterRCNNDefaultValTransform(net.short, net.max_size)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader, target_loader


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    net.hybridize(False)
    for batch in tqdm(val_data, total=len(val_data)):
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y, im_scale in zip(*batch):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(clipper(bboxes, x))
            # rescale to original resolution
            im_scale = im_scale.reshape((-1)).asscalar()
            det_bboxes[-1] *= im_scale
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_bboxes[-1] *= im_scale
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids, det_scores, gt_bboxes,
                                                                        gt_ids, gt_difficults):
            eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
    return eval_metric.get()


def get_lr_at_iter(alpha):
    return 1. / 3. * (1 - alpha) + alpha


def consistent_loss(image_feature, region_feature, scale=0.1):
    image_feature = nd.reshape(image_feature, shape=(0, -1))
    image_feature = nd.mean(image_feature, axis=1)
    distance = nd.square(region_feature - image_feature) * scale
    return distance


def train(student_list, teacher_list, train_data, val_data, target_data, eval_metric, ctx, args, se_opt_list):
    """Training pipeline"""
    trainer_list = []
    train_patterns = args.train_patterns

    for student, teacher in zip(student_list, teacher_list):
        student[0].collect_params().setattr('grad_req', 'null')
        student[0].collect_train_params(train_patterns).setattr('grad_req', 'write')
        teacher[0].collect_params().setattr('grad_req', 'null')
        teacher[0].collect_train_params(train_patterns).setattr('grad_req', 'write')

        for k, v in student[0].collect_params().items():
            logger.info("all params:" + str(k))
        for k, v in student[0].collect_train_params(train_patterns).items():
            logger.info("training params:" + str(k))

        trainer = gluon.Trainer(
            student[0].collect_train_params(train_patterns),  # fix batchnorm, fix first stage, etc...
            'sgd',
            {'learning_rate': args.lr,
             'wd': args.wd,
             'momentum': args.momentum,
             'clip_gradient': 5})
        trainer_list.append(trainer)

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
    lr_warmup = float(args.lr_warmup)  # avoid int division

    # TODO(zhreshold) losses?
    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1 / 9.)  # == smoothl1
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss()  # == smoothl1
    metrics = [mx.metric.Loss('RPN_Conf'),
               mx.metric.Loss('RPN_SmoothL1'),
               mx.metric.Loss('RCNN_CrossEntropy'),
               mx.metric.Loss('RCNN_SmoothL1'), ]

    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric, ScalarMetric(name="rpn_se_cnt"),
                ScalarMetric(name="rpn_se_loss"), ScalarMetric(name="rcnn_se_cnt"), ScalarMetric("rcnn_se_loss"),
                ScalarMetric("wd loss"), ScalarMetric("similarity loss"), ScalarMetric("inside graph loss")]

    logger.info(args)
    logger.info("net have classes:{}".format(student_list[0][0].classes))

    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    if args.validate_resume:
        map_name, mean_ap = validate(teacher_list[0], val_data, ctx, eval_metric)
        logger.info("validating resuming model")
        logger.info(map_name)
        logger.info(mean_ap)

    box_to_center = None
    box_decoder = None
    clipper = None

    if args.dense_regions:
        raise NotImplementedError

    best_map = [0]
    data_size = len(train_data)
    data_size = data_size * 1.0

    for epoch in range(args.start_epoch, args.epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer_list[0].learning_rate * lr_decay
            lr_steps.pop(0)
            for trainer in trainer_list:
                trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        for metric in metrics:
            metric.reset()
        tic = time.time()
        btic = time.time()
        for student in student_list:
            if args.hybrid:
                student.hybridize(static_alloc=True)
            else:
                student.hybridize(False)
        for teacher in teacher_list:
            if args.hybrid:
                teacher.hybridize(static_alloc=True)
            else:
                teacher.hybridize(False)
        base_lr = trainer_list[0].learning_rate

        conf_thres = 0
        if target_data is None:
            target_data_size = len(train_data)
            target_data = []
            for _ in range(target_data_size):
                target_data.append([nd.zeros((1,)), nd.zeros((1,))])
        if len(train_data) != len(target_data):
            logger.info(
                "train data has: {} items but target data has: {} items, it would be better if they have the same number".format(
                    len(train_data), len(target_data)))
        logger.info("training data has: {} items".format(min(len(train_data), len(target_data))))
        for i, (batch, target_batch) in tqdm(enumerate(zip(train_data, target_data)),
                                             total=min(len(train_data), len(target_data))):
            if epoch == 0 and i <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup)
                if new_lr != trainer_list[0].learning_rate:
                    if i % args.log_interval == 0:
                        logger.info('[Epoch 0 Iteration {}] Set learning rate to {}'.format(i, new_lr))
                    for trainer in trainer_list:
                        trainer.set_learning_rate(new_lr)
            batch = split_and_load(batch, ctx_list=ctx)
            if args.use_se:
                target_batch_new = []
                for target_batch_item1 in target_batch:
                    tmp_batch = []
                    for tmp_data in target_batch_item1:
                        tmp_data_reshape = nd.reshape(tmp_data, shape=(-1, 3, 0, 0))
                        tmp_batch.append(tmp_data_reshape)
                    target_batch_new.append(tmp_batch)
                target_batch = target_batch_new
                target_batch = split_and_load(target_batch, ctx_list=ctx)
            else:
                target_batch = [[nd.zeros(shape=(1,), ctx=ctx[ctx_idx]) for ctx_idx in range(len(ctx))],
                                [nd.zeros(shape=(1,), ctx=ctx[ctx_idx]) for ctx_idx in range(len(ctx))]]

            batch_size = len(batch[0])
            losses = []
            metric_losses = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]

            with autograd.record():
                for data, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks, target_data_image_1, target_data_image_2 in zip(
                        *(batch + target_batch)):
                    gt_label = label[:, :, 4:5]
                    gt_box = label[:, :, :4]
                    mask_score = None

                    valid_index = 1.0
                    teacher_roi_list = None

                    if args.num_teacher > 1 and args.use_se:
                        raise NotImplementedError

                    idx_list = [k for k in range(args.num_teacher)]
                    for idx, student, teacher in zip(idx_list, student_list, teacher_list):
                        if args.student_se_loss:
                            raise NotImplementedError
                        cls_pred, box_pred, roi, samples, matches, rpn_score, rpn_box, anchors, _, _ = student[0](
                            data, gt_box, None)
                        if args.use_se:
                            target_data_image_t = target_data_image_1[idx:idx + 1, :, :, :]
                            cls_pred_1, _, roi_1, _, _, rpn_score_1, rpn_box_1, anchor_for_se, _, top_feature_1 = \
                                teacher[0](
                                    target_data_image_t,
                                    None,
                                    teacher_roi_list)
                            target_data_image_t = target_data_image_2[idx:idx + 1, :, :, :]
                            if teacher_roi_list is None:
                                if args.dense_regions:
                                    raise NotImplementedError
                                else:
                                    teacher_roi_list = roi_1
                            if args.dense_regions:
                                raise NotImplementedError
                            cls_pred_2, _, roi_2, _, _, rpn_score_2, rpn_box_2, _, _, top_feature_2 = student[0](
                                target_data_image_t,
                                None,
                                teacher_roi_list)
                            # cls_pred_2 = nd.stop_gradient(cls_pred_2)
                            cls_pred_1_softmax = nd.softmax(cls_pred_1, axis=-1)
                            cls_pred_2_softmax = nd.softmax(cls_pred_2, axis=-1)

                            if not args.teacher_agree:
                                max_score = nd.max(cls_pred_1_softmax[:, :, 1:], axis=-1) * valid_index
                                if args.fixed_conf_thres:
                                    conf_thres = args.base_conf_thres
                                else:
                                    if args.continue_conf_thres:
                                        conf_thres = 1. + epoch + 1.0 * i / data_size
                                    else:
                                        conf_thres = 1. + epoch

                                    if args.linear_conf:
                                        conf_thres = (conf_thres - 1) / args.conf_decay_epoch * (
                                                1 - args.base_conf_thres) + args.base_conf_thres
                                    else:
                                        conf_thres = np.log(conf_thres) / np.log(args.conf_decay_epoch) * (
                                                1 - args.base_conf_thres) + args.base_conf_thres
                                mask_score = max_score > conf_thres

                            conf_thres = min(conf_thres, args.max_conf_thres)

                            if args.similarity_weight > 0:
                                max_label = nd.argmax(cls_pred_1_softmax[:, :, :], axis=-1) * valid_index
                                max_label_info = nd.broadcast_equal(nd.transpose(max_label), max_label)

                                if args.similarity_feature == "prob":
                                    raise NotImplementedError
                                elif args.similarity_feature == "visual":
                                    sim_feature_1 = top_feature_1[:, :, 0, 0]
                                    sim_feature_2 = top_feature_2[:, :, 0, 0]
                                else:
                                    assert 1 == 0, "unsupport similarity feature {}".format(args.similarity_feature)
                                if args.similarity_metric == "cosine":
                                    sim_feature_1_t = nd.L2Normalization(sim_feature_1, mode="instance")
                                    sim_feature_2_t = nd.L2Normalization(sim_feature_2, mode="instance")
                                    similarity_1 = nd.dot(sim_feature_1_t, nd.transpose(sim_feature_1_t))
                                    similarity_2 = nd.dot(sim_feature_2_t, nd.transpose(sim_feature_2_t))
                                else:
                                    assert 1 == 0, "unsupport similarity metric {}".format(args.similarity_metric)
                                if args.post_softmax:
                                    similarity_1 = nd.softmax(similarity_1, axis=-1)
                                    similarity_2 = nd.softmax(similarity_2, axis=-1)

                                if args.similarity_distance_metric == "l1":
                                    raise NotImplementedError
                                elif args.similarity_distance_metric == "l2":
                                    similarity_diff = nd.square(similarity_1 - similarity_2)
                                    if args.similarity_metric == "l0" or args.similarity_metric == "l1" or args.similarity_metric == "l2" or args.similarity_metric == "id":
                                        similarity_diff = nd.sum(similarity_diff, axis=2)
                                else:
                                    assert 1 == 0, "unsupport similarity distance metric {}".format(
                                        args.similarity_distance_metric)

                                similarity_mask = nd.dot(nd.transpose(mask_score), mask_score)

                                if args.similarity_mask_with_equal_label:
                                    similarity_mask = similarity_mask * max_label_info + args.similarity_negative_weight * similarity_mask

                                if args.distance_based_inside_graph:
                                    sim_feature_1 = top_feature_1[:, :, 0, 0]
                                    sim_feature_2 = top_feature_2[:, :, 0, 0]
                                    sim_feature_1_1 = nd.expand_dims(sim_feature_1, axis=1)
                                    sim_feature_1_2 = nd.expand_dims(sim_feature_1, axis=0)
                                    similarity_1 = nd.square(nd.broadcast_minus(sim_feature_1_1, sim_feature_1_2))
                                    sim_feature_2_1 = nd.expand_dims(sim_feature_2, axis=1)
                                    sim_feature_2_2 = nd.expand_dims(sim_feature_2, axis=0)
                                    similarity_2 = nd.square(nd.broadcast_minus(sim_feature_2_1, sim_feature_2_2))
                                    similarity_1_summed = nd.mean(similarity_1, axis=2)
                                    similarity_2_summed = nd.mean(similarity_2, axis=2)
                                    inside_graph_loss_1 = similarity_1_summed * max_label_info + (
                                            1 - max_label_info) * nd.relu(5 - similarity_1_summed)
                                    inside_graph_loss_2 = similarity_2_summed * max_label_info + (
                                            1 - max_label_info) * nd.relu(5 - similarity_2_summed)

                                    inside_graph_loss_1 = nd.sum(inside_graph_loss_1 * similarity_mask) / (
                                            nd.sum(similarity_mask) + 1.)
                                    inside_graph_loss_2 = nd.sum(inside_graph_loss_2 * similarity_mask) / (
                                            nd.sum(similarity_mask) + 1.)
                                else:
                                    inside_graph_loss_1 = nd.sum(
                                        nd.square(similarity_1 - max_label_info) * similarity_mask) / (
                                                                  nd.sum(similarity_mask) + 1.)
                                    inside_graph_loss_2 = nd.sum(
                                        nd.square(similarity_2 - max_label_info) * similarity_mask) / (
                                                                  nd.sum(similarity_mask) + 1.)

                                inside_graph_loss = inside_graph_loss_1 + inside_graph_loss_2
                                inside_graph_loss = inside_graph_loss * args.inside_graph_loss_weight
                                similarity_masked = similarity_diff * similarity_mask
                                similarity_loss = nd.sum(similarity_masked) / (
                                        nd.sum(similarity_mask) + 1.) * args.similarity_weight
                                add_losses[9].append([[similarity_loss], [nd.zeros_like(similarity_loss)]])
                                add_losses[10].append([[inside_graph_loss], [nd.zeros_like(inside_graph_loss)]])

                            else:
                                similarity_loss = 0.0
                                inside_graph_loss = 0.0
                                add_losses[9].append([[nd.zeros((1,), ctx=ctx[idx])], [nd.zeros((1,), ctx=ctx[idx])]])
                                add_losses[10].append([[nd.zeros((1,), ctx=ctx[idx])], [nd.zeros((1,), ctx=ctx[idx])]])

                            loss_distance = nd.sum(nd.square(cls_pred_1_softmax - cls_pred_2_softmax), axis=-1)
                            se_rcnn_loss = mask_score * loss_distance

                            se_rcnn_cnt = nd.sum(mask_score)
                            se_rcnn_loss = se_rcnn_loss * args.se_rcnn_loss
                            se_rcnn_loss = nd.sum(se_rcnn_loss) / (se_rcnn_cnt + 1.0)

                            rpn_score_1_sigmoid = nd.sigmoid(rpn_score_1)
                            rpn_score_2_sigmoid = nd.sigmoid(rpn_score_2)

                            rpn_mask = rpn_score_1_sigmoid > conf_thres
                            loss_distance_rpn = nd.sum(nd.square(rpn_score_1_sigmoid - rpn_score_2_sigmoid), axis=-1,
                                                       keepdims=True)
                            se_rpn_loss = rpn_mask * loss_distance_rpn
                            se_rpn_cnt = nd.sum(rpn_mask)
                            se_rpn_loss = se_rpn_loss * args.se_rpn_loss
                            se_rpn_loss = nd.sum(se_rpn_loss) / (se_rpn_cnt + 1.0)
                            se_all_loss = se_rpn_loss + se_rcnn_loss + similarity_loss + inside_graph_loss

                            add_losses[4].append([[se_rpn_cnt], [nd.zeros_like(se_rpn_cnt)]])
                            add_losses[5].append([[se_rpn_loss], [nd.zeros_like(se_rpn_loss)]])
                            add_losses[6].append([[se_rcnn_cnt], [nd.zeros_like(se_rcnn_cnt)]])
                            add_losses[7].append([[se_rcnn_loss], [nd.zeros_like(se_rcnn_loss)]])
                        else:
                            se_all_loss = None
                            add_losses[4].append([[nd.zeros((1,), ctx=ctx[idx])], [nd.zeros((1,), ctx=ctx[idx])]])
                            add_losses[5].append([[nd.zeros((1,), ctx=ctx[idx])], [nd.zeros((1,), ctx=ctx[idx])]])
                            add_losses[6].append([[nd.zeros((1,), ctx=ctx[idx])], [nd.zeros((1,), ctx=ctx[idx])]])
                            add_losses[7].append([[nd.zeros((1,), ctx=ctx[idx])], [nd.zeros((1,), ctx=ctx[idx])]])
                            add_losses[9].append([[nd.zeros((1,), ctx=ctx[idx])], [nd.zeros((1,), ctx=ctx[idx])]])
                            add_losses[10].append([[nd.zeros((1,), ctx=ctx[idx])], [nd.zeros((1,), ctx=ctx[idx])]])

                        # losses of rpn
                        rpn_score = rpn_score.squeeze(axis=-1)
                        num_rpn_pos = (rpn_cls_targets >= 0).sum()
                        rpn_loss1 = rpn_cls_loss(rpn_score, rpn_cls_targets,
                                                 rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos
                        rpn_loss2 = rpn_box_loss(rpn_box, rpn_box_targets, rpn_box_masks) * rpn_box.size / num_rpn_pos
                        # rpn overall loss, use sum rather than average
                        rpn_loss = rpn_loss1 + rpn_loss2
                        # generate targets for rcnn
                        cls_targets, box_targets, box_masks = teacher[0].target_generator(roi, samples, matches,
                                                                                          gt_label,
                                                                                          gt_box)
                        # losses of rcnn
                        num_rcnn_pos = (cls_targets >= 0).sum()
                        rcnn_loss1 = rcnn_cls_loss(cls_pred, cls_targets, cls_targets >= 0) * cls_targets.size / \
                                     cls_targets.shape[0] / num_rcnn_pos
                        rcnn_loss2 = rcnn_box_loss(box_pred, box_targets, box_masks) * box_pred.size / box_pred.shape[
                            0] / num_rcnn_pos
                        rcnn_loss = rcnn_loss1 + rcnn_loss2
                        # overall losses
                        if args.use_se:
                            losses.append(rpn_loss.sum() + rcnn_loss.sum() + se_all_loss.sum())
                        else:
                            losses.append(rpn_loss.sum() + rcnn_loss.sum())
                        # losses.append(rpn_loss.sum() + rcnn_loss.sum())
                        metric_losses[0].append(rpn_loss1.sum())
                        metric_losses[1].append(rpn_loss2.sum())
                        metric_losses[2].append(rcnn_loss1.sum())
                        metric_losses[3].append(rcnn_loss2.sum())

                        add_losses[0].append([[rpn_cls_targets, rpn_cls_targets >= 0], [rpn_score]])
                        add_losses[1].append([[rpn_box_targets, rpn_box_masks], [rpn_box]])
                        add_losses[2].append([[cls_targets], [cls_pred]])
                        add_losses[3].append([[box_targets, box_masks], [box_pred]])
                if args.weight_decay_loss > 0:
                    raise NotImplementedError
                else:
                    for jj in range(len(ctx)):
                        add_losses[8].append([[nd.zeros((1,), ctx=ctx[jj])], [nd.zeros((1,), ctx=ctx[jj])]])

                autograd.backward(losses)
                for metric, record in zip(metrics, metric_losses):
                    metric.update(0, record)
                for metric, records in zip(metrics2, add_losses):
                    for pred in records:
                        metric.update(pred[0], pred[1])

            for trainer in trainer_list:
                trainer.step(batch_size)
            for se_opt in se_opt_list:
                if se_opt is not None:
                    se_opt.step()
            # update metrics
            if args.log_interval and not (i + 1) % args.log_interval:
                msg = ','.join(['{}={:.5f}'.format(*metric.get()) for metric in metrics + metrics2])
                msg += ",{}={:.5f}".format("conf threshold", conf_thres)
                logger.info('[Epoch {}][Batch {}], Speed: {:.5f} samples/sec, {}'.format(
                    epoch, i, args.log_interval * batch_size / (time.time() - btic), msg))
                btic = time.time()

        msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
        logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
            epoch, (time.time() - tic), msg))

        if args.merge_teacher_after_each_epoch:
            raise NotImplementedError


        if not (epoch + 1) % args.val_interval:
            # consider reduce the frequency of validation to save time
            for tea_idx, teacher in enumerate(teacher_list):
                map_name, mean_ap = validate(teacher, val_data, ctx, eval_metric)
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                logger.info('[Epoch {}] Teacher {} Validation: \n{}'.format(epoch, tea_idx, val_msg))
                current_map = float(mean_ap[-1])
                best_map = max(best_map[0], current_map)
                best_map = [best_map]
                save_params(teacher_list[0], logger, best_map, current_map, epoch, args.save_interval,
                            args.save_prefix + "_teacher_{}_".format(tea_idx))
                if current_map < args.early_stop:
                    logger.error("early stop because ap has dropped to : {} and early stop is: {}".format(current_map,
                                                                                                          args.early_stop))
                    exit(-1)
        else:
            current_map = 0.
            for tea_idx, teacher in enumerate(teacher_list):
                save_params(teacher, logger, best_map, current_map, epoch, args.save_interval,
                            args.save_prefix + "_teacher_{}_".format(tea_idx))


if __name__ == '__main__':
    args = parse_args()

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)

    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    args.batch_size = len(ctx)  # 1 batch per device

    # network
    net_name = '_'.join(('faster_rcnn', args.network, "custom"))
    args.save_prefix += net_name

    student_list = []
    teacher_list = []
    se_opt_list = []
    classes = args.classes.strip().split(",")
    classes = tuple(classes)
    if args.with_teacher and (not args.use_se):
        logger.info("if with teacher must use se")
        raise Exception

    for m in range(args.num_teacher):
        student = gluon.nn.HybridSequential(prefix="student_{}".format(m))
        with student.name_scope():
            net = get_model(net_name, pretrained_base=args.pretrained_base, classes=classes, additional_output=False,
                            root=args.pretrained_base_path, use_vgg=args.use_vgg, output_feature=args.output_feature)
            student.add(net)
        if args.with_teacher:
            teacher = gluon.nn.HybridSequential(prefix="teacher_{}".format(m))
            with teacher.name_scope():
                net = get_model(net_name, pretrained_base=args.pretrained_base, classes=classes,
                                additional_output=False,
                                root=args.pretrained_base_path, use_vgg=args.use_vgg,
                                output_feature=args.output_feature)
                teacher.add(net)
        else:
            teacher = student

        # test resume
        if args.resume.strip():
            resumed_params = mx.nd.load(args.resume.strip())
            first_key = list(resumed_params.keys())[0]
            if first_key.startswith("0.") and args.convert_params:
                logger.info("converting params from {} to {}".format(args.resume, args.resume + "convert"))
                new_params = {}
                for k, v in resumed_params.items():
                    new_params[k[k.find(".") + 1:]] = v
                mx.nd.save(args.resume.strip() + ".convert", new_params)
                args.resume = args.resume.strip() + ".convert"
            try:
                student[0].load_parameters(args.resume.strip(), allow_missing=args.allow_missing,
                                           ignore_extra=args.ignore_extra)
                teacher[0].load_parameters(args.resume.strip(), allow_missing=args.allow_missing,
                                           ignore_extra=args.ignore_extra)
            except Exception as e:
                print("meeting exception in resume: {}".format(str(e)))
                student.load_parameters(args.resume.strip(), allow_missing=args.allow_missing,
                                        ignore_extra=args.ignore_extra)
                if not args.with_teacher:
                    teacher.load_parameters(args.resume.strip(), allow_missing=args.allow_missing,
                                            ignore_extra=args.ignore_extra)
        else:
            for key, param in student.collect_params().items():
                if param._data is not None:
                    continue
                logger.info("initializing students {}".format(key))
                param.initialize()

            if args.with_teacher:
                for key, param in teacher.collect_params().items():
                    if param._data is not None:
                        continue
                    logger.info("initializing teacher{}".format(key))
                    param.initialize()
        if args.use_se and args.with_teacher:
            se_opt = se_optimizer(student, teacher, ctx=ctx, alpha=args.se_alpha)
        else:
            se_opt = None
        se_opt_list.append(se_opt)
        student.collect_params().reset_ctx(ctx)
        teacher.collect_params().reset_ctx(ctx)
        student_list.append(student)
        teacher_list.append(teacher)

    # training data
    train_dataset, val_dataset, target_dataset, eval_metric = get_dataset(args.dataset, args)
    train_data, val_data, target_data = get_dataloader(
        student_list[0][0], train_dataset, val_dataset, target_dataset, args.batch_size, args.num_workers,
        args=args)
    # training
    train(student_list, teacher_list, train_data, val_data, target_data, eval_metric, ctx, args,
          se_opt_list=se_opt_list)
