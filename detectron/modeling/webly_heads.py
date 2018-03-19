from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils
from detectron.modeling.ResNet import add_stage

from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags

from detectron.modeling.wsl_heads import add_wsl_outputs
from detectron.modeling.wsl_heads import add_cls_pred
from detectron.modeling.wsl_heads import add_center_loss
from detectron.modeling.wsl_heads import add_min_entropy_loss
from detectron.modeling.wsl_heads import add_cross_entropy_loss
from detectron.modeling.wsl_heads import add_csc_loss
from detectron.modeling.wsl_heads import get_loss_gradients_weighted
from detectron.modeling.wsl_heads import add_VGG16_roi_2fc_head
from detectron.modeling.wsl_heads import add_VGG16_roi_context_2fc_head
from detectron.modeling.wsl_heads import DropoutIfTraining

# ---------------------------------------------------------------------------- #
# Webly outputs and losses
# ---------------------------------------------------------------------------- #


def add_webly_outputs(model, blob_in, dim, prefix=''):
    add_wsl_outputs(model, blob_in[0], dim[0], prefix=prefix)

    # Box classification layer
    model.FC(
        blob_in[1],
        prefix + 'noisy_fc8c',
        dim[1],
        model.num_classes - 1,
        weight_init=('XavierFill', {}),
        # weight_init=('GaussianFill', {
        # 'std': 0.0005
        # }),
        bias_init=const_fill(0.0))
    model.FC(
        blob_in[1],
        prefix + 'noisy_fc8d',
        dim[1],
        model.num_classes - 1,
        # weight_init=('GaussianFill', {
        # 'std': 0.0005
        # }),
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0))

    model.net.Add([prefix + 'fc8c', prefix + 'noisy_fc8c'],
                  [prefix + 'fc8c_noise'])

    model.net.Add([prefix + 'fc8d', prefix + 'noisy_fc8d'],
                  [prefix + 'fc8d_noise'])

    model.Softmax(prefix + 'fc8c_noise', prefix + 'alpha_cls_noise', axis=1)
    model.Transpose(prefix + 'fc8d_noise',
                    prefix + 'fc8d_t_noise',
                    axes=(1, 0))
    model.Softmax(prefix + 'fc8d_t_noise',
                  prefix + 'alpha_det_t_noise',
                  axis=1)
    model.Transpose(prefix + 'alpha_det_t_noise',
                    prefix + 'alpha_det_noise',
                    axes=(1, 0))
    model.net.Mul([prefix + 'alpha_cls_noise', prefix + 'alpha_det_noise'],
                  prefix + 'rois_pred_noise')


def add_webly_outputs_shared(model, blob_in, dim, prefix=''):
    """Add RoI classification and bounding box regression output ops."""
    # Box classification layer
    model.FCShared(blob_in,
                   prefix + 'fc8c',
                   dim,
                   model.num_classes - 1,
                   weight='fc8c_w',
                   bias='fc8c_b')
    model.FCShared(blob_in,
                   prefix + 'fc8d',
                   dim,
                   model.num_classes - 1,
                   weight='fc8d_w',
                   bias='fc8d_b')

    model.Softmax(prefix + 'fc8c', prefix + 'alpha_cls', axis=1)
    model.Transpose(prefix + 'fc8d', prefix + 'fc8d_t', axes=(1, 0))
    model.Softmax(prefix + 'fc8d_t', prefix + 'alpha_det_t', axis=1)
    model.Transpose(prefix + 'alpha_det_t', prefix + 'alpha_det', axes=(1, 0))
    model.net.Mul([prefix + 'alpha_cls', prefix + 'alpha_det'],
                  prefix + 'rois_pred')


def add_cross_entropy_loss_self_weight(model, pred, label, loss, cpg=None):
    model.param_init_net.ConstantFill(['labels_oh'],
                                      'labels_oh_one',
                                      value=1.0)

    model.net.Sub(['labels_oh_one', 'labels_oh'], 'labels_oh_inv')

    model.net.Mul([pred, 'labels_oh'], pred + '_gt')
    model.net.Mul([pred, 'labels_oh_inv'], pred + '_gf')

    model.net.Sub(['labels_oh_inv', pred + '_gf'], pred + '_gf_inv')

    model.net.Add([pred + '_gt', pred + '_gf_inv'], pred + '_self_weight')

    add_cross_entropy_loss(model,
                           pred,
                           label,
                           loss,
                           weight=pred + '_self_weight',
                           cpg=cpg)


def add_webly_losses(model, prefix=''):
    add_cls_pred(prefix + 'rois_pred', prefix + 'cls_prob', model, prefix='')
    add_cls_pred(prefix + 'rois_pred_noise',
                 prefix + 'cls_prob_noise',
                 model,
                 prefix='')
    class_weight = None
    if cfg.WEBLY.ENTROPY:
        # add_entropy_weight(model, prefix + 'rois_pred', prefix + 'rois')
        add_spatial_entropy_weight(model, prefix + 'rois_pred',
                                   prefix + 'cls_prob', prefix + 'rois')
        class_weight = prefix + 'rois' + '_class_weight'
        class_weight_noise = prefix + 'rois' + '_class_weight_noise'

    cpg = None
    if cfg.WSL.CPG or cfg.WSL.CSC:
        cpg_args = {}
        cpg_args['tau'] = cfg.WSL.CPG_TAU
        cpg_args['max_iter'] = max(cfg.WSL.CPG_MAX_ITER, cfg.WSL.CSC_MAX_ITER)
        # cpg_args['debug_info'] = cfg.WSL.DEBUG
        cpg_args['cpg_net_name'] = model.net.Proto().name + '_cpg'
        cpg_args['pred_blob_name'] = cfg.WSL.CPG_PRE_BLOB
        cpg_args['data_blob_name'] = cfg.WSL.CPG_DATA_BLOB

        model.net.CPG(['labels_oh', prefix + 'cls_prob'], ['cpg_raw'],
                      **cpg_args)
        model.net.CPGScale(['cpg_raw', 'labels_oh', prefix + 'cls_prob'],
                           'cpg',
                           tau=cfg.WSL.CPG_TAU)
        cpg = 'cpg'

    if cfg.WSL.CSC:
        if not cfg.MODEL.MASK_ON or True:
            loss_gradients = add_csc_loss(model,
                                          'cpg',
                                          prefix + 'cls_prob',
                                          prefix + 'rois_pred',
                                          prefix + 'rois',
                                          loss_weight=1.0,
                                          prefix='')
        else:
            loss_gradients = {}
    else:
        loss_gradients = {}
        add_cross_entropy_loss(model,
                               prefix + 'cls_prob',
                               'labels_oh',
                               prefix + 'cross_entropy',
                               weight=class_weight,
                               cpg=cpg)
        loss_cls = model.net.AveragedLoss([prefix + 'cross_entropy'],
                                          [prefix + 'loss_cls'])

        loss_gradients_orig = blob_utils.get_loss_gradients(model, [loss_cls])
        loss_gradients.update(loss_gradients_orig)
        model.Accuracy([prefix + 'cls_prob', 'labels_int32'],
                       prefix + 'accuracy_cls')
        model.AddLosses([prefix + 'loss_cls'])
        model.AddMetrics(prefix + 'accuracy_cls')

        add_cross_entropy_loss(model,
                               prefix + 'cls_prob_noise',
                               'labels_oh',
                               prefix + 'cross_entropy_noise',
                               weight=class_weight_noise,
                               cpg=cpg)
        loss_cls = model.net.AveragedLoss([prefix + 'cross_entropy_noise'],
                                          [prefix + 'loss_cls_noise'])

        loss_gradients_noise = blob_utils.get_loss_gradients(model, [loss_cls])
        loss_gradients.update(loss_gradients_noise)
        model.Accuracy([prefix + 'cls_prob_noise', 'labels_int32'],
                       prefix + 'accuracy_cls_noise')
        model.AddLosses([prefix + 'loss_cls_noise'])
        model.AddMetrics(prefix + 'accuracy_cls_noise')

    if cfg.WSL.CENTER_LOSS:
        center_dim = 4096
        rois_pred = prefix + 'rois_pred'

        loss_gradients_center = add_center_loss('labels_oh', rois_pred,
                                                prefix + 'drop7', center_dim,
                                                model)
        loss_gradients.update(loss_gradients_center)

    if cfg.WSL.MIN_ENTROPY_LOSS:
        loss_gradients_ME = add_min_entropy_loss(model,
                                                 prefix + 'rois_pred',
                                                 'labels_oh',
                                                 prefix + 'loss_entropy',
                                                 cpg=cpg)
        loss_gradients.update(loss_gradients_ME)

    return loss_gradients


def add_entropy_weight(model, rois_pred_blob, rois_blob):
    model.net.Split(rois_pred_blob,
                    [rois_pred_blob + '_bg', rois_pred_blob + '_useless'],
                    split=[1, model.num_classes - 2],
                    axis=1)
    model.net.Concat(
        [rois_pred_blob + '_bg', rois_pred_blob],
        [rois_pred_blob + '_fgbg', rois_pred_blob + '_fgbg_concat_dims'],
        axis=1)

    model.net.Split(rois_blob, [rois_blob + '_useless', rois_blob + '_4'],
                    split=[1, 4],
                    axis=1)
    model.net.Tile(rois_blob + '_4',
                   rois_blob + '_fgbg',
                   axis=1,
                   tiles=model.num_classes)

    model.net.BoxWithNMSLimit(
        [rois_pred_blob + '_fgbg', rois_blob + '_fgbg'],
        [
            rois_pred_blob + '_nms', rois_blob + '_nms',
            rois_blob + '_classes_nms'
        ],
        # score_thresh=cfg.TEST.SCORE_THRESH,
        score_thresh=0.00000000001,
        # nms=cfg.TEST.NMS,
        nms=0.9,
        detections_per_im=999999,
    )
    model.net.RoIEntropy(
        [rois_pred_blob + '_nms', rois_blob + '_classes_nms'],
        [rois_blob + '_entropy'],
        display=int(1280 / cfg.NUM_GPUS),
        num_classes=model.num_classes - 1,
    )

    model.net.ConstantFill('labels_oh', 'labels_oh_one', value=1.0)
    model.net.Sub(['labels_oh_one', 'labels_oh'], 'labels_oh_inv')
    model.net.Max([rois_blob + '_entropy', 'labels_oh_inv'],
                  rois_blob + '_class_weight')

    weight = rois_blob + '_class_weight'
    return weight


def add_spatial_entropy_weight(model, rois_pred, cls_prob, rois):
    model.net.RoIIoU([rois], [rois + '_J'])
    # model.net.Clip(rois + '_J', rois + '_J', max=1.0, min=0.5)

    # model.Transpose(rois_pred, rois_pred + '_t', axes=(1, 0))
    # model.Softmax(rois_pred + '_t', rois_pred + '_softmax_t', axis=1)
    # model.Transpose(
    # rois_pred + '_softmax_t', rois_pred + '_softmax', axes=(1, 0))
    # model.net.Log(rois_pred + '_softmax', rois_pred + '_log')
    # model.net.Mul([rois_pred + '_softmax', rois_pred + '_log'],
    # rois_pred + '__E')
    model.net.Log(rois_pred, rois_pred + '_log')
    model.net.Mul([rois_pred, rois_pred + '_log'], rois_pred + '__E')
    model.net.Scale(rois_pred + '__E', rois_pred + '_E', scale=-1.0)
    model.net.ReplaceNaN(rois_pred + '_E', rois_pred + '_E')
    model.net.MatMul([rois + '_J', rois_pred + '_E'], rois_pred + '_D')
    model.net.LeakyRelu(rois_pred + '_D', rois_pred + '_D')
    model.net.Div([rois_pred + '_E', rois_pred + '_D'], rois_pred + '_G')
    model.net.Mul([rois_pred + '_E', rois_pred + '_G'], rois_pred + '_hatE')

    model.net.ReduceSum(rois_pred + '_hatE',
                        rois_pred + '_hatE_sum',
                        axes=[0],
                        keepdims=True)

    model.net.Shape(rois_pred, rois_pred + '_N', axes=[0])
    model.net.Cast(rois_pred + '_N', rois_pred + '_N_float', to=1)

    # ====================================================================
    if True and False:
        # model.net.ConstantFill([rois_pred], rois_pred + '_one', value=1.0)
        model.net.Tile([cls_prob, rois_pred + '_N'],
                       rois_pred + '_one',
                       axis=0)

        model.net.Div([rois_pred + '_one', rois_pred + '_N_float'],
                      rois_pred + '_ave',
                      broadcast=True)
        model.net.Log(rois_pred + '_ave', rois_pred + '_log_ave')
        model.net.Mul([rois_pred + '_ave', rois_pred + '_log_ave'],
                      rois_pred + '__E_ave')
        model.net.Scale(rois_pred + '__E_ave',
                        rois_pred + '_E_ave',
                        scale=-1.0)
        model.net.ReplaceNaN(rois_pred + '_E_ave', rois_pred + '_E_ave')
        model.net.MatMul([rois + '_J', rois_pred + '_E_ave'],
                         rois_pred + '_D_ave')
        model.net.LeakyRelu(rois_pred + '_D_ave', rois_pred + '_D_ave')
        model.net.Div([rois_pred + '_E_ave', rois_pred + '_D_ave'],
                      rois_pred + '_G_ave')
        model.net.Mul([rois_pred + '_E_ave', rois_pred + '_G_ave'],
                      rois_pred + '_hatE_ave')

        model.net.ReduceSum(rois_pred + '_hatE_ave',
                            rois_pred + '_hatE_sum_ave',
                            axes=[0],
                            keepdims=True)

        model.net.Div([rois_pred + '_hatE_sum', rois_pred + '_hatE_sum_ave'],
                      [rois_pred + '_hatE_sum_norm'])

        if True:
            model.net.Log(cls_prob, cls_prob + '_logy')
            model.net.Log(rois_pred + '_N_float', rois_pred + '_logN')
            model.net.Sub([rois_pred + '_logN', cls_prob + '_logy'],
                          rois_pred + '_logN__logy')
            model.net.Mul([rois_pred + '_logN__logy', cls_prob],
                          rois_pred + '_y_logN__logy')

    else:
        model.net.Log(cls_prob, cls_prob + '_logy')
        model.net.Log(rois_pred + '_N_float', rois_pred + '_logN')
        model.net.Sub([rois_pred + '_logN', cls_prob + '_logy'],
                      rois_pred + '_logN__logy')
        model.net.Mul([rois_pred + '_logN__logy', cls_prob],
                      rois_pred + '_y_logN__logy')

        # model.net.Div([rois_pred + '_hatE_sum', rois_pred + '_logN'],
        # [rois_pred + '_hatE_sum_norm'],
        # broadcast=True)
        model.net.Div([rois_pred + '_hatE_sum', rois_pred + '_y_logN__logy'],
                      [rois_pred + '_hatE_sum_norm'],
                      broadcast=True)
    # ====================================================================

    model.net.Clip(rois_pred + '_hatE_sum_norm',
                   rois_pred + '_hatE_sum_norm',
                   max=1.0,
                   min=0.0)

    model.net.ConstantFill(['labels_oh'], 'labels_oh_one', value=1.0)

    # ====================================================================
    # model.net.Scale(['labels_oh'], ['labels_oh__1'], scale=-1.0)
    # model.net.Mul([rois_pred + '_hatE_sum_norm', 'labels_oh__1'],
    # [rois_pred + '_hatE_sum_norm_sign'])
    # l = model.net.Add([rois_pred + '_hatE_sum_norm_sign', 'labels_oh_one'],
    # [rois + '_class_weight'])

    # ====================================================================

    # bg
    # ====================================================================
    model.net.Sub(['labels_oh_one', 'labels_oh'], ['labels_oh_bg'])

    model.net.Mul([rois_pred + '_hatE_sum_norm', 'labels_oh_bg'], 
                  [rois + '_class_weight_noise'])

    model.net.Sub(['labels_oh_one', rois + '_class_weight_noise'],
                  [rois + '_class_weight'])


    # fg
    # ====================================================================
    # model.net.Mul([rois_pred + '_hatE_sum_norm', 'labels_oh'],
                  # [rois_pred + '_hatE_sum_norm_fg'])

    # model.net.Sub(['labels_oh_one', rois_pred + '_hatE_sum_norm_fg'],
                  # [rois + '_class_weight'])

    # model.net.Sub(['labels_oh_one', 'labels_oh'], ['1__labels_oh'])

    # model.net.Add([rois_pred + '_hatE_sum_norm_fg', '1__labels_oh'],
                  # [rois + '_class_weight_noise'])

    model.StopGradient(rois + '_class_weight', rois + '_class_weight')
    model.StopGradient(rois + '_class_weight_noise', rois + '_class_weight_noise')

    # ====================================================================

    # model.net.Mul([cls_prob, 'labels_oh__1'], [cls_prob + '__1'])
    # model.net.Add([cls_prob + '__1', 'labels_oh'], [cls_prob + '_selfrun'])
    # l = model.net.Add([cls_prob + '_selfrun', rois + '_class_weight_'],
    # [rois + '_class_weight'])

    model.net.Stat(
        [rois + '_class_weight', 'labels_oh_bg'],
        [rois + '_class_weight_stat', 'labels_oh_stat0'],
        display=int(1280 / cfg.NUM_GPUS),
        # display=1,
        prefix='class_weight      ')

    model.net.Stat(
        [rois + '_class_weight_noise', 'labels_oh_bg'],
        [rois + '_class_weight_noise_stat', 'labels_oh_stat1'],
        display=int(1280 / cfg.NUM_GPUS),
        # display=1,
        prefix='class_weight_noise')

    model.net.Stat(
        [rois_pred + '_hatE_sum', 'labels_oh_bg'],
        [rois_pred + '_hatE_sum_bg_stat', 'labels_oh_one_stat2'],
        display=int(1280 / cfg.NUM_GPUS),
        # display=1,
        prefix='hatE_sum bg       ')

    model.net.Stat(
        [rois_pred + '_hatE_sum', 'labels_oh'],
        [rois_pred + '_hatE_sum_fg_stat', 'labels_oh_one_stat3'],
        display=int(1280 / cfg.NUM_GPUS),
        # display=1,
        prefix='hatE_sum fg       ')

    model.net.Stat(
        [rois_pred + '_hatE_sum_norm', 'labels_oh_bg'],
        [rois_pred + '_hatE_sum_norm_bg_stat', 'labels_oh_one_stat4'],
        display=int(1280 / cfg.NUM_GPUS),
        # display=1,
        prefix='hatE_sum_norm bg  ')

    model.net.Stat(
        [rois_pred + '_hatE_sum_norm', 'labels_oh'],
        [rois_pred + '_hatE_sum_norm_fg_stat', 'labels_oh_one_stat5'],
        display=int(1280 / cfg.NUM_GPUS),
        # display=1,
        prefix='hatE_sum_norm fg  ')


    # model.net.Stat(
    # rois_pred + '_hatE_sum_ave',
    # rois_pred + '_hatE_sum_ave_stat',
    # display=int(1280 / cfg.NUM_GPUS),
    # # display=1,
    # prefix='hatE_sum_ave')

    # model.net.Stat(
    # rois_pred + '_y_logN__logy',
    # rois_pred + '_y_logN__logy_stat',
    # display=int(1280 / cfg.NUM_GPUS),
    # # display=1,
    # prefix='y_logN__logy')


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #


def add_VGG16_roi_2fc_noise_head(model,
                                 blob_in,
                                 dim_in,
                                 spatial_scale,
                                 prefix=''):
    ls = []
    dims = []
    l, dim_out = add_VGG16_roi_2fc_head(model,
                                        blob_in,
                                        dim_in,
                                        spatial_scale,
                                        prefix=prefix)
    ls += [l]
    dims += [dim_out]

    # ls += [l]
    # dims += [dim_out]
    # return ls, dims

    if cfg.WSL.CONTEXT:
        return add_VGG16_roi_context_2fc_noise_head(model,
                                                    blob_in,
                                                    dim_in,
                                                    spatial_scale,
                                                    prefix=prefix)
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

    l = prefix + 'roi_feat'

    l = model.FC(l, '_[' + prefix + 'noisy]_' + 'fc6',
                 dim_in * roi_size * roi_size, 4096)
    l = model.Relu(l, '_[' + prefix + 'noisy]_' + 'fc6')
    l = DropoutIfTraining(model, l, '_[' + prefix + 'noisy]_' + 'drop6', 0.5)
    l = model.FC(l, '_[' + prefix + 'noisy]_' + 'fc7', 4096, 4096)
    l = model.Relu(l, '_[' + prefix + 'noisy]_' + 'fc7')
    l = DropoutIfTraining(model, l, '_[' + prefix + 'noisy]_' + 'drop7', 0.5)

    ls += [l]
    dims += [4096]
    return ls, dims


def add_VGG16_roi_2fc_head_shared(model,
                                  blob_in,
                                  dim_in,
                                  spatial_scale,
                                  prefix=''):
    if cfg.WSL.CONTEXT:
        return add_VGG16_roi_context_2fc_head(model, blob_in, dim_in,
                                              spatial_scale)
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat',
        blob_rois=prefix + 'rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    # l = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)
    l = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        model.StopGradient(l, l)

    l = model.FCShared(l,
                       prefix + 'fc6',
                       dim_in * 7 * 7,
                       4096,
                       weight='fc6_w',
                       bias='fc6_b')
    l = model.Relu(l, prefix + 'fc6')
    l = DropoutIfTraining(model, l, prefix + 'drop6', 0.5)
    l = model.FCShared(l,
                       prefix + 'fc7',
                       4096,
                       4096,
                       weight='fc7_w',
                       bias='fc7_b')
    l = model.Relu(l, prefix + 'fc7')
    l = DropoutIfTraining(model, l, prefix + 'drop7', 0.5)

    return l, 4096
