#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import pdb
import numpy as np
import os
import pickle
import json
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_info, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()

    num_videos = test_info['num_videos']
    num_cls = test_info['num_class']
    ensemble_method = test_info['ensemble_methods']
    num_clips = test_info['num_clips']

    video_preds = torch.zeros((num_videos, num_cls)) # 存储所有视频的预测结果; # 242 * 37 ## 最后获取Topk数值的时候不要把0号概率包括在内; 

    for cur_iter, (inputs, video_idx, time, meta) in enumerate(
        test_loader
    ):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            video_idx = video_idx.cuda() # 这个参数有什么用? 
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        preds = model(inputs) # 直接进行预测, 那么如何将多个View的结果进行整合? Normal aggregation; 
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, video_idx = du.all_gather([preds, video_idx]) # 多卡测试? 
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            video_idx = video_idx.cpu()


        if not cfg.VIS_MASK.ENABLE:
            # Update and log stats.
            """
            test_meter.update_stats(
                preds.detach(), torch.arange(preds.shape[0]), video_idx.detach()
            ) 
            """
            for ind in range(preds.shape[0]):
                vid_id = int(video_idx[ind]) // num_clips # 每个视频有多少个CLIP进行聚合? 

                if ensemble_method == "sum":
                    video_preds[vid_id] += preds[ind]
                elif ensemble_method == "max":
                    video_preds[vid_id] = torch.max(
                        video_preds[vid_id], preds[ind]
                    )
                else:
                    raise NotImplementedError(
                        "Ensemble Method {} is not supported".format(
                            ensemble_method
                        )
                    )
    
    if cfg.TEST.SAVE_SCORES:
        video_preds = video_preds / num_clips 

    return video_preds 


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]


    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS: # 在时间维度上进行遍历; 

        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view 

        # Print config.
        logger.info("Test with config:")
        logger.info(cfg)

        # Build the video model and print model statistics.
        model = build_model(cfg) 
        flops, params = 0.0, 0.0 
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            model.eval()
            flops, params = misc.log_model_info(
                model, cfg, use_train_input=False
            ) 

        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=False)

        cu.load_test_checkpoint(cfg, model)

        # Create video testing loaders.
        test_loader = loader.construct_loader(cfg, "test")
        logger.info("Testing model for {} iterations".format(len(test_loader)))


        assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        
        test_info = {
            'num_videos':test_loader.dataset.num_videos // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            'num_clips': cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            'num_class':cfg.MODEL.NUM_CLASSES,
            'overall_iters': len(test_loader), 
            'ensemble_methods': cfg.DATA.ENSEMBLE_METHOD, 
        }

        # # Perform multi-view test on the entire dataset.
        video_preds = perform_test(test_loader, model, test_info, cfg) # 的到所有视频的预测结果; 

        tag = 'rgb_SF_224_MIX_HR'
        save_path = 'xx/action_scores_{}'.format(tag)
        np.save(save_path, video_preds)

        ## save the results
        results = {}
        for idx, pred in enumerate(video_preds, start=1):
            
            if cfg.H2O.BACKGROUND_CLASS:
                assert cfg.MODEL.NUM_CLASSES==37
                pred_ = torch.argmax(pred[1:])+1 # 不要背景类别
            
            else:
                assert cfg.MODEL.NUM_CLASSES==36
                pred_ = torch.argmax(pred)+1 # 不要背景类别
            
            results[str(idx)] = pred_.item()

        results["modality"] = "RGB"

        save_path = 'xx/action_labels_{}.json'.format(tag)
        with open(save_path, 'w') as f:
            json.dump(results, f)

    result_string_views = "_p{:.2f}_f{:.2f}".format(params / 1e6, flops) # parameters and flops

    for view in cfg.TEST.NUM_TEMPORAL_CLIPS:
        logger.info(
            "Finalized testing with {} temporal clips and {} spatial crops".format(
                view, cfg.TEST.NUM_SPATIAL_CROPS
            )
        ) 
        result_string = (
            "_p{:.2f}_f{:.2f}_{} MEM: {:.2f} f: {:.4f}"
            "".format(
                params / 1e6,
                flops,
                view,
                misc.gpu_mem_usage(),
                flops,
            )
        )
        logger.info("{}".format(result_string))