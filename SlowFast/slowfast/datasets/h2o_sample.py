#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import pdb
import numpy as np
import os
import random
import pandas
import torch
import torch.utils.data
from torchvision import transforms
from itertools import chain as chain

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import decoder as decoder
from . import transform as transform
from . import utils as utils
from . import h2o_helper as h2o_helper
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import (
    MaskingGenerator,
    MaskingGenerator3D,
    create_random_augment,
)

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class H2os(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for H2O".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB  # ? 是否需要diff?
        self._num_retries = num_retries
        self.fps = cfg.H2O.FPS
        # self.cfg.AUG.NUM_SAMPLE = 1
        self._num_epoch = 0.0
        self._num_yielded = 0

        self.dummy_output = None
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1  # 代表一个视频需要取出集合clip;
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )  # 采集多少个片段, 每个片段取多少个CROPS;

        logger.info("Constructing H2O {}...".format(mode))
        self._construct_loader()
        self.aug = False  # 默认初始化
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0  # 这些都是啥? 咋想出来的;
        self.cur_epoch = 0

        if self.mode == "train" and self.cfg.AUG.ENABLE:  # 关闭了?
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:  # 训练时被起用, 这是什么参数?
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.H2O.ANNOTATION_DIR, "action_{}.txt".format(self.mode)
        )  # annos and path info

        assert pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self.cur_iter = 0  #
        self.epoch = 0.0

        ## This is the customs defiend function;
        (self._path_to_videos, self._labels, self._action_se) = h2o_helper.load_image_lists(
            self.cfg,
            path_to_file,
            self.cfg.H2O.DATA_ROOT,
            mode=self.mode,
            return_list=True,
        )  # 视频和对应的标签 Video Level;

        self._path_to_videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._path_to_videos]
            )  # 视频复制若干次;
        )  # repeat the videos;
        self._labels = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._labels]
            )  # 每个视频的label复制若干次;
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                # [range(clip_idx, clip_idx+self._num_clips) for clip_idx in range(len(self._labels))]
                [range(self._num_clips) for _ in range(len(self._labels))]
            )
        )  # 转化为1维度的列表;

        logger.info(
            "Constructing H2O dataloader (size: {}) from {} ".format(
                len(self._path_to_videos), path_to_file
            )  # skip rows; # path_to_videos; 有多少个视频? 这里训练集合569个
        )  # the loadded dataset infomation;

    def get_seq_frames(self, index):
        """
        Given the video index, return the list of indexs of sampled frames.
        Args:
            index (int): the video index.
        Returns:
            seq (list): the indexes of sampled frames from the video.
        """
        temporal_sample_index = (
            -1
            if self.mode in ["train", "val"]
            else self._spatial_temporal_idx[index]
            // self.cfg.TEST.NUM_SPATIAL_CROPS
        )  # 时间维度上要采样几个小片段?
        num_frames = self.cfg.DATA.NUM_FRAMES  # 视频的总帧数
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,  #
        )  # 采样帧率? 如果不使用循环采样的话, 就使用原始的视频帧率;
        video_length = len(self._path_to_videos[index])  # 每个clip所对应的视频总帧数;

        clip_length = (
            num_frames - 1
        ) * sampling_rate + 1  # 整个clip的长度; 每个CLIP的目标长度;
        
        action_start, action_end = self._action_se[index]
        action_length = action_end - action_start
        action_center = (action_start + action_end) / 2
        
        if temporal_sample_index == -1:
            
            if clip_length > action_length:
                center_range_helf = int((clip_length - action_length) / 2)
            else:
                center_range_helf = int((-clip_length + action_length) / 2)
            
            clip_shift = np.random.randint(-center_range_helf, center_range_helf + 1)
            clip_center = action_center + clip_shift # 默认情况下两者是重合的; clip_center决定了采样clip的具体具体位置; 
            
        else:
            # gap = float(max(video_length - clip_length, 0)) / (self.cfg.TEST.NUM_ENSEMBLE_VIEWS - 1)

            if clip_length > action_length: # 包含不同的上下文?(Multi-View得有区别才行;)
                center_range = ((clip_length - action_length))
            else:
                center_range = ((-clip_length + action_length)) # 表示一个偏移的范围; 
            

            clip_gap = (center_range) / (self.cfg.TEST.NUM_ENSEMBLE_VIEWS - 1) # 10 个CLIP对应了9个GAP,
            center_start = int(action_center - (center_range / 2)) # 中心点的变化范围
            
            clip_center = int(
                round(clip_gap * temporal_sample_index) + center_start, 
            )

            # pdb.set_trace()


        index = np.arange(-clip_length//2, clip_length//2, 2) 
        index = clip_center + index

        index = np.clip(index, 0, video_length - 1).astype(np.int32)
        
        return index
        """
        if temporal_sample_index == -1:
            if clip_length > video_length:  # 超出原视频长度的采样方式;
                start = random.randint(video_length - clip_length, 0)
            else:
                start = random.randint(0, video_length - clip_length)
        else:  # 先不用管测试细节;
            gap = float(max(video_length - clip_length, 0)) / (
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS - 1
            )  # 这个gap用于尽量覆盖整个视频;
            start = int(
                round(gap * temporal_sample_index)
            )  # 测试的时候需要剪裁多帧, 与GAP相乘的到最后的预测;
        seq = [
            max(min(start + i * sampling_rate, video_length - 1), 0)  #
            for i in range(num_frames)  # 确定起始点之后等间隔的进行采样;
        ]  # 当序列不够长的时候, 采用视频的首帧和末尾帧来补齐序列长度;
        return seq
        """
        

    def _set_epoch_num(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
            time index (zero): The time index is currently not supported.
            {} extra data, currently not supported
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, self._num_yielded = index
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        seq = self.get_seq_frames(index)  # return the index of the video;

        frames = torch.as_tensor(
            utils.retry_load_images(
                [
                    self._path_to_videos[index][frame] for frame in seq
                ],  # 逐帧进行读取;
                self._num_retries,
            )
        )
        label = self._labels[index]
        # label = torch.as_tensor(
        #     utils.as_binary_vector([label, self.cfg.MODEL.NUM_CLASSES)
        # )
        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(
            3, 0, 1, 2
        )  # data Augmentation from the spatial dimension;
        # Perform data augmentation. ## These are tricks;
        # pdb.set_trace()
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,  # TV, 永远保持索引是-1;
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )
        frames = utils.pack_pathway_output(self.cfg, frames)
        if self.mode != 'test':
            return frames, label, index, 0, {}
        else:
            return frames, index, 0, {}

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)  # 视频解码;


@DATASET_REGISTRY.register()
class H2osf(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for H2O".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB  # ? 是否需要diff?
        self._num_retries = num_retries
        self.fps = cfg.H2O.FPS
        # self.cfg.AUG.NUM_SAMPLE = 1
        self._num_epoch = 0.0
        self._num_yielded = 0

        self.dummy_output = None
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1  # 代表一个视频需要取出集合clip;
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )  # 采集多少个片段, 每个片段取多少个CROPS;

        logger.info("Constructing H2O {}...".format(mode))
        self._construct_loader()
        self.aug = False  # 默认初始化
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0  # 这些都是啥? 咋想出来的;
        self.cur_epoch = 0

        if self.mode == "train" and self.cfg.AUG.ENABLE:  # 关闭了?
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:  # 训练时被起用, 这是什么参数?
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.H2O.ANNOTATION_DIR, "action_{}.txt".format(self.mode)
        )  # annos and path info

        assert pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self.cur_iter = 0  #
        self.epoch = 0.0

        ## This is the customs defiend function;
        (self._path_to_videos, self._labels, self._action_se) = h2o_helper.load_flow_lists(
            self.cfg,
            path_to_file,
            self.cfg.H2O.DATA_ROOT,
            mode=self.mode,
            return_list=True,
        )  # 视频和对应的标签 Video Level; # 所有的视频、所有的视频所有的帧、每帧对应视频的RGB; 
        
        self._path_to_videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._path_to_videos]
            )  # 视频复制若干次;
        )  # repeat the videos;
        
        self._labels = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._labels]
            )  # 每个视频的label复制若干次;
        )
        
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                # [range(clip_idx, clip_idx+self._num_clips) for clip_idx in range(len(self._labels))]
                [range(self._num_clips) for _ in range(len(self._labels))]
            )
        )  # 转化为1维度的列表;

        logger.info(
            "Constructing H2O dataloader (size: {}) from {} ".format(
                len(self._path_to_videos), path_to_file
            )  # skip rows; # path_to_videos; 有多少个视频? 这里训练集合569个
        )  # the loadded dataset infomation;

    def get_seq_frames(self, index):
        """
        Given the video index, return the list of indexs of sampled frames.
        Args:
            index (int): the video index.
        Returns:
            seq (list): the indexes of sampled frames from the video.
        """
        temporal_sample_index = (
            -1
            if self.mode in ["train", "val"]
            else self._spatial_temporal_idx[index]
            // self.cfg.TEST.NUM_SPATIAL_CROPS
        )  # 时间维度上要采样几个小片段?
        num_frames = self.cfg.DATA.NUM_FRAMES  # 视频的总帧数
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,  #
        )  # 采样帧率? 如果不使用循环采样的话, 就使用原始的视频帧率;
        video_length = len(self._path_to_videos[index])  # 每个clip所对应的视频总帧数;

        clip_length = (
            num_frames - 1
        ) * sampling_rate + 1  # 整个clip的长度; 每个CLIP的目标长度;
        
        action_start, action_end = self._action_se[index]
        action_length = action_end - action_start
        action_center = (action_start + action_end) / 2
        
        if temporal_sample_index == -1:
            
            if clip_length > action_length:
                center_range_helf = int((clip_length - action_length) / 2)
            else:
                center_range_helf = int((-clip_length + action_length) / 2)
            
            clip_shift = np.random.randint(-center_range_helf, center_range_helf + 1)
            clip_center = action_center + clip_shift # 默认情况下两者是重合的; clip_center决定了采样clip的具体具体位置; 
            
        else:
            # gap = float(max(video_length - clip_length, 0)) / (self.cfg.TEST.NUM_ENSEMBLE_VIEWS - 1)
            if clip_length > action_length: # 包含不同的上下文?(Multi-View得有区别才行;)
                center_range = ((clip_length - action_length))
            else:
                center_range = ((-clip_length + action_length)) # 表示一个偏移的范围; 
            

            clip_gap = (center_range) / (self.cfg.TEST.NUM_ENSEMBLE_VIEWS - 1) # 10 个CLIP对应了9个GAP,
            center_start = int(action_center - (center_range / 2)) # 中心点的变化范围
            
            clip_center = int(
                round(clip_gap * temporal_sample_index) + center_start, 
            )

            # pdb.set_trace()


        index = np.arange(-clip_length//2, clip_length//2, 2) 
        index = clip_center + index

        index = np.clip(index, 0, video_length - 1).astype(np.int32)
        
        return index

    def _set_epoch_num(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
            time index (zero): The time index is currently not supported.
            {} extra data, currently not supported
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, self._num_yielded = index
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        seq = self.get_seq_frames(index)  # return the index of the video;

        frames = torch.as_tensor(
            utils.retry_load_flows(
                [
                    self._path_to_videos[index][frame] for frame in seq
                ],  # 逐帧进行读取;
                self._num_retries,
            )
        )
        label = self._labels[index]
        # label = torch.as_tensor(
        #     utils.as_binary_vector([label, self.cfg.MODEL.NUM_CLASSES)
        # )
        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(
            3, 0, 1, 2
        )  # data Augmentation from the spatial dimension;
        # Perform data augmentation. ## These are tricks;
        # pdb.set_trace()
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,  # TV, 永远保持索引是-1;
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )
        frames = utils.pack_pathway_output(self.cfg, frames)
        if self.mode != 'test':
            return frames, label, index, 0, {}
        else:
            return frames, index, 0, {}

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)  # 视频解码;