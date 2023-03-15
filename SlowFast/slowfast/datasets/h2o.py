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
class H2o(torch.utils.data.Dataset):
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
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB  # ? 是否需要diff?
        self._num_retries = num_retries
        self.fps = cfg.H2O.FPS
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

        if self.mode == "train" and self.cfg.AUG.ENABLE:
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

        self.epoch = 0.0

        ## This is the customs defiend function;
        (
            self._path_to_videos,
            self._labels,
            self._action_se,
        ) = h2o_helper.load_image_lists(
            self.cfg,
            path_to_file,
            self.cfg.H2O.DATA_ROOT,
            mode=self.mode,
            return_list=True,
        )  # 视频和对应的标签 Video Level;

        if self.cfg.H2O.NOUN_VERB:
            self.noun_verb_maping = np.load(self.cfg.H2O.NOUN_VERB_MAPING, allow_pickle=True).item()

        if self.cfg.H2O.BACKGROUND_CLASS and self.mode == 'train':
            path_to_file_back = os.path.join(
                self.cfg.H2O.ANNOTATION_DIR, "background_{}_36.txt".format(self.mode)
            )  # annos and path info

            assert pathmgr.exists(path_to_file_back), "{} dir not found".format(
                path_to_file_back
            )

            (
                _path_to_videos_back,
                _labels_back,
                _action_se_back,
            ) = h2o_helper.load_image_lists(
                self.cfg,
                path_to_file_back,  
                self.cfg.H2O.DATA_ROOT,
                mode=self.mode,
                return_list=True,
            )  # 视频和对应的标签 Video Level;

            self._path_to_videos.extend(_path_to_videos_back)
            self._labels.extend(_labels_back)
            self._action_se.extend(_action_se_back)

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

        self._action_se = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._action_se]
            )  # 代表每个动作CLIP具体的开始结束的时间, 用于从原始视频中定位到对应的视频片段; 当然需要进一步再对应范围内进行采样; 
        )
        
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [
                    range(self._num_clips)
                    for _ in range(
                        len(self._labels) // self._num_clips
                    )  # 一个完整的视频, 只需要产生一组 clip就好了;
                ]
            )
        )  # 转化为1维度的列表;

        assert len(self._path_to_videos) > 0, "Failed to load H2O dataset"

        logger.info(
            "Constructing H2O dataloader (size: {}) from {} ".format(
                len(self._path_to_videos), path_to_file
            )  # skip rows; # path_to_videos; 有多少个视频? 这里训练集合569个
        )

    def _set_epoch_num(self, epoch):
        self.epoch = epoch  # 每个epoch都会重置一下, 这个有啥用?

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, self._num_yielded = index
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                index, short_cycle_idx = index
        if self.dummy_output is not None:  # this is for code debug.
            return self.dummy_output
        
        if self.mode in ["train"]:
            # -1 indicates random sampling.
            
            temporal_sample_index = -1
            spatial_sample_index = -1  # 训练集合、空间和时间都仅只需要一个样本;

            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]  # 随机缩放(尺寸不定);

            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE  # 剪切尺寸恒定为224;
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if (
                self.cfg.MULTIGRID.DEFAULT_S > 0
            ):  # ?? here the value is 0 temporally
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )

        elif self.mode in ["val"]:
            # -1 indicates random sampling.
            # temporal_sample_index = -2
            # spatial_sample_index = -2  
            spatial_sample_index = -1
            temporal_sample_index = -1

            min_scale = self.cfg.DATA.TEST_CROP_SIZE
            max_scale = self.cfg.DATA.TEST_CROP_SIZE  # 随机缩放(尺寸不定);

            crop_size = self.cfg.DATA.TEST_CROP_SIZE  # 剪切尺寸恒定为224;
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if (
                self.cfg.MULTIGRID.DEFAULT_S > 0
            ):  
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )

        elif self.mode in ["test"]:  ## ?先不看这个把训练集合处理好了再说;
            temporal_sample_index = (
                self._spatial_temporal_idx[index]  # 从时间和空间的角度来看总共有多少个视频?
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )  # 代表时间上的第几个clip, 实际上关于这一点, 指需要进行-1的判断就可以了; 但是仍然可以进行修复; 取值范围(0~10)
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )  # 取值范围(0~3)
                ## 实际上就是建立了spatial-temporal indices 与时空位置的对应关系; 存索引，然后按照指定的规则将数据取出;
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )  # 代表空间上的第几块?
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )  # 这里有一些区别?? 观察一下这一部分;
            # The testing is deterministic and no jitter should be performed. ## 测试的时候不需要对视频进抖动;
            # min_scale, max_scale, and crop_size are expect to be the same. ## 测试的时候是这样;
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            ) 

        num_decode = (
            self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL
            if self.mode in ["train"]
            else 1
        )  # 从一个视频里解码出多少个片段来, 仅训练集合才有可能解码出多个clip来; 用于支持multi-grid训练;

        min_scale, max_scale, crop_size = (
            [min_scale],
            [max_scale],
            [crop_size],
        )  # 视频的空间裁剪情况(最大、最小、保留的图像的尺寸);
        if len(min_scale) < num_decode:  # 定义为1的时候不会进行复制;
            min_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * (
                num_decode - len(min_scale)
            )  # 还差几个需要解码, 每个被解码出来的CLIP都应该满足相同的长度参数;
            max_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[1]] * (
                num_decode - len(max_scale)
            )
            crop_size += (
                [self.cfg.MULTIGRID.DEFAULT_S] * (num_decode - len(crop_size))
                if self.cfg.MULTIGRID.LONG_CYCLE  # ? 循环裁剪? 长边裁剪; 还是短边裁剪?
                or self.cfg.MULTIGRID.SHORT_CYCLE  # ?
                else [self.cfg.DATA.TRAIN_CROP_SIZE]
                * (num_decode - len(crop_size))
            )  # clip的裁剪情况; 这里使能了MULTI_GRID的训练模式, 应该需要注意;
            assert self.mode in ["train", "val"]  # MG模式下才会出现关于crop size的图像复制操作; 

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(
            self._num_retries
        ):  # self._num_retries=100次, 尝试重复读取100次;

            frames_decoded, time_idx_decoded = (
                [None] * num_decode,
                [None] * num_decode,
            )  # 解码出来的不同片段, 按照列表进行保存;
            # for i in range(num_decode):
            num_frames = [
                self.cfg.DATA.NUM_FRAMES
            ]  # 视频里面要保留多少帧? 按照num_decode进行复制;
            sampling_rate = utils.get_random_sampling_rate(  # 这个函数是干嘛的??
                self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
                self.cfg.DATA.SAMPLING_RATE,
            )  # 实际上是一个帧率比较函数, 返回一个视频帧率
            sampling_rate = [sampling_rate]  # [2]

            if len(num_frames) < num_decode:  # 这里是数值的不断重复;
                num_frames.extend(
                    [
                        num_frames[-1]
                        for i in range(num_decode - len(num_frames))
                    ]
                )
                # base case where keys have same frame-rate as query ## ???``
                sampling_rate.extend(
                    [
                        sampling_rate[-1]
                        for i in range(num_decode - len(sampling_rate))
                    ]
                )

            if self.mode in ["train"]:
                assert (
                    len(min_scale)
                    == len(max_scale)
                    == len(crop_size)
                    == num_decode
                )  # 测试的时候本来就不会从单个视频中解码出多个视频片段来;
            # Decode video. Meta info is used to perform selective decoding. # num_decode 永远是1, 仅解码出一个视频片段就可以了;

            target_fps = self.cfg.DATA.TARGET_FPS  # 让模型实际处理的视频帧数有所区别；

            if self.cfg.DATA.TRAIN_JITTER_FPS > 0.0 and self.mode in ["train"]:
                target_fps += random.uniform(
                    0.0, self.cfg.DATA.TRAIN_JITTER_FPS
                )  # 按照一定的概率对FPS进行增强;

            frames, time_idx, tdiff = h2o_helper.decode_frames(
                self._path_to_videos[index],
                self._action_se[index], 
                sampling_rate,  # 设置重复的, 从一个视频中采样多少个片段;
                num_frames,  # 需要保持多少视频帧？
                temporal_sample_index,  # 穿进去的就是一个数据, 索引到时间的第几帧?;
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                fps=self.cfg.H2O.FPS,
                target_fps=target_fps,  # 解决视频帧率不一样的问题;
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                max_spatial_scale=min_scale[0]
                if all(x == min_scale[0] for x in min_scale)
                else 0,  # if self.mode in ["test"] else 0,
                time_diff_prob=self.p_convert_dt
                if self.mode in ["train"]
                else 0.0,
                temporally_rnd_clips=True,
                min_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MIN,  # 对比模型, CLIP之间的最小间隔;
                max_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MAX,  # 对比模型, clip之间的最大间隔? ## 应用在对比学习内, 保证生成的视频样本不要太像;
                context_append=self.cfg.DATA.CONTEXT_APPEND if self.mode in ['train', 'val'] else self.cfg.TEST.CONTEXT_APPEND
            )  # frames = torch.Size([16, 720, 1280, 3]) T H W C; # time_idx array([[13.36949863, 76.36949863,  0.]])
            
            frames_decoded = frames
            time_idx_decoded = time_idx  # ? 构造一个这样的视频样本看看啥情况?
            
            num_aug = (
                self.cfg.DATA.TRAIN_CROP_NUM_SPATIAL
                * self.cfg.AUG.NUM_SAMPLE  # 对样本数目进行扩充;
                if self.mode in ["train"]
                else 1
            )

            num_out = num_aug * num_decode  # decode表示几个clip; # 计算的到的总共有多少个clip;
            f_out, time_idx_out = [None] * num_out, [None] * num_out
            idx = -1
            label = self._labels[index]  

            if self.cfg.H2O.NOUN_VERB:

                noun_label = (self.noun_verb_maping[str(label)]['noun'])
                verb_label = (self.noun_verb_maping[str(label)]['verb'])
            
            ## Here applys data augmentation;
            for i in range(num_decode):  # 几个clip; 几个时空view;
                for _ in range(num_aug):  # 每个clip: 有多少数据增强? # 就是把样本复制了两份?
                    idx += 1
                    f_out[idx] = frames_decoded[
                        i
                    ].clone()  
                    time_idx_out[idx] = time_idx_decoded[i, :]

                    f_out[idx] = f_out[idx].float()
                    f_out[idx] = f_out[idx] / 255.0  # for image; # 是否有必要??

                    if (
                        self.mode in ["train"]
                        and self.cfg.DATA.SSL_COLOR_JITTER  # false: Self-supervised data augmentation;
                    ):
                        f_out[idx] = transform.color_jitter_video_ssl(
                            f_out[idx],
                            bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT, # [0.4, 0.4, 0.4]
                            hue=self.cfg.DATA.SSL_COLOR_HUE, # 0.1
                            p_convert_gray=self.p_convert_gray, # 0.0
                            moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG, # F
                            gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                            gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                        )  # 往里面填充数据;
                    if (
                        self.aug and self.cfg.AUG.AA_TYPE
                    ):  # 'rand-m7-n4-mstd0.5-inc1' # 没有数据增强;
                        aug_transform = create_random_augment(
                            input_size=(
                                f_out[idx].size(1),
                                f_out[idx].size(2),
                            ),  # 对应形状3 * H W
                            auto_augment=self.cfg.AUG.AA_TYPE,
                            interpolation=self.cfg.AUG.INTERPOLATION,
                        )
                        # T H W C -> T C H W.
                        f_out[idx] = f_out[idx].permute(0, 3, 1, 2)
                        list_img = self._frame_to_list_img(f_out[idx])
                        list_img = aug_transform(list_img)
                        f_out[idx] = self._list_img_to_frames(list_img)
                        f_out[idx] = f_out[idx].permute(0, 2, 3, 1)

                    # Perform color normalization.
                    f_out[idx] = utils.tensor_normalize(
                        f_out[idx], self.cfg.DATA.MEAN, self.cfg.DATA.STD
                    )

                    # T H W C -> C T H W.
                    f_out[idx] = f_out[idx].permute(3, 0, 1, 2)

                    scl, asp = (
                        self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
                        self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
                    )
                    relative_scales = (
                        None
                        if (self.mode not in ["train"] or len(scl) == 0)
                        else scl
                    )
                    relative_aspect = (
                        None
                        if (self.mode not in ["train"] or len(asp) == 0)
                        else asp
                    )
                                        
                    f_out[idx] = utils.spatial_sampling(
                        f_out[idx],
                        spatial_idx=spatial_sample_index,  # -1
                        min_scale=min_scale[i],
                        max_scale=max_scale[i],
                        crop_size=crop_size[i],
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,  # True;
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,  # False;
                        aspect_ratio=relative_aspect,  # None;
                        scale=relative_scales,
                        motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT  # False;
                        if self.mode in ["train"] 
                        else False,
                    )

                    if self.rand_erase:  # Default false;
                        erase_transform = RandomErasing(
                            self.cfg.AUG.RE_PROB,
                            mode=self.cfg.AUG.RE_MODE,
                            max_count=self.cfg.AUG.RE_COUNT,
                            num_splits=self.cfg.AUG.RE_COUNT,
                            device="cpu",
                        )
                        f_out[idx] = erase_transform(
                            f_out[idx].permute(1, 0, 2, 3)
                        ).permute(1, 0, 2, 3)

                    f_out[idx] = utils.pack_pathway_output(self.cfg, f_out[idx])
                    if self.cfg.AUG.GEN_MASK_LOADER:  # Default False;
                        mask = self._gen_mask()
                        f_out[idx] = f_out[idx] + [torch.Tensor(), mask]
            frames = f_out[0] if num_out == 1 else f_out
            time_idx = np.array(time_idx_out)
            if (
                num_aug * num_decode > 1
                and not self.cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            ):
                label = [label] * num_aug * num_decode
                index = [index] * num_aug * num_decode

                if self.cfg.H2O.NOUN_VERB:
                    noun_label = [noun_label] * num_aug * num_decode
                    verb_label = [verb_label] * num_aug * num_decode

            if self.cfg.DATA.DUMMY_LOAD:
                if self.dummy_output is None:
                    self.dummy_output = (frames, label, index, time_idx, {})

            # frames.shape: channel, T, H, W;
            if self.mode != 'test':
                if self.cfg.H2O.NOUN_VERB:
                    return frames, (label, noun_label, verb_label), index, time_idx, {}
                
                return frames, label, index, time_idx, {}
            else:
                return frames, index, time_idx, {}
        else:
            logger.warning(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def _gen_mask(self):
        if self.cfg.AUG.MASK_TUBE:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            min_mask = num_masking_patches // 5
            masked_position_generator = MaskingGenerator(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=None,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
            mask = np.tile(mask, (8, 1, 1))
        elif self.cfg.AUG.MASK_FRAMES:
            mask = np.zeros(shape=self.cfg.AUG.MASK_WINDOW_SIZE, dtype=np.int)
            n_mask = round(
                self.cfg.AUG.MASK_WINDOW_SIZE[0] * self.cfg.AUG.MASK_RATIO
            )
            mask_t_ind = random.sample(
                range(0, self.cfg.AUG.MASK_WINDOW_SIZE[0]), n_mask
            )
            mask[mask_t_ind, :, :] += 1
        else:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            max_mask = np.prod(self.cfg.AUG.MASK_WINDOW_SIZE[1:])
            min_mask = max_mask // 5
            masked_position_generator = MaskingGenerator3D(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=max_mask,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
        return mask

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

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
class H2of(torch.utils.data.Dataset):
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
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB  # ? 是否需要diff?
        self._num_retries = num_retries
        self.fps = cfg.H2O.FPS
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

        if self.mode == "train" and self.cfg.AUG.ENABLE:
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

        self.epoch = 0.0

        ## This is the customs defiend function;
        (
            self._path_to_videos,
            self._labels,
            self._action_se,
        ) = h2o_helper.load_flow_lists(
            self.cfg,
            path_to_file,
            self.cfg.H2O.DATA_ROOT,
            mode=self.mode,
            return_list=True,
        )  # 视频和对应的标签 Video Level;

        if self.cfg.H2O.NOUN_VERB:
            self.noun_verb_maping = np.load(self.cfg.H2O.NOUN_VERB_MAPING, allow_pickle=True).item()

        if self.cfg.H2O.BACKGROUND_CLASS and self.mode == 'train':
            path_to_file_back = os.path.join(
                self.cfg.H2O.ANNOTATION_DIR, "background_{}_12.txt".format(self.mode)
            )  # annos and path info

            assert pathmgr.exists(path_to_file_back), "{} dir not found".format(
                path_to_file_back
            )

            (
                _path_to_videos_back,
                _labels_back,
                _action_se_back,
            ) = h2o_helper.load_flow_lists(
                self.cfg,
                path_to_file_back,  
                self.cfg.H2O.DATA_ROOT,
                mode=self.mode,
                return_list=True,
            )  # 视频和对应的标签 Video Level;

            self._path_to_videos.extend(_path_to_videos_back)
            self._labels.extend(_labels_back)
            self._action_se.extend(_action_se_back)

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

        self._action_se = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._action_se]
            )  # 代表每个动作CLIP具体的开始结束的时间, 用于从原始视频中定位到对应的视频片段; 当然需要进一步再对应范围内进行采样; 
        )
        
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [
                    range(self._num_clips)
                    for _ in range(
                        len(self._labels) // self._num_clips
                    )  # 一个完整的视频, 只需要产生一组 clip就好了;
                ]
            )
        )  # 转化为1维度的列表;

        assert len(self._path_to_videos) > 0, "Failed to load H2O dataset"

        logger.info(
            "Constructing H2O dataloader (size: {}) from {} ".format(
                len(self._path_to_videos), path_to_file
            )  # skip rows; # path_to_videos; 有多少个视频? 这里训练集合569个
        )

    def _set_epoch_num(self, epoch):
        self.epoch = epoch  # 每个epoch都会重置一下, 这个有啥用?

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, self._num_yielded = index
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                index, short_cycle_idx = index
        if self.dummy_output is not None:  # this is for code debug.
            return self.dummy_output
        
        if self.mode in ["train"]:
            # -1 indicates random sampling.
            
            temporal_sample_index = -1
            spatial_sample_index = -1  # 训练集合、空间和时间都仅只需要一个样本;

            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]  # 随机缩放(尺寸不定);

            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE  # 剪切尺寸恒定为224;
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if (
                self.cfg.MULTIGRID.DEFAULT_S > 0
            ):  # ?? here the value is 0 temporally
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )

        elif self.mode in ["val"]:
            # -1 indicates random sampling.
            # temporal_sample_index = -2
            # spatial_sample_index = -2  
            spatial_sample_index = -1
            temporal_sample_index = -1

            min_scale = self.cfg.DATA.TEST_CROP_SIZE
            max_scale = self.cfg.DATA.TEST_CROP_SIZE  # 随机缩放(尺寸不定);

            crop_size = self.cfg.DATA.TEST_CROP_SIZE  # 剪切尺寸恒定为224;
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if (
                self.cfg.MULTIGRID.DEFAULT_S > 0
            ):  
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )

        elif self.mode in ["test"]:  ## ?先不看这个把训练集合处理好了再说;
            temporal_sample_index = (
                self._spatial_temporal_idx[index]  # 从时间和空间的角度来看总共有多少个视频?
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )  # 代表时间上的第几个clip, 实际上关于这一点, 指需要进行-1的判断就可以了; 但是仍然可以进行修复; 取值范围(0~10)
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )  # 取值范围(0~3)
                ## 实际上就是建立了spatial-temporal indices 与时空位置的对应关系; 存索引，然后按照指定的规则将数据取出;
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )  # 代表空间上的第几块?
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )  # 这里有一些区别?? 观察一下这一部分;
            # The testing is deterministic and no jitter should be performed. ## 测试的时候不需要对视频进抖动;
            # min_scale, max_scale, and crop_size are expect to be the same. ## 测试的时候是这样;
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            ) 

        num_decode = (
            self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL
            if self.mode in ["train"]
            else 1
        )  # 从一个视频里解码出多少个片段来, 仅训练集合才有可能解码出多个clip来; 用于支持multi-grid训练;

        min_scale, max_scale, crop_size = (
            [min_scale],
            [max_scale],
            [crop_size],
        )  # 视频的空间裁剪情况(最大、最小、保留的图像的尺寸);
        if len(min_scale) < num_decode:  # 定义为1的时候不会进行复制;
            min_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * (
                num_decode - len(min_scale)
            )  # 还差几个需要解码, 每个被解码出来的CLIP都应该满足相同的长度参数;
            max_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[1]] * (
                num_decode - len(max_scale)
            )
            crop_size += (
                [self.cfg.MULTIGRID.DEFAULT_S] * (num_decode - len(crop_size))
                if self.cfg.MULTIGRID.LONG_CYCLE  # ? 循环裁剪? 长边裁剪; 还是短边裁剪?
                or self.cfg.MULTIGRID.SHORT_CYCLE  # ?
                else [self.cfg.DATA.TRAIN_CROP_SIZE]
                * (num_decode - len(crop_size))
            )  # clip的裁剪情况; 这里使能了MULTI_GRID的训练模式, 应该需要注意;
            assert self.mode in ["train", "val"]  # MG模式下才会出现关于crop size的图像复制操作; 

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(
            self._num_retries
        ):  # self._num_retries=100次, 尝试重复读取100次;

            frames_decoded, time_idx_decoded = (
                [None] * num_decode,
                [None] * num_decode,
            )  # 解码出来的不同片段, 按照列表进行保存;
            # for i in range(num_decode):
            num_frames = [
                self.cfg.DATA.NUM_FRAMES
            ]  # 视频里面要保留多少帧? 按照num_decode进行复制;
            sampling_rate = utils.get_random_sampling_rate(  # 这个函数是干嘛的??
                self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
                self.cfg.DATA.SAMPLING_RATE,
            )  # 实际上是一个帧率比较函数, 返回一个视频帧率
            sampling_rate = [sampling_rate]  # [2]

            if len(num_frames) < num_decode:  # 这里是数值的不断重复;
                num_frames.extend(
                    [
                        num_frames[-1]
                        for i in range(num_decode - len(num_frames))
                    ]
                )
                # base case where keys have same frame-rate as query ## ???``
                sampling_rate.extend(
                    [
                        sampling_rate[-1]
                        for i in range(num_decode - len(sampling_rate))
                    ]
                )

            if self.mode in ["train"]:
                assert (
                    len(min_scale)
                    == len(max_scale)
                    == len(crop_size)
                    == num_decode
                )  # 测试的时候本来就不会从单个视频中解码出多个视频片段来;
            # Decode video. Meta info is used to perform selective decoding. # num_decode 永远是1, 仅解码出一个视频片段就可以了;

            target_fps = self.cfg.DATA.TARGET_FPS  # 让模型实际处理的视频帧数有所区别；

            if self.cfg.DATA.TRAIN_JITTER_FPS > 0.0 and self.mode in ["train"]:
                target_fps += random.uniform(
                    0.0, self.cfg.DATA.TRAIN_JITTER_FPS
                )  # 按照一定的概率对FPS进行增强;

            frames, time_idx, tdiff = h2o_helper.decode_flows(
                self._path_to_videos[index],
                self._action_se[index], 
                sampling_rate,  # 设置重复的, 从一个视频中采样多少个片段;
                num_frames,  # 需要保持多少视频帧？
                temporal_sample_index,  # 穿进去的就是一个数据, 索引到时间的第几帧?;
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                fps=self.cfg.H2O.FPS,
                target_fps=target_fps,  # 解决视频帧率不一样的问题;
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                max_spatial_scale=min_scale[0]
                if all(x == min_scale[0] for x in min_scale)
                else 0,  # if self.mode in ["test"] else 0,
                time_diff_prob=self.p_convert_dt
                if self.mode in ["train"]
                else 0.0,
                temporally_rnd_clips=True,
                min_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MIN,  # 对比模型, CLIP之间的最小间隔;
                max_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MAX,  # 对比模型, clip之间的最大间隔? ## 应用在对比学习内, 保证生成的视频样本不要太像;
                context_append=self.cfg.DATA.CONTEXT_APPEND if self.mode in ['train', 'val'] else self.cfg.TEST.CONTEXT_APPEND
            )  # frames = torch.Size([16, 720, 1280, 3]) T H W C; # time_idx array([[13.36949863, 76.36949863,  0.]])
            
            frames_decoded = frames
            time_idx_decoded = time_idx  # ? 构造一个这样的视频样本看看啥情况?
            
            num_aug = (
                self.cfg.DATA.TRAIN_CROP_NUM_SPATIAL
                * self.cfg.AUG.NUM_SAMPLE  # 对样本数目进行扩充;
                if self.mode in ["train"]
                else 1
            )

            num_out = num_aug * num_decode  # decode表示几个clip; # 计算的到的总共有多少个clip;
            f_out, time_idx_out = [None] * num_out, [None] * num_out
            idx = -1
            label = self._labels[index]  

            if self.cfg.H2O.NOUN_VERB:

                noun_label = (self.noun_verb_maping[str(label)]['noun'])
                verb_label = (self.noun_verb_maping[str(label)]['verb'])
            
            ## Here applys data augmentation;
            for i in range(num_decode):  # 几个clip; 几个时空view;
                for _ in range(num_aug):  # 每个clip: 有多少数据增强? # 就是把样本复制了两份?
                    idx += 1
                    f_out[idx] = frames_decoded[
                        i
                    ].clone()  
                    time_idx_out[idx] = time_idx_decoded[i, :]

                    f_out[idx] = f_out[idx].float()
                    f_out[idx] = f_out[idx] / 255.0  # for image; # 是否有必要??

                    if (
                        self.mode in ["train"]
                        and self.cfg.DATA.SSL_COLOR_JITTER  # false: Self-supervised data augmentation;
                    ):
                        f_out[idx] = transform.color_jitter_video_ssl(
                            f_out[idx],
                            bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT, # [0.4, 0.4, 0.4]
                            hue=self.cfg.DATA.SSL_COLOR_HUE, # 0.1
                            p_convert_gray=self.p_convert_gray, # 0.0
                            moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG, # F
                            gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                            gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                        )  # 往里面填充数据;
                    if (
                        self.aug and self.cfg.AUG.AA_TYPE
                    ):  # 'rand-m7-n4-mstd0.5-inc1' # 没有数据增强;
                        aug_transform = create_random_augment(
                            input_size=(
                                f_out[idx].size(1),
                                f_out[idx].size(2),
                            ),  # 对应形状3 * H W
                            auto_augment=self.cfg.AUG.AA_TYPE,
                            interpolation=self.cfg.AUG.INTERPOLATION,
                        )
                        # T H W C -> T C H W.
                        f_out[idx] = f_out[idx].permute(0, 3, 1, 2)
                        list_img = self._frame_to_list_img(f_out[idx])
                        list_img = aug_transform(list_img)
                        f_out[idx] = self._list_img_to_frames(list_img)
                        f_out[idx] = f_out[idx].permute(0, 2, 3, 1)

                    # Perform color normalization.
                    f_out[idx] = utils.tensor_normalize(
                        f_out[idx], self.cfg.DATA.MEAN, self.cfg.DATA.STD
                    )

                    # T H W C -> C T H W.
                    f_out[idx] = f_out[idx].permute(3, 0, 1, 2)

                    scl, asp = (
                        self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
                        self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
                    )
                    relative_scales = (
                        None
                        if (self.mode not in ["train"] or len(scl) == 0)
                        else scl
                    )
                    relative_aspect = (
                        None
                        if (self.mode not in ["train"] or len(asp) == 0)
                        else asp
                    )
                                        
                    f_out[idx] = utils.spatial_sampling(
                        f_out[idx],
                        spatial_idx=spatial_sample_index,  # -1
                        min_scale=min_scale[i],
                        max_scale=max_scale[i],
                        crop_size=crop_size[i],
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,  # True;
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,  # False;
                        aspect_ratio=relative_aspect,  # None;
                        scale=relative_scales,
                        motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT  # False;
                        if self.mode in ["train"] 
                        else False,
                    )

                    if self.rand_erase:  # Default false;
                        erase_transform = RandomErasing(
                            self.cfg.AUG.RE_PROB,
                            mode=self.cfg.AUG.RE_MODE,
                            max_count=self.cfg.AUG.RE_COUNT,
                            num_splits=self.cfg.AUG.RE_COUNT,
                            device="cpu",
                        )
                        f_out[idx] = erase_transform(
                            f_out[idx].permute(1, 0, 2, 3)
                        ).permute(1, 0, 2, 3)

                    f_out[idx] = utils.pack_pathway_output(self.cfg, f_out[idx])
                    if self.cfg.AUG.GEN_MASK_LOADER:  # Default False;
                        mask = self._gen_mask()
                        f_out[idx] = f_out[idx] + [torch.Tensor(), mask]
            frames = f_out[0] if num_out == 1 else f_out
            time_idx = np.array(time_idx_out)
            if (
                num_aug * num_decode > 1
                and not self.cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            ):
                label = [label] * num_aug * num_decode
                index = [index] * num_aug * num_decode

                if self.cfg.H2O.NOUN_VERB:
                    noun_label = [noun_label] * num_aug * num_decode
                    verb_label = [verb_label] * num_aug * num_decode

            if self.cfg.DATA.DUMMY_LOAD:
                if self.dummy_output is None:
                    self.dummy_output = (frames, label, index, time_idx, {})
            # frames.shape: channel, T, H, W;
            
            if self.mode != 'test':
                if self.cfg.H2O.NOUN_VERB:
                    return frames, (label, noun_label, verb_label), index, time_idx, {}
                
                return frames, label, index, time_idx, {}
            else:
                return frames, index, time_idx, {}
        else:
            logger.warning(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def _gen_mask(self):
        if self.cfg.AUG.MASK_TUBE:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            min_mask = num_masking_patches // 5
            masked_position_generator = MaskingGenerator(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=None,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
            mask = np.tile(mask, (8, 1, 1))
        elif self.cfg.AUG.MASK_FRAMES:
            mask = np.zeros(shape=self.cfg.AUG.MASK_WINDOW_SIZE, dtype=np.int)
            n_mask = round(
                self.cfg.AUG.MASK_WINDOW_SIZE[0] * self.cfg.AUG.MASK_RATIO
            )
            mask_t_ind = random.sample(
                range(0, self.cfg.AUG.MASK_WINDOW_SIZE[0]), n_mask
            )
            mask[mask_t_ind, :, :] += 1
        else:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            max_mask = np.prod(self.cfg.AUG.MASK_WINDOW_SIZE[1:])
            min_mask = max_mask // 5
            masked_position_generator = MaskingGenerator3D(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=max_mask,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
        return mask

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

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

