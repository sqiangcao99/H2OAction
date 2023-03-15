import pdb

import math
import logging
import numpy as np
import os
import random
import time
from collections import defaultdict
import cv2
import torch
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms
import torchvision.io as io

from slowfast.utils.env import pathmgr

from . import transform as transform

from .random_erasing import RandomErasing
from .transform import create_random_augment
from . import transform as transform
from . import utils as utils  # 对相关方法进行二次包装;


logger = logging.getLogger(__name__)


def load_clip_image_lists(
    cfg, frame_list_file, prefix="", mode="train", return_list=False
):
    """
    Load image paths and labels from a "frame list".
    Each line of the frame list contains:
    `original_vido_id video_id frame_id path labels`
    Args:
        frame_list_file (string): path to the frame list.
        prefix (str): the prefix for the path.
        return_list (bool): if True, return a list. If False, return a dict.
    Returns:
        image_paths (list or dict): list of list containing path to each frame.
            If return_list is False, then return in a dict form.
        labels (list or dict): list of list containing label of each frame.
            If return_list is False, then return in a dict form.
    """
    image_paths = defaultdict(list)  # 定义字典值的数据类型;
    labels = {}
    # labels = defaultdict(list)
    # image_paths = {}

    with pathmgr.open(frame_list_file, "r") as f:
        assert f.readline().startswith("id")
        for line in f:
            row = line.split()
            # original_vido_id video_id frame_id path labels

            if mode == "test":
                assert len(row) == 6
                (
                    video_idx,
                    video_path,
                    action_start,
                    action_end,
                    video_start,
                    video_end,
                ) = row
            else:
                assert len(row) == 7
                (
                    video_idx,
                    video_path,
                    video_labels,
                    action_start,
                    action_end,
                    video_start,
                    video_end,
                ) = row

            video_name = video_path

            if prefix == "":  # 没有给定数据集的根路径;
                # video_indices = np.arange(int(video_start), int(video_end)) 
                video_indices = np.arange(int(action_start), int(action_end)) 
                for indice in video_indices:
                    fname = cfg.H2O.FRAME_TEMPL.format(indice)
                    path = os.path.join(
                        video_path, cfg.H2O.VIEW[0], cfg.H2O.FRAME_FOLDER, fname
                    )
                    image_paths[video_idx].append(path)
            else:
                # path = os.path.join(prefix, path) # retrievl the absolute path of the video;
                # video_indices = np.arange(int(video_start), int(video_end)) 
                video_indices = np.arange(int(action_start), int(action_end)) 

                for indice in video_indices:
                    fname = cfg.H2O.FRAME_TEMPL.format(indice)
                    path = os.path.join(
                        prefix,
                        video_path,
                        cfg.H2O.VIEW[0],
                        cfg.H2O.FRAME_FOLDER,
                        fname,
                    )
                    image_paths[video_idx].append(path)
    
            if mode != "test":
                labels[video_idx] = int(video_labels) - 1 
            else:
                labels[video_idx] = None # 对于测试集合而言, 不需要返回标签; 

    assert len(image_paths) == len(labels)

    if return_list:
        keys = image_paths.keys()
        image_paths = [image_paths[key] for key in keys]
        labels = [labels[key] for key in keys]
        return image_paths, labels # 返回的是所有帧的地址, 对应CLIP的动作标签, 以及相应CLIP的开始和结束时间; 

    return dict(image_paths), dict(labels)


def load_image_lists(
    cfg, frame_list_file, prefix="", mode="train", return_list=False
):
    """
    Load image paths and labels from a "frame list".
    Each line of the frame list contains:
    `original_vido_id video_id frame_id path labels`
    Args:
        frame_list_file (string): path to the frame list.
        prefix (str): the prefix for the path.
        return_list (bool): if True, return a list. If False, return a dict.
    Returns:
        image_paths (list or dict): list of list containing path to each frame.
            If return_list is False, then return in a dict form.
        labels (list or dict): list of list containing label of each frame.
            If return_list is False, then return in a dict form.
    """
    image_paths = defaultdict(list)  # 定义字典值的数据类型;
    action_se = defaultdict(list)  # 定义字典值的数据类型;
    labels = {}
    # labels = defaultdict(list)
    # image_paths = {}

    with pathmgr.open(frame_list_file, "r") as f:
        assert f.readline().startswith("id")
        for line in f:
            row = line.split()
            # original_vido_id video_id frame_id path labels

            if mode == "test":
                assert len(row) == 6
                (
                    video_idx,
                    video_path,
                    action_start,
                    action_end,
                    video_start,
                    video_end,
                ) = row
            else:
                assert len(row) == 7
                (
                    video_idx,
                    video_path,
                    video_labels,
                    action_start,
                    action_end,
                    video_start,
                    video_end,
                ) = row

            video_name = video_path
            if prefix == "":  # 没有给定数据集的根路径;
                video_indices = np.arange(int(video_start), int(video_end)) 
                for indice in video_indices:
                    fname = cfg.H2O.FRAME_TEMPL.format(indice)
                    path = os.path.join(
                        video_path, cfg.H2O.VIEW[0], cfg.H2O.FRAME_FOLDER, fname
                    )
                    image_paths[video_idx].append(path)
            else:
                # path = os.path.join(prefix, path) # retrievl the absolute path of the video;
                video_indices = np.arange(int(video_start), int(video_end)) 
                for indice in video_indices:
                    fname = cfg.H2O.FRAME_TEMPL.format(indice)
                    path = os.path.join(
                        prefix,
                        video_path,
                        cfg.H2O.VIEW[0],
                        cfg.H2O.FRAME_FOLDER,
                        fname,
                    )
                    image_paths[video_idx].append(path)
            action_se[video_idx].extend([int(action_start), int(action_end)])
            
            if mode != "test":
                if cfg.H2O.BACKGROUND_CLASS:
                    labels[video_idx] = int(video_labels) 
                else:
                    labels[video_idx] = int(video_labels) - 1
            else:
                labels[video_idx] = None # 对于测试集合而言, 不需要返回标签; 

    assert len(image_paths) == len(labels) and len(labels) == len(action_se)

    if return_list:
        keys = image_paths.keys()
        image_paths = [image_paths[key] for key in keys]
        labels = [labels[key] for key in keys]
        action_se = [action_se[key] for key in keys]
        return image_paths, labels, action_se # 返回的是所有帧的地址, 对应CLIP的动作标签, 以及相应CLIP的开始和结束时间; 

    return dict(image_paths), dict(labels), dict(action_se)


def load_flow_lists(
    cfg, frame_list_file, prefix="", mode="train", return_list=False
): # 每个索引对应的应该是一张图片的两个方向的光流; 
    """
    Load image paths and labels from a "frame list".
    Each line of the frame list contains:
    `original_vido_id video_id frame_id path labels`
    Args:
        frame_list_file (string): path to the frame list.
        prefix (str): the prefix for the path.
        return_list (bool): if True, return a list. If False, return a dict.
    Returns:
        image_paths (list or dict): list of list containing path to each frame.
            If return_list is False, then return in a dict form.
        labels (list or dict): list of list containing label of each frame.
            If return_list is False, then return in a dict form.
    """
    image_paths = defaultdict(list)  # 定义字典值的数据类型;
    action_se = defaultdict(list)  # 定义字典值的数据类型;
    labels = {}
    # labels = defaultdict(list)
    # image_paths = {}

    with pathmgr.open(frame_list_file, "r") as f:
        assert f.readline().startswith("id")
        for line in f:
            row = line.split()
            # original_vido_id video_id frame_id path labels

            if mode == "test":
                assert len(row) == 6
                (
                    video_idx,
                    video_path,
                    action_start,
                    action_end,
                    video_start,
                    video_end,
                ) = row
            else:
                assert len(row) == 7
                (
                    video_idx,
                    video_path,
                    video_labels,
                    action_start,
                    action_end,
                    video_start,
                    video_end,
                ) = row

            video_name = video_path
            # path = os.path.join(prefix, path) # retrievl the absolute path of the video;
            video_indices = np.arange(int(video_start), int(video_end)-1) 
            for indice in video_indices:

                flow_x_name = cfg.H2O.FRAME_TEMPL.format('x', indice)
                flow_y_name = cfg.H2O.FRAME_TEMPL.format('y', indice)

                flow_x_path = os.path.join(
                    prefix,
                    video_path,
                    cfg.H2O.VIEW[0],
                    cfg.H2O.FRAME_FOLDER,
                    flow_x_name,
                )

                flow_y_path = os.path.join(
                    prefix,
                    video_path,
                    cfg.H2O.VIEW[0],
                    cfg.H2O.FRAME_FOLDER,
                    flow_y_name,
                )
                image_paths[video_idx].append([flow_x_path, flow_y_path]) # X,Y
            action_se[video_idx].extend([int(action_start), int(action_end)])
            
            if mode != "test":
                if cfg.H2O.BACKGROUND_CLASS:
                    labels[video_idx] = int(video_labels) 
                else:
                    labels[video_idx] = int(video_labels) - 1
            else:
                labels[video_idx] = None # 对于测试集合而言, 不需要返回标签; 

    assert len(image_paths) == len(labels) and len(labels) == len(action_se)

    if return_list:
        keys = image_paths.keys()
        image_paths = [image_paths[key] for key in keys]
        labels = [labels[key] for key in keys]
        action_se = [action_se[key] for key in keys]
        return image_paths, labels, action_se # 返回的是所有帧的地址, 对应CLIP的动作标签, 以及相应CLIP的开始和结束时间; 
    
    return dict(image_paths), dict(labels), dict(action_se)



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
    num_frames = self.cfg.DATA.NUM_FRAMES  # 视频的总帧数, 这里是仅采样一个视频;
    sampling_rate = utils.get_random_sampling_rate(
        self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
        self.cfg.DATA.SAMPLING_RATE,  #
    )  # 采样帧率? 如果不使用循环采样的话, 就使用原始的视频帧率;
    video_length = len(self._path_to_videos[index])  # 每个clip所对应的视频总帧数;
    assert video_length == len(self._labels[index])  # ?? 为什么会想等? 也就是做这个判断的目的是啥?

    clip_length = (
        num_frames - 1
    ) * sampling_rate + 1  # 整个clip的长度; 每个CLIP的目标长度;

    if temporal_sample_index == -1:
        if clip_length > video_length:
            start = random.randint(video_length - clip_length, 0)
        else:
            start = random.randint(0, video_length - clip_length)
    else:
        gap = float(max(video_length - clip_length, 0)) / (
            self.cfg.TEST.NUM_ENSEMBLE_VIEWS - 1
        )  # 这个gap用于尽量覆盖整个视频;
        start = int(
            round(gap * temporal_sample_index)
        )  # 测试的时候需要剪裁多帧, 与GAP相乘的到最后的预测;
    seq = [
        max(min(start + i * sampling_rate, video_length - 1), 0)  #
        for i in range(num_frames)  # 确定起始点之后等间隔的进行采样;
    ]
    return seq


def load_frames(self, index):
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
    num_frames = self.cfg.DATA.NUM_FRAMES  # 视频的总帧数, 这里是仅采样一个视频;
    sampling_rate = utils.get_random_sampling_rate(
        self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
        self.cfg.DATA.SAMPLING_RATE,  #
    )  # 采样帧率? 如果不使用循环采样的话, 就使用原始的视频帧率;
    video_length = len(self._path_to_videos[index])  # 每个clip所对应的视频总帧数;
    assert video_length == len(self._labels[index])  # ?? 为什么会想等? 也就是做这个判断的目的是啥?

    clip_length = (
        num_frames - 1
    ) * sampling_rate + 1  # 整个clip的长度; 每个CLIP的目标长度;

    if temporal_sample_index == -1:
        if clip_length > video_length:
            start = random.randint(video_length - clip_length, 0)
        else:
            start = random.randint(0, video_length - clip_length)
    else:
        gap = float(max(video_length - clip_length, 0)) / (
            self.cfg.TEST.NUM_ENSEMBLE_VIEWS - 1
        )  # 这个gap用于尽量覆盖整个视频;
        start = int(
            round(gap * temporal_sample_index)
        )  # 测试的时候需要剪裁多帧, 与GAP相乘的到最后的预测;
    seq = [
        max(min(start + i * sampling_rate, video_length - 1), 0)  #
        for i in range(num_frames)  # 确定起始点之后等间隔的进行采样;
    ]
    return seq


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`. # 帧数乘channel数量乘高度成宽度;
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)  # 仅需要知道起始和终止时间就可以了;
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)  # 在某个维度上直接按照索引将对应的数据取出来;
    return frames

def temporal_uniform_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`. # 帧数乘channel数量乘高度成宽度;
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    sampling_rate = int((end_idx + 1 - start_idx) / num_samples)   
    index =torch.arange(start_idx, end_idx+1, sampling_rate).long()
    # index = torch.clamp(index, 0, frames.shape[0] - 1) # 因为已经提前截取好了, 因此这里只需要将对应的视频帧检索出来就可以了; 
    pdb.set_trace()
    frames = torch.index_select(frames, 0, index)  # 在某个维度上直接按照索引将对应的数据取出来;
    return frames

def get_start_end_idx(
    video_size, clip_size, clip_idx, num_clips_uniform, use_offset=False
):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips_uniform clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips_uniform
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips_uniform == 1:
                # Take the center clip if num_clips_uniform is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(
                    delta / (num_clips_uniform - 1)
                )

        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips_uniform
    end_idx = start_idx + clip_size - 1

    return start_idx, end_idx, start_idx / delta if delta != 0 else 0.0


def get_multiple_start_end_idx( 
    video_size,  # 视频的总帧数;
    action_start,
    action_end,
    clip_sizes,  # 按照原始的帧率进行采样;
    clip_idx,  # 第几个ID;
    num_clips_uniform,  # 均匀采样总共几个视频片段; 这里的时间采样采用的是均匀采样策略; 且最终只返回一个视频clip;
    min_delta=0,  # 视频片段之间的最小间隔;
    max_delta=math.inf,  # 视频片段的最大间隔
    use_offset=False,  #
    context_append = 0,
):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled(测试集), otherwise uniformly split the video to
    num_clips_uniform clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_sizes (list): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips_uniform
            clips, and select the start and end index of the clip_idx-th video
            clip. # 训练集采用的是一种随机裁剪的策略, 测试时采用的是
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video for testing. # 总共需要的视频帧数是多少?
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index. # 返回的是一个视频的索引信息;
    """

    def sample_clips(
        video_size,
        action_start, # 目标视频的尺寸; 
        action_end, 
        clip_sizes, # 采样的到的clip尺寸; 
        clip_idx,  # 采集第几个clip?
        num_clips_uniform,
        min_delta=0,
        max_delta=math.inf,  # 如果采集多个CLIP, 那么他们之间的间隔最小不能太小, 最大不能太大; (间隔的计算方法: 末尾帧索引减去起始帧的索引)
        num_retries=100,
        use_offset=False,
        context_append=0
    ):
        se_inds = np.empty((0, 2))  # 形状初始化, 存储CLIP的起始帧和终止帧
        dt = np.empty((0))
        clip_indices = [] # 对应clip的indices; 
        
        action_length = action_end - action_start
        action_center = (action_end + action_start) / 2
    
        for clip_size in clip_sizes:  # 这几个clip的不同之处在于clip的长度不同;
            for i_try in range(num_retries):  # 尝试几次?
                
                if clip_idx == -1: # 训练集和验证集合都是如此; 
                    if clip_size > action_length: # 与目标clip互动; 
                        center_range = ((clip_size - action_length)/2)
                    else: 
                        center_range = -((clip_size - action_length)/2) # 这里确定的是
                    
                    context_append = context_append * action_length
                    center_range += context_append
                    clip_shift = np.random.randint(-center_range, center_range + 1) # 变成整数了
                    clip_center = clip_shift + action_center # 让实际的clip相对于动作片段的中心进行偏移; 
                
                elif clip_idx == -2: # extend the clip_idx==-2 for validation; 
                    pdb.set_trace()
                    clip_shift = 0 # 去除了视频采样的随机性; 
                    clip_center = clip_shift + action_center # 让实际的clip相对于动作片段的中心进行偏移; 

                else: # For test settings; 测试的时候不应该引入任何随机因素; 
                    
                    if clip_size > action_length: # 包含不同的上下文?(Multi-View得有区别才行;)
                        center_range = max(((clip_size - action_length)), 0)
                    else:
                        center_range = max(((-clip_size + action_length)), 0)# 表示一个偏移的范围; 
                    
                    context_append = context_append * action_length
                    center_range += context_append 

                    clip_gap = (center_range) / (num_clips_uniform - 1) # 10 个CLIP对应了9个GAP,
                    center_start = int(action_center - (center_range / 2)) # 中心点的变化范围
                    
                    clip_center = int(
                        round(clip_gap * clip_idx) + center_start, 
                    )

                index = np.arange(-clip_size//2, clip_size//2) # 这里取到了连续的视频片段； 
                index = clip_center + index
                
                index = np.clip(index, 0, video_size - 1).astype(np.int32) # 不要超出视频本身的长度; # 这里可以控制不让齐超过视频本身的长度; 
                # index = np.clip(index, action_start, action_end - 1).astype(np.int32) # 不要超出视频本身的长度; # 这里可以控制不让齐超过视频本身的长度; 
                clip_indices.append(index)
                start_idx = index[0]             
                end_idx = index[-1]

                se_inds_new = np.append(
                    se_inds, [[start_idx, end_idx]], axis=0
                )  # 起始终止的索引数值;
                # 每个clip仅出一个索引;
                if se_inds.shape[0] < 1:  # 如果始止点原来没有点, 那就直接终止
                    se_inds = se_inds_new
                    break

                se_inds_new = np.sort(se_inds_new, 0)  # 按照视频的起始帧进行对应的排序;
                t_start, t_end = (
                    se_inds_new[:, 0],
                    se_inds_new[:, 1],
                )  # 所有索引的起始帧;
                dt = t_start[1:] - t_end[:-1]  # 计算CLIP之间相隔的距离, 保证对比样本之间相似性和差异性维持在一个范围内; 
                if (
                    any(dt < min_delta) or any(dt > max_delta)
                ) and i_try < num_retries - 1:  # 直到采样都满足要求的时候才算是成功了; 
                    continue  # there is overlap CLIP 之间不能有重叠(不满足条件，需要重现尝试一次)
                else:
                    se_inds = se_inds_new
                    break  # 满足条件就退出循环;
        return se_inds, dt, clip_indices  # 返回这些clip的索引，以及这些索引

    # End of the function;
    num_retries, goodness = 100, -math.inf
    for _ in range(num_retries):  # 重复100多次;
        se_inds, dt, clip_indices = sample_clips(
            video_size,
            action_start, 
            action_end, 
            clip_sizes,
            clip_idx,
            num_clips_uniform,
            min_delta,
            max_delta,
            100,
            use_offset,
            context_append, # center_clip的偏移距离; 
        ) # 只有一个CLIPdt计算的到的长度为空集; 
        
        success = not (
            any(dt < min_delta) or any(dt > max_delta)
        )  # delta 代表的是什么?
        if success or clip_idx != -1:  # 训练集合、或者解码成功了;
            se_final, dt_final = se_inds, dt
            break
        else:
            cur_goodness = np.r_[dt[dt < min_delta], -dt[dt > max_delta]].sum()
            if goodness < cur_goodness:  # 保证程序是可以运行的; 问题是，没有额外的随机因素, 尝试再多次也没有用;
                se_final, dt_final = se_inds, dt
                goodness = cur_goodness

    delta_clips = np.concatenate((np.array([0]), dt_final))
    start_end_delta_time = np.c_[se_final, delta_clips]  # 转化成一个存储两个数字的np数组;
    return start_end_delta_time, clip_indices

## Define the Frame decode functions
def decode_frames(
    image_paths,
    action_se,
    sampling_rate,  # 视频采样帧率
    num_frames,  # 要多少帧
    clip_idx=-1,  # temporal_sample_index
    num_clips_uniform=10,  # 时间上采集多少帧?
    video_meta=None,
    fps=30,
    target_fps=30,  # 让视频适应不同的分辨率?
    max_spatial_scale=0,
    use_offset=False,  # 采样方法实际上是一致的, 只不过是计算方法不同;
    time_diff_prob=0.0,  # RGB 转换成时间Diff的概率?
    gaussian_prob=0.0,
    min_delta=-math.inf,
    max_delta=math.inf,  # clip之间的时间偏差是多少?
    temporally_rnd_clips=True,
    context_append = 0, 
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (list of ints): frame sampling rate (interval between two sampled
            frames).
        num_frames (list of ints): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips_uniform clips, and select the
            clip_idx-th video clip.
        num_clips_uniform (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        backend (str): decoding backend includes `pyav` and `torchvision`. The
            default one is `pyav`.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
    Returns:
        frames (tensor): decoded frames from the video.
    """
    # Currently support two decoders: 1) PyAV, and 2) TorchVision.
    assert clip_idx >= -2, "Not valied clip_idx {}".format(clip_idx) # extend clip_idx = -2 for validation
    assert len(sampling_rate) == len(num_frames)  # 对每个解码片段的采样帧率和采样帧数都有数值指定;
    num_decode = len(num_frames)  # 解码几个片段;
    num_frames_orig = num_frames
    num_retries = 10  # try times of the frame loading;

    if num_decode > 1 and temporally_rnd_clips:  # decode 出来的clip在时间上是有偏移的;
        ind_clips = np.random.permutation(num_decode)
        sampling_rate = [sampling_rate[i] for i in ind_clips]
        num_frames = [num_frames[i] for i in ind_clips]
    else:
        ind_clips = np.arange(
            num_decode
        )  # clips come temporally ordered from decoder

    clip_sizes = [
        np.maximum(1.0, sampling_rate[i] * num_frames[i] / target_fps * fps)
        for i in range(len(sampling_rate))
    ]  # 对应回原始视频的帧率, 然后按照目标帧率进行二次采样, 就得到了可以输入到网络里面的固定帧数;

    start_end_delta_time, clip_indices = get_multiple_start_end_idx(
        len(image_paths),  # CLIP, 视频的总帧数;
        action_se[0],
        action_se[1],
        clip_sizes,  # 多个片段, 每个片段所对应的帧数;
        clip_idx,  # 对应的clip的取值范围为0～10;
        num_clips_uniform,
        min_delta=min_delta,
        max_delta=max_delta,
        use_offset=use_offset,
        context_append=context_append, 
    )
    # load frames according to the image paths;
    
    frames_out, start_inds, time_diff_aug = (
        [None] * num_decode,
        [None] * num_decode,
        [None] * num_decode,
    )  # 解码多个clip的情况;

    augment_vid = (
        gaussian_prob > 0.0 or time_diff_prob > 0.0
    )  # 按照一定的概率进行数据增强(进行时间轴上的数据增强); Either one to be True. 
    
    for k in range(num_decode):  # 有几个clip要进行解码? 解码的clip到底是在进行什么操作?
        
        # Perform temporal sampling from the decoded video.
        start_idx, end_idx = (
            start_end_delta_time[k, 0],
            start_end_delta_time[k, 1],
        ) # 如果仅记忆起始帧和末尾帧的话, 就会丢失掉边界的重复帧所对应的信息; 

        clip_indice = clip_indices[k]
        
        frames = utils.retry_load_images(
            [
                image_paths[int(frame)] for frame in clip_indice
            ],  
            num_retries,
        ) # shape: torch.Size([149, 720, 1280, 3])
        

        if augment_vid:
            frames = frames.clone()
            frames, time_diff_aug[k] = transform.augment_raw_frames( # 是否是施加在原始的视频帧上? 
                frames, time_diff_prob, gaussian_prob
            )  # 原始帧和diff; (这个是在原始帧上计算还是在实际的视频上计算?)
        ## 在时间上如何进行采? 完成数据增强之后再对数据进行时间维度上的采样;
        ### 利用对应的索引将相关的视频帧检索出来;

        # frames_k = temporal_uniform_sampling(frames, start_idx, end_idx, T)
        frames_k = frames[::sampling_rate[k]]

        assert frames_k.shape[0] == num_frames[k]
        frames_out[k] = frames_k

    # if we shuffle, need to randomize the output, otherwise it will always be past->future
    # This part is the details of the Data Augmentation;
    if num_decode > 1 and temporally_rnd_clips:
        frames_out_, time_diff_aug_ = [None] * num_decode, [None] * num_decode
        start_end_delta_time_ = np.zeros_like(start_end_delta_time)
        for i, j in enumerate(ind_clips):  # 有几个?
            frames_out_[j] = frames_out[i]
            start_end_delta_time_[j, :] = start_end_delta_time[i, :]
            time_diff_aug_[j] = time_diff_aug[i]

        frames_out = frames_out_
        start_end_delta_time = start_end_delta_time_
        time_diff_aug = time_diff_aug_
        assert all(
            frames_out[i].shape[0] == num_frames_orig[i]
            for i in range(num_decode)
        )
    return frames_out, start_end_delta_time, time_diff_aug # 这个样本是否使用了? 



def decode_flows(
    image_paths,
    action_se,
    sampling_rate,  # 视频采样帧率
    num_frames,  # 要多少帧
    clip_idx=-1,  # temporal_sample_index
    num_clips_uniform=10,  # 时间上采集多少帧?
    video_meta=None,
    fps=30,
    target_fps=30,  # 让视频适应不同的分辨率?
    max_spatial_scale=0,
    use_offset=False,  # 采样方法实际上是一致的, 只不过是计算方法不同;
    time_diff_prob=0.0,  # RGB 转换成时间Diff的概率?
    gaussian_prob=0.0,
    min_delta=-math.inf,
    max_delta=math.inf,  # clip之间的时间偏差是多少?
    temporally_rnd_clips=True,
    context_append = 0, 
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (list of ints): frame sampling rate (interval between two sampled
            frames).
        num_frames (list of ints): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips_uniform clips, and select the
            clip_idx-th video clip.
        num_clips_uniform (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        backend (str): decoding backend includes `pyav` and `torchvision`. The
            default one is `pyav`.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
    Returns:
        frames (tensor): decoded frames from the video.
    """
    # Currently support two decoders: 1) PyAV, and 2) TorchVision.
    assert clip_idx >= -2, "Not valied clip_idx {}".format(clip_idx) # extend clip_idx = -2 for validation
    assert len(sampling_rate) == len(num_frames)  # 对每个解码片段的采样帧率和采样帧数都有数值指定;
    num_decode = len(num_frames)  # 解码几个片段;
    num_frames_orig = num_frames
    num_retries = 10  # try times of the frame loading;

    if num_decode > 1 and temporally_rnd_clips:  # decode 出来的clip在时间上是有偏移的;
        ind_clips = np.random.permutation(num_decode)
        sampling_rate = [sampling_rate[i] for i in ind_clips]
        num_frames = [num_frames[i] for i in ind_clips]
    else:
        ind_clips = np.arange(
            num_decode
        )  # clips come temporally ordered from decoder

    clip_sizes = [
        np.maximum(1.0, sampling_rate[i] * num_frames[i] / target_fps * fps)
        for i in range(len(sampling_rate))
    ]  # 对应回原始视频的帧率, 然后按照目标帧率进行二次采样, 就得到了可以输入到网络里面的固定帧数;

    start_end_delta_time, clip_indices = get_multiple_start_end_idx(
        len(image_paths),  # CLIP, 视频的总帧数;
        action_se[0],
        action_se[1],
        clip_sizes,  # 多个片段, 每个片段所对应的帧数;
        clip_idx,  # 对应的clip的取值范围为0～10;
        num_clips_uniform,
        min_delta=min_delta,
        max_delta=max_delta,
        use_offset=use_offset,
        context_append=context_append, 
    )
    # load frames according to the image paths;
    
    frames_out, start_inds, time_diff_aug = (
        [None] * num_decode,
        [None] * num_decode,
        [None] * num_decode,
    )  # 解码多个clip的情况;

    augment_vid = (
        gaussian_prob > 0.0 or time_diff_prob > 0.0
    )  # 按照一定的概率进行数据增强(进行时间轴上的数据增强); Either one to be True. 
    
    for k in range(num_decode):  # 有几个clip要进行解码? 解码的clip到底是在进行什么操作?
        
        # Perform temporal sampling from the decoded video.
        start_idx, end_idx = (
            start_end_delta_time[k, 0],
            start_end_delta_time[k, 1],
        ) # 如果仅记忆起始帧和末尾帧的话, 就会丢失掉边界的重复帧所对应的信息; 

        clip_indice = clip_indices[k]
        
        frames = utils.retry_load_flows(
            [
                image_paths[int(frame)] for frame in clip_indice
            ],  
            num_retries,
        ) # shape: torch.Size([149, 720, 1280, 3])
        
        if augment_vid:
            frames = frames.clone()
            frames, time_diff_aug[k] = transform.augment_raw_frames( # 是否是施加在原始的视频帧上? 
                frames, time_diff_prob, gaussian_prob
            )  # 原始帧和diff; (这个是在原始帧上计算还是在实际的视频上计算?)
        ## 在时间上如何进行采? 完成数据增强之后再对数据进行时间维度上的采样;
        ### 利用对应的索引将相关的视频帧检索出来;

        # frames_k = temporal_uniform_sampling(frames, start_idx, end_idx, T)
        frames_k = frames[::sampling_rate[k]]

        assert frames_k.shape[0] == num_frames[k]
        frames_out[k] = frames_k

    # if we shuffle, need to randomize the output, otherwise it will always be past->future
    # This part is the details of the Data Augmentation;
    if num_decode > 1 and temporally_rnd_clips:
        frames_out_, time_diff_aug_ = [None] * num_decode, [None] * num_decode
        start_end_delta_time_ = np.zeros_like(start_end_delta_time)
        for i, j in enumerate(ind_clips):  # 有几个?
            frames_out_[j] = frames_out[i]
            start_end_delta_time_[j, :] = start_end_delta_time[i, :]
            time_diff_aug_[j] = time_diff_aug[i]

        frames_out = frames_out_
        start_end_delta_time = start_end_delta_time_
        time_diff_aug = time_diff_aug_
        assert all(
            frames_out[i].shape[0] == num_frames_orig[i]
            for i in range(num_decode)
        )
    return frames_out, start_end_delta_time, time_diff_aug # 这个样本是否使用了? 
