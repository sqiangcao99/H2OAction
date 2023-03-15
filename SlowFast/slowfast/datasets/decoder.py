#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import math
import numpy as np
import random
import torch
import torchvision.io as io

from . import transform as transform
from . import utils as utils

logger = logging.getLogger(__name__)


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
    clip_sizes,  # 按照原始的帧率进行采样;
    clip_idx,  # 第几个ID;
    num_clips_uniform,  # 均匀采样总共几个视频片段; 这里的时间采样采用的是均匀采样策略; 且最终只返回一个视频clip;
    min_delta=0,  # 视频片段之间的最小间隔;
    max_delta=math.inf,  # 视频片段的最大间隔
    use_offset=False,  #
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
        clip_sizes,
        clip_idx,  # 采集第几个clip?
        num_clips_uniform,
        min_delta=0,
        max_delta=math.inf,  # 如果采集多个CLIP, 那么他们之间的间隔最小不能太小, 最大不能太大; (间隔的计算方法: 末尾帧索引减去起始帧的索引)
        num_retries=100,
        use_offset=False,
    ):
        se_inds = np.empty((0, 2))  # 形状初始化, 存储CLIP的起始帧和终止帧
        dt = np.empty((0))
        for clip_size in clip_sizes:  # 这几个clip的不同之处在于clip的长度不同;
            for i_try in range(num_retries):  # 尝试几次?
                # clip_size = int(clip_size)
                max_start = max(video_size - clip_size, 0)
                if clip_idx == -1:
                    # Random temporal sampling.
                    start_idx = random.uniform(
                        0, max_start
                    )  # 帧索引的最大数值? 保证能够采集到一个CLIP的最大起始帧的索引?
                else:  # Uniformly sample the clip with the given index.
                    if use_offset:  #
                        if (
                            num_clips_uniform == 1
                        ):  # 只从视频中采集一帧的情况? 这里相当于是把特殊情况列写出来了;
                            # Take the center clip if num_clips is 1.
                            start_idx = math.floor(max_start / 2)
                        else:
                            start_idx = clip_idx * math.floor(
                                max_start / (num_clips_uniform - 1)
                            )
                    else:
                        start_idx = (
                            max_start * clip_idx / num_clips_uniform
                        )  # 采集的是哪一帧;
                        ## 这两种采样方式实际上是等价的, 一种计算clip之间的stride(确定偏移之后, 在进行递归); 另外一种方法, 则是直接均匀摆放起始点的位置;
                end_idx = start_idx + clip_size - 1  # 确定了起始帧之后, 末帧根据CLIP的长度来确定;
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
                dt = t_start[1:] - t_end[:-1]  # 计算clip之间的间隔?
                if (
                    any(dt < min_delta) or any(dt > max_delta)
                ) and i_try < num_retries - 1:  # 直到采样都满足要求的时候才算是成功了;
                    continue  # there is overlap CLIP 之间不能有重叠(不满足条件，需要重现尝试一次)
                else:
                    se_inds = se_inds_new
                    break  # 满足条件就退出循环;
        return se_inds, dt  # 返回这些clip的索引，以及这些索引

    # End of the function;

    num_retries, goodness = 100, -math.inf
    for _ in range(num_retries):  # 重复100多次;
        se_inds, dt = sample_clips(
            video_size,
            clip_sizes,
            clip_idx,
            num_clips_uniform,
            min_delta,
            max_delta,
            100,
            use_offset,
        )
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
    ## ??
    return start_end_delta_time


def pyav_decode_stream(
    container, start_pts, end_pts, stream, stream_name, buffer_size=0
):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
        buffer_size (int): number of additional frames to decode beyond end_pts.
    Returns:
        result (list): list of frames decoded.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """
    # Seeking in the stream is imprecise. Thus, seek to an ealier PTS by a
    # margin pts.
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts


def torchvision_decode(
    video_handle,
    sampling_rate,
    num_frames,
    clip_idx,
    video_meta,
    num_clips_uniform=10,
    target_fps=30,
    modalities=("visual",),
    max_spatial_scale=0,
    use_offset=False,
    min_delta=-math.inf,
    max_delta=math.inf,
):
    """
    If video_meta is not empty, perform temporal selective decoding to sample a
    clip from the video with TorchVision decoder. If video_meta is empty, decode
    the entire video and update the video_meta.
    Args:
        video_handle (bytes): raw bytes of the video file.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips_uniform clips, and select the clip_idx-th video clip.
        video_meta (dict): a dict contains VideoMetaData. Details can be found
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps.
        modalities (tuple): tuple of modalities to decode. Currently only
            support `visual`, planning to support `acoustic` soon.
        max_spatial_scale (int): the resolution of the spatial shorter
            edge size during decoding.
        min_delta (int): minimum distance between clips when sampling multiple.
        max_delta (int): max distance between clips when sampling multiple.
    Returns:
        frames (tensor): decoded frames from the video.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): if True, the entire video was decoded.
    """
    # Convert the bytes to a tensor.
    video_tensor = torch.from_numpy(np.frombuffer(video_handle, dtype=np.uint8))

    decode_all_video = True
    video_start_pts, video_end_pts = 0, -1
    # The video_meta is empty, fetch the meta data from the raw video.
    if len(video_meta) == 0:
        # Tracking the meta info for selective decoding in the future.
        meta = io._probe_video_from_memory(video_tensor)
        # Using the information from video_meta to perform selective decoding.
        video_meta["video_timebase"] = meta.video_timebase
        video_meta["video_numerator"] = meta.video_timebase.numerator
        video_meta["video_denominator"] = meta.video_timebase.denominator
        video_meta["has_video"] = meta.has_video
        video_meta["video_duration"] = meta.video_duration
        video_meta["video_fps"] = meta.video_fps
        video_meta["audio_timebas"] = meta.audio_timebase
        video_meta["audio_numerator"] = meta.audio_timebase.numerator
        video_meta["audio_denominator"] = meta.audio_timebase.denominator
        video_meta["has_audio"] = meta.has_audio
        video_meta["audio_duration"] = meta.audio_duration
        video_meta["audio_sample_rate"] = meta.audio_sample_rate

    fps = video_meta["video_fps"]

    if len(video_meta) > 0 and (
        video_meta["has_video"]
        and video_meta["video_denominator"] > 0
        and video_meta["video_duration"] > 0
        and fps * video_meta["video_duration"]
        > sum(T * tau for T, tau in zip(num_frames, sampling_rate))
    ):  # 视频总共的帧数有多少?;
        decode_all_video = False  # try selective decoding,

        clip_sizes = [
            np.maximum(1.0, sampling_rate[i] * num_frames[i] / target_fps * fps)
            for i in range(len(sampling_rate))
        ]
        start_end_delta_time = get_multiple_start_end_idx(
            fps * video_meta["video_duration"],
            clip_sizes,
            clip_idx,
            num_clips_uniform,
            min_delta=min_delta,
            max_delta=max_delta,
            use_offset=use_offset,
        )  # 得到采样对应的相关索引数值;
        frames_out = [None] * len(num_frames)
        for k in range(len(num_frames)):
            pts_per_frame = (
                video_meta["video_denominator"] / video_meta["video_fps"]
            )
            video_start_pts = int(start_end_delta_time[k, 0] * pts_per_frame)
            video_end_pts = int(
                start_end_delta_time[k, 1] * pts_per_frame
            )  # 这是一个数字;

            # Decode the raw video with the tv decoder.
            v_frames, _ = io._read_video_from_memory(
                video_tensor,
                seek_frame_margin=1.0,
                read_video_stream="visual" in modalities,
                video_width=0,
                video_height=0,
                video_min_dimension=max_spatial_scale,
                video_pts_range=(video_start_pts, video_end_pts),
                video_timebase_numerator=video_meta["video_numerator"],
                video_timebase_denominator=video_meta["video_denominator"],
                read_audio_stream=0,
            )  # here retriave the raw video from the dataset;
            if v_frames is None or v_frames.shape == torch.Size([0]):
                decode_all_video = True
                logger.info("TV decode FAILED try decode all")
                break
            frames_out[k] = v_frames

    # 优先尝试的是对视频进行局部的解码;
    if decode_all_video:
        # failed selective decoding, 两者在函数调用上的区别是什么?
        decode_all_video = True
        video_start_pts, video_end_pts = 0, -1  # 定义了视频的起始位置和终止位置,
        start_end_delta_time = (
            None  # 不规定视频的起始位置和终止位置; 因为selective decode也可能解码失败, 因此外面需要再尝试一次；
        )
        v_frames, _ = io._read_video_from_memory(
            video_tensor,
            seek_frame_margin=1.0,
            read_video_stream="visual" in modalities,
            video_width=0,
            video_height=0,
            video_min_dimension=max_spatial_scale,
            video_pts_range=(video_start_pts, video_end_pts),
            video_timebase_numerator=video_meta["video_numerator"],
            video_timebase_denominator=video_meta["video_denominator"],
            read_audio_stream=0,
        )
        if v_frames.shape == torch.Size([0]):
            v_frames = None
            logger.info(
                "TV decode FAILED try decode all"
            )  # 没有解码出视频帧来, 因此需要重新解码;

        frames_out = [v_frames]

    if any([t.shape[0] < 0 for t in frames_out]):
        frames_out = [None]
        logger.info("TV decode FAILED: Decoded empty video")

    # 这个函数的目的尝试从视频中直接docode出相应的片段, 但是有可能会失败, 失败的话，就返回所有的视频帧, 并且从视频的外部进行帧数划分；
    return frames_out, fps, decode_all_video, start_end_delta_time


def pyav_decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx,
    num_clips_uniform=10,
    target_fps=30,
    use_offset=False,
):
    """
    Convert the video from its original fps to the target_fps. If the video
    support selective decoding (contain decoding information in the video head),
    the perform temporal selective decoding and sample a clip from the video
    with the PyAV decoder. If the video does not support selective decoding,
    decode the entire video.

    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames.
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips_uniform
            clips, and select the clip_idx-th video clip.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): If True, the entire video was decoded.
    """
    # Try to fetch the decoding information from the video head. Some of the
    # videos does not support fetching the decoding information, for that case
    # it will get None duration.
    fps = float(container.streams.video[0].average_rate)
    frames_length = container.streams.video[0].frames
    duration = container.streams.video[0].duration

    if duration is None:
        # If failed to fetch the decoding information, decode the entire video.
        decode_all_video = True
        video_start_pts, video_end_pts = 0, math.inf
    else:
        # Perform selective decoding.
        decode_all_video = False
        clip_size = np.maximum(
            1.0,
            np.ceil(
                sampling_rate * (num_frames - 1) / target_fps * fps
            ),  # 采样应该在目标帧率下进行, 计算得到的帧数是在实际帧数下的帧率;
        )

        start_idx, end_idx, fraction = get_start_end_idx(
            frames_length,
            clip_size,
            clip_idx,
            num_clips_uniform,
            use_offset=use_offset,
        )
        timebase = duration / frames_length
        video_start_pts = int(start_idx * timebase)
        video_end_pts = int(end_idx * timebase)

    frames = None
    # If video stream was found, fetch video frames from the video.
    if container.streams.video:
        video_frames, max_pts = pyav_decode_stream(
            container,
            video_start_pts,
            video_end_pts,
            container.streams.video[0],
            {"video": 0},
        )
        container.close()

        frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
        frames = torch.as_tensor(np.stack(frames))
    return frames, fps, decode_all_video


def decode(
    container,  # 视频容器对象
    sampling_rate,  # 视频采样帧率
    num_frames,  # 要多少帧
    clip_idx=-1,  # temporal_sample_index
    num_clips_uniform=10,  # 时间上采集多少帧?
    video_meta=None,
    target_fps=30,  # 让视频适应不同的分辨率?
    backend="pyav",
    max_spatial_scale=0,
    use_offset=False,  # 采样方法实际上是一致的, 只不过是计算方法不同;
    time_diff_prob=0.0,  # RGB 转换成时间Diff的概率?
    gaussian_prob=0.0,
    min_delta=-math.inf,
    max_delta=math.inf,  # clip之间的时间偏差是多少?
    temporally_rnd_clips=True,
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
    assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)
    assert len(sampling_rate) == len(num_frames)  # 对每个解码片段的采样帧率和采样帧数都有数值指定;

    num_decode = len(num_frames)  # 解码几个片段;
    num_frames_orig = num_frames

    if num_decode > 1 and temporally_rnd_clips:  # decode 出来的clip在时间上是有偏移的;
        ind_clips = np.random.permutation(num_decode)
        sampling_rate = [sampling_rate[i] for i in ind_clips]
        num_frames = [num_frames[i] for i in ind_clips]
    else:
        ind_clips = np.arange(
            num_decode
        )  # clips come temporally ordered from decoder
    try:
        if backend == "pyav":
            assert (
                min_delta == -math.inf and max_delta == math.inf
            ), "delta sampling not supported in pyav"
            frames_decoded, fps, decode_all_video = pyav_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                num_clips_uniform,
                target_fps,
                use_offset=use_offset,
            )
        elif backend == "torchvision":
            (
                frames_decoded,
                fps,
                decode_all_video,
                start_end_delta_time,
            ) = torchvision_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,  # 这个参数是怎么使用的?
                video_meta,
                num_clips_uniform,
                target_fps,  # 目标帧率是多少?
                ("visual",),
                max_spatial_scale,
                use_offset=use_offset,
                min_delta=min_delta,
                max_delta=max_delta,
            )
        else:
            raise NotImplementedError(
                "Unknown decoding backend {}".format(backend)
            )
    except Exception as e:
        print("Failed to decode by {} with exception: {}".format(backend, e))
        return None, None, None

    # Return None if the frames was not decoded successfully.
    if frames_decoded is None or None in frames_decoded:
        return None, None, None

    if not isinstance(frames_decoded, list):
        frames_decoded = [frames_decoded]  # 几个片段的解码策略是按照列表进行顺序存放的;
    num_decoded = len(
        frames_decoded
    )  # 实际解码出来了几个clip? 因为默认是想直接从视频中解码出对应的片段来的; 选择性的解码失败之后, 返回的就是完整视频的解码结果;
    clip_sizes = [
        np.maximum(1.0, sampling_rate[i] * num_frames[i] / target_fps * fps)
        for i in range(len(sampling_rate))
    ]  # 对应回原始视频的帧率, 然后按照目标帧率进行二次采样, 就得到了可以输入到网络里面的固定帧数;

    if decode_all_video:  # full video was decoded (not trimmed yet) 解码完整的视频
        assert (
            num_decoded == 1 and start_end_delta_time is None
        )  ## 证明解码视频的时候失败了, 因此这里需要针对视频帧进行采样;
        start_end_delta_time = get_multiple_start_end_idx(
            frames_decoded[0].shape[0],  # 多少帧?
            clip_sizes,  # 多个片段, 每个片段所对应的帧数;
            clip_idx if decode_all_video else 0,  # 对应的clip的取值范围为0～10;
            num_clips_uniform if decode_all_video else 1,
            min_delta=min_delta,
            max_delta=max_delta,
            use_offset=use_offset,
        )

    frames_out, start_inds, time_diff_aug = (
        [None] * num_decode,
        [None] * num_decode,
        [None] * num_decode,
    )  # 解码多个clip的情况;

    augment_vid = gaussian_prob > 0.0 or time_diff_prob > 0.0  # 按照一定的概率进行数据增强;
    for k in range(num_decode):  # 有几个clip要进行解码? 解码的clip到底是在进行什么操作?
        T = num_frames[k]  # 有几帧?
        # Perform temporal sampling from the decoded video.

        if (
            decode_all_video
        ):  # Operates on the whole frames of the original videos;
            frames = frames_decoded[0]  # 一个视频所对应的所有视频帧;
            if augment_vid:
                frames = frames.clone()
            start_idx, end_idx = (
                start_end_delta_time[k, 0],
                start_end_delta_time[k, 1],
            )
        else:
            frames = frames_decoded[k]
            # video is already trimmed so we just need subsampling # 按照测试时增强的策略在时间维度上进行采样;
            (
                start_idx,
                end_idx,
                clip_position,
            ) = get_start_end_idx(  # 下采样的具体操作是什么?
                frames.shape[0], clip_sizes[k], 0, 1
            )  # 这个是啥? 如何处理的?
        if augment_vid:
            frames, time_diff_aug[k] = transform.augment_raw_frames(
                frames, time_diff_prob, gaussian_prob
            )  # 原始帧和diff;
        ## 在时间上如何进行采? 完成数据增强之后再对数据进行时间维度上的采样;
        ### 利用对应的索引将相关的视频帧检索出来;

        frames_k = temporal_sampling(
            frames, start_idx, end_idx, T
        )  # 看懂几个关键的采样函数就可以了;
        frames_out[
            k
        ] = frames_k  # 得到第几个clip的视频帧; 可能会有多个clip的情况;(多个clip的时间索引是否是一致的?)

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

    return frames_out, start_end_delta_time, time_diff_aug


## Define the Frame decode functions
def decode_frames(
    sampling_rate,  # 视频采样帧率
    num_frames,  # 要多少帧
    clip_idx=-1,  # temporal_sample_index
    num_clips_uniform=10,  # 时间上采集多少帧?
    video_meta=None,
    target_fps=30,  # 让视频适应不同的分辨率?
    backend="pyav",
    max_spatial_scale=0,
    use_offset=False,  # 采样方法实际上是一致的, 只不过是计算方法不同;
    time_diff_prob=0.0,  # RGB 转换成时间Diff的概率?
    gaussian_prob=0.0,
    min_delta=-math.inf,
    max_delta=math.inf,  # clip之间的时间偏差是多少?
    temporally_rnd_clips=True,
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
    assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)
    assert len(sampling_rate) == len(num_frames)  # 对每个解码片段的采样帧率和采样帧数都有数值指定;

    num_decode = len(num_frames)  # 解码几个片段;
    num_frames_orig = num_frames

    if num_decode > 1 and temporally_rnd_clips:  # decode 出来的clip在时间上是有偏移的;
        ind_clips = np.random.permutation(num_decode)
        sampling_rate = [sampling_rate[i] for i in ind_clips]
        num_frames = [num_frames[i] for i in ind_clips]
    else:
        ind_clips = np.arange(
            num_decode
        )  # clips come temporally ordered from decoder
    try:
        if backend == "pyav":
            assert (
                min_delta == -math.inf and max_delta == math.inf
            ), "delta sampling not supported in pyav"
            frames_decoded, fps, decode_all_video = pyav_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                num_clips_uniform,
                target_fps,
                use_offset=use_offset,
            )
        elif backend == "torchvision":
            (
                frames_decoded,
                fps,
                decode_all_video,
                start_end_delta_time,
            ) = torchvision_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,  # 这个参数是怎么使用的?
                video_meta,
                num_clips_uniform,
                target_fps,  # 目标帧率是多少?
                ("visual",),
                max_spatial_scale,
                use_offset=use_offset,
                min_delta=min_delta,
                max_delta=max_delta,
            )
        else:
            raise NotImplementedError(
                "Unknown decoding backend {}".format(backend)
            )
    except Exception as e:
        print("Failed to decode by {} with exception: {}".format(backend, e))
        return None, None, None

    # Return None if the frames was not decoded successfully.
    if frames_decoded is None or None in frames_decoded:
        return None, None, None

    if not isinstance(frames_decoded, list):
        frames_decoded = [frames_decoded]  # 几个片段的解码策略是按照列表进行顺序存放的;
    num_decoded = len(
        frames_decoded
    )  # 实际解码出来了几个clip? 因为默认是想直接从视频中解码出对应的片段来的; 选择性的解码失败之后, 返回的就是完整视频的解码结果;
    clip_sizes = [
        np.maximum(1.0, sampling_rate[i] * num_frames[i] / target_fps * fps)
        for i in range(len(sampling_rate))
    ]  # 对应回原始视频的帧率, 然后按照目标帧率进行二次采样, 就得到了可以输入到网络里面的固定帧数;

    if decode_all_video:  # full video was decoded (not trimmed yet) 解码完整的视频
        assert (
            num_decoded == 1 and start_end_delta_time is None
        )  ## 证明解码视频的时候失败了, 因此这里需要针对视频帧进行采样;
        start_end_delta_time = get_multiple_start_end_idx(
            frames_decoded[0].shape[0],  # 多少帧?
            clip_sizes,  # 多个片段, 每个片段所对应的帧数;
            clip_idx if decode_all_video else 0,  # 对应的clip的取值范围为0～10;
            num_clips_uniform if decode_all_video else 1,
            min_delta=min_delta,
            max_delta=max_delta,
            use_offset=use_offset,
        )

    frames_out, start_inds, time_diff_aug = (
        [None] * num_decode,
        [None] * num_decode,
        [None] * num_decode,
    )  # 解码多个clip的情况;

    augment_vid = gaussian_prob > 0.0 or time_diff_prob > 0.0  # 按照一定的概率进行数据增强;
    for k in range(num_decode):  # 有几个clip要进行解码? 解码的clip到底是在进行什么操作?
        T = num_frames[k]  # 有几帧?
        # Perform temporal sampling from the decoded video.

        if (
            decode_all_video
        ):  # Operates on the whole frames of the original videos;
            frames = frames_decoded[0]  # 一个视频所对应的所有视频帧;
            if augment_vid:
                frames = frames.clone()
            start_idx, end_idx = (
                start_end_delta_time[k, 0],
                start_end_delta_time[k, 1],
            )
        else:
            frames = frames_decoded[k]
            # video is already trimmed so we just need subsampling # 按照测试时增强的策略在时间维度上进行采样;
            (
                start_idx,
                end_idx,
                clip_position,
            ) = get_start_end_idx(  # 下采样的具体操作是什么?
                frames.shape[0], clip_sizes[k], 0, 1
            )  # 这个是啥? 如何处理的?
        if augment_vid:
            frames, time_diff_aug[k] = transform.augment_raw_frames(
                frames, time_diff_prob, gaussian_prob
            )  # 原始帧和diff;
        ## 在时间上如何进行采? 完成数据增强之后再对数据进行时间维度上的采样;
        ### 利用对应的索引将相关的视频帧检索出来;

        frames_k = temporal_sampling(
            frames, start_idx, end_idx, T
        )  # 看懂几个关键的采样函数就可以了;
        frames_out[
            k
        ] = frames_k  # 得到第几个clip的视频帧; 可能会有多个clip的情况;(多个clip的时间索引是否是一致的?)

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

    return frames_out, start_end_delta_time, time_diff_aug
