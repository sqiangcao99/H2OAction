#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
from collections import defaultdict

from slowfast.utils.env import pathmgr

logger = logging.getLogger(__name__)

FPS = 30
AVA_VALID_FRAMES = range(902, 1799)  # 哪些是有效帧?


def load_image_lists(cfg, is_train):
    """
    Loading image paths from corresponding files.

    Args:
        cfg (CfgNode): config.
        is_train (bool): if it is training dataset or not.

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """
    list_filenames = [
        os.path.join(cfg.AVA.FRAME_LIST_DIR, filename)
        for filename in (
            cfg.AVA.TRAIN_LISTS if is_train else cfg.AVA.TEST_LISTS
        )
    ]
    image_paths = defaultdict(list)
    video_name_to_idx = {}
    video_idx_to_name = []
    for list_filename in list_filenames:
        with pathmgr.open(list_filename, "r") as f:  # loading annotations;
            f.readline()
            for line in f:
                row = line.split()  # load all the frames;
                # The format of each row should follow:
                # original_vido_id video_id frame_id path labels.
                assert len(row) == 5
                video_name = row[0]

                if video_name not in video_name_to_idx:  # 没有保存过的视频, 遍历读取每一行;
                    idx = len(video_name_to_idx)  # 往后排;
                    video_name_to_idx[video_name] = idx  # 存IDX;
                    video_idx_to_name.append(video_name)  # 双向保存;

                data_key = video_name_to_idx[video_name]  # 视频的索引;
                image_paths[data_key].append(
                    os.path.join(cfg.AVA.FRAME_DIR, row[3])
                )  # 把视频帧添加到对应的路径里面;

    image_paths = [
        image_paths[i] for i in range(len(image_paths))
    ]  # 列表, 视频索引到所有视频帧的映射;

    logger.info(
        "Finished loading image paths from: %s" % ", ".join(list_filenames)
    )

    return image_paths, video_idx_to_name  # 列表的每个元素保存了对应视频的所有帧, 列表对应的视频的名称;
    ## 仅保存路径就可以了;


def load_boxes_and_labels(cfg, mode):
    """
    Loading boxes and labels from csv files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test' mode.
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            [`box_coord`, `box_labels`] where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """
    gt_lists = cfg.AVA.TRAIN_GT_BOX_LISTS if mode == "train" else []  # 测试没有标签;
    pred_lists = (
        cfg.AVA.TRAIN_PREDICT_BOX_LISTS
        if mode == "train"
        else cfg.AVA.TEST_PREDICT_BOX_LISTS
    )
    ann_filenames = [
        os.path.join(cfg.AVA.ANNOTATION_DIR, filename)
        for filename in gt_lists + pred_lists
    ]
    ann_is_gt_box = [True] * len(gt_lists) + [False] * len(
        pred_lists
    )  # 可以两边都用？

    detect_thresh = cfg.AVA.DETECTION_SCORE_THRESH  # 置信度阈值框的过滤性问题;
    # Only select frame_sec % 4 = 0 samples for validation if not
    # set FULL_TEST_ON_VAL.
    boxes_sample_rate = (
        4 if mode == "val" and not cfg.AVA.FULL_TEST_ON_VAL else 1
    )  # 还需要有测试时候的帧率来决定的;
    all_boxes, count, unique_box_count = parse_bboxes_file(
        ann_filenames=ann_filenames,
        ann_is_gt_box=ann_is_gt_box,
        detect_thresh=detect_thresh,
        boxes_sample_rate=boxes_sample_rate,
    )
    logger.info(
        "Finished loading annotations from: %s" % ", ".join(ann_filenames)
    )
    logger.info("Detection threshold: {}".format(detect_thresh))
    logger.info("Number of unique boxes: %d" % unique_box_count)
    logger.info("Number of annotations: %d" % count)

    return all_boxes


def get_keyframe_data(boxes_and_labels):
    """
    Getting keyframe indices, boxes and labels in the dataset.

    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.

    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    def sec_to_frame(sec):
        """
        Convert time index (in second) to frame index.
        0: 900
        30: 901
        """
        return (sec - 900) * FPS  # 从第30秒开始计算时间索引?

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0

    for video_idx in range(len(boxes_and_labels)):  # 第几个视频;
        sec_idx = 0  # 单个视频内的关键帧的计数;
        keyframe_boxes_and_labels.append([])

        for sec in boxes_and_labels[video_idx].keys():  # 字典

            if sec not in AVA_VALID_FRAMES:  # 不是有效帧不读;
                continue

            if (
                len(boxes_and_labels[video_idx][sec]) > 0
            ):  # 遍历视频中的所有秒数, 如果有box, 就是关键帧;
                keyframe_indices.append(
                    (video_idx, sec_idx, sec, sec_to_frame(sec))
                )
                keyframe_boxes_and_labels[video_idx].append(
                    boxes_and_labels[video_idx][sec]
                )
                sec_idx += 1
                count += 1
    logger.info("%d keyframes used." % count)  # 有几个关键帧?

    return keyframe_indices, keyframe_boxes_and_labels


def get_num_boxes_used(keyframe_indices, keyframe_boxes_and_labels):
    """
    Get total number of used boxes.

    Args:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.

    Returns:
        count (int): total number of used boxes.
    """

    count = 0
    for video_idx, sec_idx, _, _ in keyframe_indices:
        count += len(keyframe_boxes_and_labels[video_idx][sec_idx])
    return count


def parse_bboxes_file(
    ann_filenames, ann_is_gt_box, detect_thresh, boxes_sample_rate=1
):
    """
    Parse AVA bounding boxes files.
    Args:
        ann_filenames (list of str(s)): a list of AVA bounding boxes annotation files.
        ann_is_gt_box (list of bools): a list of boolean to indicate whether the corresponding
            ann_file is ground-truth. `ann_is_gt_box[i]` correspond to `ann_filenames[i]`.
        detect_thresh (float): threshold for accepting predicted boxes, range [0, 1].
        boxes_sample_rate (int): sample rate for test bounding boxes. Get 1 every `boxes_sample_rate`.
    """
    all_boxes = {}
    count = 0
    unique_box_count = 0
    for filename, is_gt_box in zip(ann_filenames, ann_is_gt_box):
        with pathmgr.open(filename, "r") as f:
            for line in f:
                row = line.strip().split(",")
                # When we use predicted boxes to train/eval, we need to
                # ignore the boxes whose scores are below the threshold.
                if not is_gt_box:
                    score = float(row[7])
                    if score < detect_thresh:
                        continue

                video_name, frame_sec = row[0], int(row[1])  # 对应帧所对应的秒数;
                if frame_sec % boxes_sample_rate != 0:  # 原始视频帧进行一个采样; 标签采样;
                    continue

                # Box with format [x1, y1, x2, y2] with a range of [0, 1] as float.
                box_key = ",".join(row[2:6])
                box = list(map(float, row[2:6]))
                label = -1 if row[6] == "" else int(row[6])

                if video_name not in all_boxes:
                    all_boxes[video_name] = {}
                    for sec in AVA_VALID_FRAMES:
                        all_boxes[video_name][sec] = {}  # 每一帧可能有若干个标签;

                if box_key not in all_boxes[video_name][frame_sec]:
                    all_boxes[video_name][frame_sec][box_key] = [box, []]
                    unique_box_count += 1  # box 可能存在重复?

                all_boxes[video_name][frame_sec][box_key][1].append(
                    label
                )  # 保存标签的类别和坐标;
                if label != -1:
                    count += 1

    for video_name in all_boxes.keys():
        for frame_sec in all_boxes[video_name].keys():
            # Save in format of a list of [box_i, box_i_labels].
            all_boxes[video_name][frame_sec] = list(
                all_boxes[video_name][frame_sec].values()
            )

    return all_boxes, count, unique_box_count  # 视频->帧->帧所对应的所有box信息;
