#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import copy
import numpy as np
import random
import os
import pandas as pd
import torch
import torch.utils.data
from torchvision import transforms

import slowfast.utils.logging as logging

from .build import DATASET_REGISTRY
from .epickitchens_record import EpicKitchensVideoRecord

from . import autoaugment as autoaugment
from . import transform as transform
from . import utils as utils
from .frame_loader import pack_frames_to_video_clip

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Drive_and_act(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):

        assert mode in [
            "train",
            "val",
            "test",
            "train+val"
        ], "Split '{}' not supported for EPIC-KITCHENS".format(mode)
        self.cfg = cfg
        self.mode = mode
        self.target_fps = cfg.DATA.TARGET_FPS
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing EPIC-KITCHENS {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == "train":
            path_annotations_pickle = [
                os.path.join(self.cfg.Drive_and_act.ANNOTATIONS_DIR, self.cfg.Drive_and_act.TRAIN_LIST)]
            if self.cfg.MVIT.INPUT_MODAL == 'RGBIR':
                prompt_path_annotations_pickle = [
                    os.path.join(self.cfg.Drive_and_act.PROMPT_DATA_DIR, self.cfg.Drive_and_act.TRAIN_LIST)]
        elif self.mode == "val":
            path_annotations_pickle = [
                os.path.join(self.cfg.Drive_and_act.ANNOTATIONS_DIR, self.cfg.Drive_and_act.VAL_LIST)]
            if self.cfg.MVIT.INPUT_MODAL == 'RGBIR':
                prompt_path_annotations_pickle = [
                    os.path.join(self.cfg.Drive_and_act.PROMPT_DATA_DIR, self.cfg.Drive_and_act.VAL_LIST)]
        elif self.mode == "test":
            path_annotations_pickle = [
                os.path.join(self.cfg.Drive_and_act.ANNOTATIONS_DIR, self.cfg.Drive_and_act.TEST_LIST)]
            if self.cfg.MVIT.INPUT_MODAL == 'RGBIR':
                prompt_path_annotations_pickle = [
                    os.path.join(self.cfg.Drive_and_act.PROMPT_DATA_DIR, self.cfg.Drive_and_act.TEST_LIST)]
        else:
            path_annotations_pickle = [
                os.path.join(self.cfg.Drive_and_act.ANNOTATIONS_DIR, file)
                    for file in [self.cfg.Drive_and_act.TRAIN_LIST, self.cfg.Drive_and_act.VAL_LIST]]

        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(
                file
            )

        self._video_records = []
        self._prompt_records = []
        if prompt_path_annotations_pickle is not None:
            for file in prompt_path_annotations_pickle:
                with open(file, 'r') as fin:
                    for line in fin:
                        line_split = line.strip().split()
                        video_info = {}
                        idx = 0
                        # idx for frame_dir
                        frame_dir = line_split[idx]
                        video_info['frame_dir'] = frame_dir
                        idx += 1
                        video_info['offset'] = int(line_split[idx])
                        video_info['total_frames'] = int(line_split[idx + 1])
                        idx += 2
                        label = [int(x) for x in line_split[idx:]]
                        video_info['label'] = label[0]
                        self._prompt_records.append(video_info)
        for file in path_annotations_pickle:
            with open(file, 'r') as fin:
                for line in fin:
                    line_split = line.strip().split()
                    video_info = {}
                    idx = 0
                    # idx for frame_dir
                    frame_dir = line_split[idx]
                    video_info['frame_dir'] = frame_dir
                    idx += 1
                    video_info['offset'] = int(line_split[idx])
                    video_info['total_frames'] = int(line_split[idx + 1])
                    idx += 2
                    label = [int(x) for x in line_split[idx:]]
                    video_info['label'] = label[0]
                    self._video_records.append(video_info)
        assert (
                len(self._video_records) > 0
        ), "Failed to load Drive-act split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing Drive_and_act dataloader (size: {}) from {}".format(
                len(self._video_records), path_annotations_pickle
            )
        )

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
        if self.cfg.Drive_and_act.TRAIN_MODE == 'pretrain_event':
            inputs=list()
            if self.mode in ["train", "val", "train+val"]:
                # -1 indicates random sampling.
                temporal_sample_index = -1
                spatial_sample_index = -1
                min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
                max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
                crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            elif self.mode in ["test"]:
                # temporal_sample_index = (
                #     self._video_records[index]
                #     // self.cfg.TEST.NUM_SPATIAL_CROPS
                # )
                # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
                # center, or right if width is larger than height, and top, middle,
                # or bottom if height is larger than width.
                if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                    spatial_sample_index = 3
                    # spatial_sample_index = (
                    #     self._video_records[index]
                    #     % self.cfg.TEST.NUM_SPATIAL_CROPS
                    # )
                elif self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                    spatial_sample_index = 1
                min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
                # The testing is deterministic and no jitter should be performed.
                # min_scale, max_scale, and crop_size are expect to be the same.
                assert len({min_scale, max_scale, crop_size}) == 1
            else:
                raise NotImplementedError(
                    "Does not support {} mode".format(self.mode)
                )
            results = copy.deepcopy(self._video_records[index])
            results['event_tmpl']='event{:06d}.npy'
            results['event_path'] = self.cfg.Drive_and_act.EVENT_PATH
            events = pack_frames_to_video_clip(self.cfg, results,train_mode=self.cfg.Drive_and_act.TRAIN_MODE, target_fps=self.target_fps,input_modal=self.cfg.MVIT.INPUT_MODAL)
            if self.cfg.DATA.USE_RAND_AUGMENT and self.mode in ["train"]:
                img_size_min = crop_size
                auto_augment_desc = "rand-m15-mstd0.5-inc1"
                aa_params = dict(
                    translate_const=int(img_size_min * 0.45),
                    img_mean=tuple([min(255, round(255 * x)) for x in self.cfg.DATA.MEAN]),
                )
                seed = random.randint(0, 100000000)
                events = [autoaugment.rand_augment_transform(
                    auto_augment_desc, aa_params, seed)(event) for event in events]
            events = [torch.tensor(np.array(event)) for event in events]
            events = torch.stack(events)
            events = utils.tensor_normalize(
                events, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )
            # T H W C -> C T H W.
            events = events.permute(3, 0, 1, 2)
            
            # Perform data augmentation.
            use_random_resize_crop = self.cfg.DATA.USE_RANDOM_RESIZE_CROPS
            if use_random_resize_crop:
                if self.mode in ["train", "val"]:
                    frames = transform.random_resize_crop_video(frames, crop_size, interpolation_mode="bilinear")
                    frames, _ = transform.horizontal_flip(0.5, frames)
                else:
                    assert len({min_scale, max_scale, crop_size}) == 1
                    frames, _ = transform.random_short_side_scale_jitter(
                        frames, min_scale, max_scale
                    )
                    frames, _ = transform.uniform_crop(frames, crop_size, spatial_sample_index)
            else:
                # Perform data augmentation.
                events = utils.spatial_sampling(
                    events,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )     
            # T H W C -> T C H W.
            if self.mode in ["train", "val"]:
                events = events.permute(1, 0, 2, 3) # C T H W -> T C H W
                events = utils.frames_augmentation(
                    events,
                    colorjitter=None,
                    use_grayscale=None,
                    use_gaussian=None
                )
            label = self._video_records[index]['label']
            inputs.append(events)
            metadata = {}
            return inputs, label, index, metadata
        else:
            if self.mode in ["train", "val", "train+val"]:
                # -1 indicates random sampling.
                temporal_sample_index = -1
                spatial_sample_index = -1
                min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
                max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
                crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            elif self.mode in ["test"]:
                # temporal_sample_index = (
                #     self._video_records[index]
                #     // self.cfg.TEST.NUM_SPATIAL_CROPS
                # )
                # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
                # center, or right if width is larger than height, and top, middle,
                # or bottom if height is larger than width.
                if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                    spatial_sample_index = 3
                    # spatial_sample_index = (
                    #     self._video_records[index]
                    #     % self.cfg.TEST.NUM_SPATIAL_CROPS
                    # )
                elif self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                    spatial_sample_index = 1
                min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
                # The testing is deterministic and no jitter should be performed.
                # min_scale, max_scale, and crop_size are expect to be the same.
                assert len({min_scale, max_scale, crop_size}) == 1
            else:
                raise NotImplementedError(
                    "Does not support {} mode".format(self.mode)
                )
            results = copy.deepcopy(self._video_records[index])
            prompt_results = copy.deepcopy(self._prompt_records[index])
            if self.cfg.Drive_and_act.EVENT_PATH is not None:
                if self.cfg.MVIT.INPUT_MODAL== 'RGBIR':
                    prompt_results['prompt_tmpl']='img_{:05d}.jpg' #'event{:06d}.npy'
                    frames,prompts = pack_frames_to_video_clip(self.cfg, results,train_mode=self.cfg.Drive_and_act.TRAIN_MODE, target_fps=self.target_fps,input_modal=self.cfg.MVIT.INPUT_MODAL,prompt_record=prompt_results)
                else:
                    results['event_tmpl']='event{:06d}.npy' #'event{:06d}.npy'
                    results['event_path'] = self.cfg.Drive_and_act.EVENT_PATH
                    frames,prompts = pack_frames_to_video_clip(self.cfg, results,train_mode=self.cfg.Drive_and_act.TRAIN_MODE, target_fps=self.target_fps,input_modal=self.cfg.MVIT.INPUT_MODAL)
            # frames [T,256,456,3]
            else:
                frames = pack_frames_to_video_clip(self.cfg, results,train_mode=self.cfg.Drive_and_act.TRAIN_MODE,target_fps=self.target_fps,input_modal=self.cfg.MVIT.INPUT_MODAL)

            if self.cfg.DATA.USE_RAND_AUGMENT and self.mode in ["train"]:
                # Transform to PIL Image
                frames = [transforms.ToPILImage()(frame.squeeze().numpy()) for frame in frames]

                # Perform RandAugment
                img_size_min = crop_size
                auto_augment_desc = "rand-m15-mstd0.5-inc1"
                aa_params = dict(
                    translate_const=int(img_size_min * 0.45),
                    img_mean=tuple([min(255, round(255 * x)) for x in self.cfg.DATA.MEAN]),
                )
                seed = random.randint(0, 100000000)
                frames = [autoaugment.rand_augment_transform(
                    auto_augment_desc, aa_params, seed)(frame) for frame in frames]
                if self.cfg.Drive_and_act.EVENT_PATH is not None:
                    if self.cfg.MVIT.INPUT_MODAL== 'RGBIR':
                        prompts = [prompt.squeeze().numpy() for prompt in prompts]
                        prompts = [autoaugment.rand_augment_transform(
                            auto_augment_desc, aa_params, seed)(prompt) for prompt in prompts]
                    else:
                        prompts = [autoaugment.rand_augment_transform(
                            auto_augment_desc, aa_params, seed)(prompt) for prompt in prompts]
                # To Tensor: T H W C
                frames = [torch.tensor(np.array(frame)) for frame in frames]
                frames = torch.stack(frames)
            if self.cfg.Drive_and_act.EVENT_PATH is not None:
                prompts = [torch.tensor(np.array(prompt)) for prompt in prompts]
                prompts = torch.stack(prompts)
                prompts = utils.tensor_normalize(
                    prompts, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )
                prompts = prompts.permute(3, 0, 1, 2)
            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )

            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)

            # Perform data augmentation.
            use_random_resize_crop = self.cfg.DATA.USE_RANDOM_RESIZE_CROPS
            if use_random_resize_crop:
                if self.mode in ["train", "val"]:
                    frames = transform.random_resize_crop_video(frames, crop_size, interpolation_mode="bilinear")
                    frames, _ = transform.horizontal_flip(0.5, frames)
                else:
                    assert len({min_scale, max_scale, crop_size}) == 1
                    frames, _ = transform.random_short_side_scale_jitter(
                        frames, min_scale, max_scale
                    )
                    frames, _ = transform.uniform_crop(frames, crop_size, spatial_sample_index)
            else:
                # Perform data augmentation.
                frames = utils.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )
                if self.cfg.Drive_and_act.EVENT_PATH is not None:
                    prompts = utils.spatial_sampling(
                        prompts,
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale,
                        max_scale=max_scale,
                        crop_size=crop_size,
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    )     
            # T H W C -> T C H W.
            if self.mode in ["train", "val"]:
                frames = frames.permute(1, 0, 2, 3) # C T H W -> T C H W
                frames = utils.frames_augmentation(
                    frames,
                    colorjitter=self.cfg.DATA.COLORJITTER,
                    use_grayscale=self.cfg.DATA.GRAYSCALE,
                    use_gaussian=self.cfg.DATA.GAUSSIAN
                )
                if self.cfg.Drive_and_act.EVENT_PATH is not None:
                    prompts = prompts.permute(1, 0, 2, 3) # C T H W -> T C H W
                    prompts = utils.frames_augmentation(
                        prompts,
                        colorjitter=self.cfg.DATA.COLORJITTER,
                        use_grayscale=self.cfg.DATA.GRAYSCALE,
                        use_gaussian=self.cfg.DATA.GAUSSIAN
                    )
            label = self._video_records[index]['label']
            frames = utils.pack_pathway_output(self.cfg, frames)
            if self.cfg.Drive_and_act.EVENT_PATH is not None:
                frames.append(prompts)
            #prompts = utils.pack_pathway_output(self.cfg, prompts)

            metadata = {}
            return frames, label, index, metadata


    def __len__(self):
        return len(self._video_records)

    def spatial_sampling(
            self,
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
    
    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._video_records)