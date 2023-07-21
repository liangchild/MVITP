#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import torch
from . import utils as utils
import numpy as np
from .decoder import get_start_end_idx


def temporal_sampling(
    num_frames, start_idx, end_idx, num_samples, start_frame=0
):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        num_frames (int): number of frames of the trimmed action clip
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
        start_frame (int): starting frame of the action clip in the untrimmed video
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, num_frames - 1).long()
    return start_frame + index


def pack_frames_to_video_clip(
    cfg, video_record, train_mode='pretrain_rgb',target_fps=60,input_modal='rgb',prompt_record=None
):
    # Load video by loading its extracted frames
    path_to_video = video_record['frame_dir']
    img_tmpl = 'img_{:05}.jpg'
    num_samples = cfg.DATA.NUM_FRAMES  # sampling frames
    start_idx, end_idx = 1,video_record['total_frames']
    frame_idx = temporal_sampling(
        video_record['total_frames'],
        start_idx, end_idx, num_samples,
        start_frame=video_record['offset']
    )
    if prompt_record is not None:
        prompt_start_idx, prompt_end_idx = 1,prompt_record['total_frames']
        prompt_frame_idx = temporal_sampling(
            prompt_record['total_frames'],
            prompt_start_idx, prompt_end_idx, num_samples,
            start_frame=prompt_record['offset']
        )
    if train_mode == 'pretrain_event':
        if 'event_path' in video_record:  
            events=list()
            for idx in frame_idx:
                imgpath=os.path.join(path_to_video, img_tmpl.format(idx.item()))
                ids=imgpath.split('/')[-3]+ '/'+ imgpath.split('/')[-2]
                eventpath = os.path.join(video_record['event_path'],ids,video_record['event_tmpl'].format(idx.item()))
                eventframe = np.load(eventpath,allow_pickle=True, encoding="latin1")
                eventframe=np.transpose(eventframe,(1,2,0))
                events.append(eventframe)
            return events

    elif train_mode == 'prompt_fintuning': 
            img_paths=list()
            if prompt_record is not None:
                prompts=list()
                img_paths = [
                    os.path.join(
                        path_to_video, 
                        img_tmpl.format(idx.item()
                    )) for idx in frame_idx]
                frames = utils.retry_load_images(img_paths)
                prompts_paths = [
                    os.path.join(
                        prompt_record['frame_dir'], 
                        prompt_record['prompt_tmpl'].format(idx.item()
                    )) for idx in prompt_frame_idx]
                prompts = utils.retry_load_images(prompts_paths)     
                return frames,prompts
            else:
                events=list()
                for idx in frame_idx:
                    imgpath=os.path.join(path_to_video, img_tmpl.format(idx.item()))
                    ids=imgpath.split('/')[-3]+ '/'+ imgpath.split('/')[-2]
                    eventpath = os.path.join(video_record['event_path'],ids,video_record['event_tmpl'].format(idx.item()))
                    eventframe = np.load(eventpath,allow_pickle=True, encoding="latin1")
                    eventframe=np.transpose(eventframe,(1,2,0))
                    events.append(eventframe)
                    img_paths.append(imgpath)
                frames = utils.retry_load_images(img_paths) 
                return frames,events
    else:
        img_paths = [
            os.path.join(
                path_to_video, 
                img_tmpl.format(idx.item()
            )) for idx in frame_idx]
        frames = utils.retry_load_images(img_paths)
        return frames