from __future__ import print_function
import torch
import torch.utils.data as data
from torchvision import transforms
import os
import random
from PIL import Image
import numpy as np
import ipdb
import pickle
import pandas as pd
from pycocotools import mask as maskUtils
import lintel
import time
from torch.utils.data.dataloader import default_collate
from random import shuffle
from loader.videodataset import VideoDataset
import jpeg4py as jpeg


class EPIC(VideoDataset):
    """
    Loader for the EPIC dataset
    """

    def __init__(self, options, **kwargs):
        super().__init__(options, **kwargs)

        self.class_type = options['class-type']
        # Metadata pickle
        self.metadata_pickle = os.path.join(self.root, 'meta/train_labels.pkl')
        self.video_metadata = self.get_metadata(self.metadata_pickle)

        # Videos paths
        self.list_video  = self.get_videos()


    def get_metadata(self, metadata_pickle):
        labels = pd.read_pickle(metadata_pickle)
        data = []
        for i, row in labels.iterrows():
            metadata = row.to_dict()
            metadata["uid"] = i
            data.append(metadata)
        return data

    def get_videos(self):
        list_video = []
        for root, dirs, files in os.walk(self.video_dir_full):
            for d in dirs:
                list_video.append(os.path.join(root, d))

        return list_video

    def starting_point(self, id):
        return 0

    def get_video_fn(self, id):
        return self.list_video[id]

    def get_length(self, id):
        data = self.video_metadata[id]
        length = data['stop_frame'] - data['start_frame']
        return length

    def get_target(self, id):
        if self.class_type == 'verb':
            label = self.video_metadata[id].verb_class
        elif self.class_type == 'noun':
            label = self.video_metadata[id].noun_class
        elif self.class_type == 'verb+noun':
            label = self.video_metadata[id].verb_class, self.video_segments[id].noun_class
        return torch.FloatTensor(label)

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             dict: info about the video
        """
        #try:
        # Target
        torch_target = self.get_target(index)

        np_clip, np_masks, np_bbox, np_obj_id, np_max_nb_obj = self.retrieve_segment_and_masks(index)

        # Torch world
        torch_clip = torch.from_numpy(np_clip)
        torch_masks = torch.from_numpy(np_masks)
        torch_obj_id = torch.from_numpy(np_obj_id)
        torch_obj_bboxes = torch.from_numpy(np_bbox)
        torch_max_nb_objs = torch.from_numpy(np_max_nb_obj)

        return {"target": torch_target,
                    "clip": torch_clip,
                    "mask": torch_masks,
                    "obj_id": torch_obj_id,
                    "obj_bbox": torch_obj_bboxes,
                    "max_nb_obj": torch_max_nb_objs
                    }
        #except Exception as e:
        #    return None


    def __len__(self):
        return len(self.video_metadata)

    """
    Get frame indexes which need to be used from segment.
    """
    def time_sampling(self, video_len):
        len_subseq = video_len / float(self.t)
        timesteps = [int(random.sample(range(int(len_subseq)), 1)[0] + t * len_subseq)
                for t in range(self.t)]
        return timesteps

    def load_frames(self, video_fn, timesteps):
        frames = []
        for i in timesteps:
            img_path = os.path.join(video_fn, 'frame_{0:{fill}{align}10}.jpg'.format(i, fill=0, align='>'))
            img = jpeg.JPEG(img_path).decode()
            frames.append(np.array(img, dtype=np.uint8))
        np_frames = np.array(frames, dtype=np.float32)
        return np_frames.transpose([3, 0, 1, 2])

    def retrieve_segment_and_masks(self, id):
        video_len = self.get_length(id)
        timesteps = self.time_sampling(video_len)
        np_frames = self.load_frames(self.get_video_fn(id), timesteps)

        # Get the masks
        (np_obj_id, np_bbox, np_masks, np_max_nb_obj) = self.retrieve_associated_masks(
                self.get_mask_file(id), video_len, timesteps,
                add_background_mask=self.add_background)

        # Data processing on the super video
        np_frames, np_masks, np_bbox = self.video_transform(np_frames, np_masks, np_bbox)
        return np_frames, np_masks, np_bbox, np_obj_id, np_max_nb_obj


    def get_mask_file(self, id):
        # Get the approriate masks
        data = self.video_metadata[id]
        fn = '{}_{}_{}.pkl'.format(data.video_id, data.start_timestamp, data.stop_timestamp)
        mask_fn = os.path.join(self.mask_dir, fn)
        return mask_fn
