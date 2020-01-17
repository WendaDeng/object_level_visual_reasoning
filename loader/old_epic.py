from __future__ import print_function
import torch
import torch.utils.data as data
from torchvision import transforms
import os
import random
from PIL import Image
import ipdb
import pickle
from pycocotools import mask as maskUtils
import time
from random import shuffle

import numpy as np
from gulpio import GulpDirectory
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset, EpicVideoFlowDataset, GulpVideoSegment
from pathlib import Path


class EPIC(EpicVideoDataset):
    """
    Loader for the EPIC dataset
    """

    def __init__(self, gulp_path, class_type, options, *, nb_classes=125, with_metadata=None, sample_transform=None, w=224, h=224, nb_obj_t_max=10, mask_confidence=0.5, mask_size=28, nb_crops=1, real_w=256, real_h=256, add_background=True, usual_transform=False):
        super().__init__(gulp_path, class_type)
        self.class_type = class_type
        self.frames_per_segment = options['t']
        self.mask_dir = options['mask_dir']
        self.nb_classes = nb_classes
        self.w, self.h = w, h
        self.real_w, self.real_h = real_w, real_h
        self.w_mask, self.h_mask = mask_size, mask_size
        self.ratio_real_crop_w, self.ratio_real_crop_h = self.real_w / self.w, self.real_h / self.h
        self.real_mask_w, self.real_mask_h = int(self.ratio_real_crop_w * self.w_mask), int(self.ratio_real_crop_h * self.h_mask)
        self.nb_obj_max_t = nb_obj_t_max
        self.mask_confidence = mask_confidence
        self.nb_crops = nb_crops
        self.add_background = add_background
        self.usual_transform = usual_transform


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
        return len(self.video_segments)

    """
    Get frame indexes which need to be used from segment.
    """
    def time_sampling(self, segment_len):
        len_subseq = segment_len / float(self.frames_per_segment)
        timesteps = [int(random.sample(range(int(len_subseq)), 1)[0] + t * len_subseq)
                for t in range(self.frames_per_segment)]
        return timesteps


    def retrieve_segment_and_masks(self, id):
        segment = self.video_segments[id]
        timesteps = self.time_sampling(segment.num_frames)

        # Get PIL images and transform them to np arrays
        frames = self.load_frames(segment, timesteps)
        resize = transforms.Resize((self.real_h, self.real_w))
        np_frames = []
        for frame in frames:
            frame = resize(frame)
            np_frames.append(np.array(frame, dtype=np.uint8))
        np_frames = np.array(np_frames, dtype=np.float32)
        np_frames = np_frames.transpose([3, 0, 1, 2])
        # Get the masks
        (np_obj_id, np_bbox, np_masks, np_max_nb_obj) = self.retrieve_associated_masks(
                self.get_mask_file(segment),
                segment.num_frames,
                timesteps,
                add_background_mask=self.add_background)

        # Data processing on the super video
        np_frames, np_masks, np_bbox = self.video_transform(np_frames, np_masks, np_bbox)
        return np_frames, np_masks, np_bbox, np_obj_id, np_max_nb_obj


    def get_mask_file(self, segment):
        # Get the approriate masks
        fn = '{}_{}_{}.pkl'.format(segment.video_id, segment.start_timestamp, segment.stop_timestamp)
        mask_fn = os.path.join(self.mask_dir, fn)

        return mask_fn


    def load_masks(self, file):
        with open(file, 'rb') as f:
            masks = pickle.load(f, encoding='latin-1')
        return (masks['segms'], masks['boxes'])


    def retrieve_associated_masks(self, mask_file, video_len, timesteps, add_background_mask=True):
        T = len(timesteps)
        np_obj_id = np.zeros((T, self.nb_obj_max_t, 81)).astype(np.float32)
        np_bbox = np.zeros((T, self.nb_obj_max_t, 4)).astype(np.float32)
        np_masks = np.zeros((T, self.nb_obj_max_t, self.real_mask_h, self.real_mask_w)).astype(np.float32)
        np_max_nb_obj = np.asarray([self.nb_obj_max_t]).reshape((1,))

        # raise Exception
        # try:
        segms, boxes = self.load_masks(mask_file)

        # Timestep factor
        factor = video_len / len(segms)
        timesteps = [int(t / factor) for t in timesteps]
        # Retrieve information
        list_nb_obj = []
        for t_for_clip, t in enumerate(timesteps):
            nb_obj_t = 0
            # Range of objects
            range_objects = list(range(2, 81))
            shuffle(range_objects)
            range_objects = [1] + range_objects
            for c in range_objects:
                for i in range(len(boxes[t][c])):
                    if boxes[t][c][i] is not None and len(boxes[t][c]) > 0 and boxes[t][c][i][-1] > self.mask_confidence:
                        # Obj id
                        np_obj_id[t_for_clip, nb_obj_t, c] = 1
                        # Bounding box
                        H, W = segms[t][c][i]['size']
                        x1, y1, x2, y2, _ = boxes[t][c][i]
                        x1, x2 = (x1 / W) * self.real_w, (x2 / W) * self.real_w
                        y1, y2 = (y1 / H) * self.real_h, (y2 / H) * self.real_h
                        np_bbox[t_for_clip, nb_obj_t] = [x1, y1, x2, y2]
                        # Masks
                        rle_obj = segms[t][c][i]
                        m = maskUtils.decode(rle_obj)  # Python COCO API
                        # Resize
                        m_pil = Image.fromarray(m)
                        m_pil = m_pil.resize((self.real_mask_w, self.real_mask_h))
                        m = np.array(m_pil, copy=False)
                        np_masks[t_for_clip, nb_obj_t] = m
                        nb_obj_t += 1
                        # Break if too much objects
                        if nb_obj_t > (self.nb_obj_max_t - 1):
                            break
                # Break if too much objects
                if nb_obj_t > (self.nb_obj_max_t - 1):
                    break
            # Append
            list_nb_obj.append(nb_obj_t)
        # And now fill numpy array
        np_max_nb_obj[0] = max(list_nb_obj)
        # except Exception as e:
        #     print("mask reading problem: ", )
        #     ipdb.set_trace()
        #     np_max_nb_obj[0] = 1.

        # Add the background mask
        if add_background_mask:
            # Find the background pixels
            sum_masks = np.clip(np.sum(np_masks, 1), 0, 1)
            background_mask = 1 - sum_masks

            # Add meta data about background
            idx_bg_mask = int(np_max_nb_obj[0])
            idx_bg_mask -= 1 if self.nb_obj_max_t == idx_bg_mask else 0
            np_masks[:, idx_bg_mask] = background_mask
            np_obj_id[:, idx_bg_mask, 0] = 1
            np_bbox[:, idx_bg_mask] = [0, 0, 1, 1]
            # Update the number of mask
            np_max_nb_obj[0] = np_max_nb_obj[0] + 1 if np_max_nb_obj < self.nb_obj_max_t else np_max_nb_obj[0]
        return (np_obj_id, np_bbox, np_masks, np_max_nb_obj)


    def get_target(self, id):
        if self.class_type == 'verb':
            label = self.video_segments[id].verb_class
        elif self.class_type == 'noun':
            label = self.video_segments[id].noun_class
        elif self.class_type == 'verb+noun':
            label = self.video_segments[id].verb_class, self.video_segments[id].noun_class
        return torch.FloatTensor(label)


    def video_transform(self, np_clip, np_masks, np_bbox):
        # Random crop
        _, _, h, w = np_clip.shape
        w_min, h_min = random.sample(range(w - self.w), 1)[0], random.sample(range(h - self.h), 1)[0]
        # clip
        np_clip = np_clip[:, :, h_min:(self.h + h_min), w_min:(self.w + w_min)]
        # mask
        h_min_mask, w_min_mask = round((h_min / self.h) * self.h_mask), round((w_min / self.w) * self.w_mask)
        np_masks = np_masks[:, :, h_min_mask:(self.h_mask + h_min_mask), w_min_mask:(self.w_mask + w_min_mask)]
        # bbox
        np_bbox[:, :, [0, 2]] = np.clip(np_bbox[:, :, [0, 2]] - w_min, 0, self.w)
        np_bbox[:, :, [1, 3]] = np.clip(np_bbox[:, :, [1, 3]] - h_min, 0, self.h)
        # rescale to 0->1
        np_bbox[:, :, [0, 2]] /= self.w
        np_bbox[:, :, [1, 3]] /= self.h

        if self.usual_transform:
            # Div by 255
            np_clip /= 255.

            # Normalization
            np_clip -= np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)  # mean
            np_clip /= np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)  # std

        return np_clip, np_masks, np_bbox

