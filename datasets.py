import os, glob
import pandas as pd
import itertools
import numpy as np
import random
import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms



NUM_TRIPLETS = 2500


class JIGSAWS(Dataset):
    def __init__(self, train_test_val, experiment, return_triplets=False):
        self.train_test_val = train_test_val
        self.return_triplets = return_triplets
        self.dataset_loc = "datasets/experiment{:02}_{}.csv".format(experiment, train_test_val)
        self.dataset = self.create_dataset()
        self.classes = ["G1","G2","G3","G4","G5","G6","G8","G9","G10","G11","G12","G13","G14","G15"]
        self.classes = [x + "_expert" for x in self.classes] + [x + "_novice" for x in self.classes]




    def create_dataset(self):
        # Read dataset
        df = pd.read_csv(self.dataset_loc)

        # Turn score into binary "expert"/"novice"
        ### TODO: play with these numbers - balance tradeoff between number of gestures and extremes of skill
        expert_min = 18
        novice_max = 18
        def binarize_score(score):
            if score > expert_min:
                return "expert"
            elif score < novice_max:
                return "novice"
            return "remove_me"

        # Duplicate items that are right/left hand
        def duplicate_gestures(df):
            dups = {"G1": "G12", "G12": "G1"}
            df_dups = df[df["gesture"].isin(dups.keys())].copy()
            df_dups["gesture"] = df_dups["gesture"].apply(lambda x: dups[x])
            df_dups["duplicate"] = True
            df["duplicate"] = False
            return pd.concat([df, df_dups], ignore_index=True)

        # Convert to expert/novice
        df["skill"] = df["score"].apply(binarize_score)
        df = df[df["skill"] != "remove_me"]
        df["gesture_skill"] = df.apply(lambda x: x["gesture"] + "_" + x["skill"], axis=1)

        # Duplicate
        if self.train_test_val == "train":
            df = duplicate_gestures(df)

        # Sort values so always using same ids
        df = df.sort_values(["frame_start", "video"]).reset_index(drop=True)

        return df





    def __len__(self):
        return NUM_TRIPLETS if self.return_triplets else len(self.dataset)





    def get_cached_video(self, video_idx, gesture):
        cap = "1" if np.random.rand() < 0.5 else "2"
        all_ims = self.video_frames["{:04}_{}".format(video_idx, cap)]
        if gesture in ["G2", "G3", "G5", "G8", "G10", "G11", "G15"] and np.random.rand() < 0.5:
            all_ims = [transforms.functional.hflip(im) for im in all_ims]
        return all_ims





    def get_op_flow(self, frame0_path, frame1_path, vid, frame, cap):
        cache_path = './cached_op_flow/{}_{:05}{}.png'.format(vid, frame, cap)

        # Cached image exists
        if os.path.isfile(cache_path):
            try:
                return Image.open(cache_path).copy()
            except OSError:
                pass


        # Create opflow+rgb image
        bound = 15
        im1 = Image.open(frame0_path).convert('L').copy()
        im2 = Image.open(frame1_path).convert('L').copy() if os.path.isfile(frame1_path) else im1.copy()
        op_flow = cv2.calcOpticalFlowFarneback(np.asarray(im1), np.asarray(im2), None, 0.4, 1, 15, 3, 8, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        op_flow = np.clip(op_flow, -bound, bound)
        op_flow = (((op_flow + bound) * 255) / (bound * 2)).astype(np.uint8)
        op_flow = np.concatenate([op_flow, np.expand_dims(np.asarray(im1),2)], axis=2)
        i = transforms.functional.to_pil_image(op_flow)

        # Cache
        i.save(cache_path, "PNG")

        return i





    def sample_video(self, video, gesture, frame_start, frame_stop):
        # Randomly choose endoscope view
        cap = "_capture1" if np.random.rand() < 0.5 else "_capture2"

        # Sample image paths
        # im_paths = sorted(glob.glob("/home/mike/Projects/Query_By_Example/JIGSAWS/video_frames/{}{}/{:05}.jpg".format(video, cap)))


        # Get sample indices
        sample_idxs = np.linspace(frame_start, frame_stop, num=50, endpoint=False, dtype=int)
        # max_ims = 800
        # if frame_stop - frame_start < max_ims:
        #     sample_idxs = [x for x in range(frame_start, frame_stop)]
        # else:
        #     sample_idxs = np.linspace(frame_start, frame_stop, max_ims, endpoint=False, dtype=int)

        # Read in all images
        all_ims = [
            self.get_op_flow(
                "/home/mike/Projects/Query_By_Example/JIGSAWS/video_frames/{}{}/{:05}.png".format(video, cap, x+1),
                "/home/mike/Projects/Query_By_Example/JIGSAWS/video_frames/{}{}/{:05}.png".format(video, cap, x+2),
                video, x+1, cap) for x in sample_idxs]

        # Randomly flip horizontally if allowed
        if gesture in ["G2", "G3", "G5", "G8", "G10", "G11", "G15"] and np.random.rand() < 0.5:
            all_ims = [transforms.functional.hflip(im) for im in all_ims]

        return all_ims




    def transforms(self, ims):
        transformed_ims = []
        color_jitter = transforms.ColorJitter.get_params(.25,.25,.25,.25)
        rotation_param = transforms.RandomRotation.get_params((-20,20))
        random_crop_params = transforms.RandomResizedCrop(224).get_params(ims[0], (0.8, 1.0), (3. / 4., 4. / 3.))
        for i in ims:
            if self.train_test_val == 'train':
                i = transforms.functional.resized_crop(i, *random_crop_params, (224, 224))
                i = color_jitter(i)
                i = transforms.functional.rotate(i, rotation_param)
            else:
                i = transforms.functional.resize(i, (224, 224))
            i = transforms.functional.to_tensor(i)
            i = transforms.functional.normalize(i, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transformed_ims.append(i)
        return torch.stack(transformed_ims)





    def __getitem__(self, idx):
        if self.return_triplets:
            idx = np.random.randint(len(self.dataset))
        anchor = self.dataset.iloc[idx]
        ims = self.sample_video(anchor['video'], anchor['gesture'], anchor["frame_start"], anchor["frame_stop"])
        transformed_ims = self.transforms(ims)

        if self.return_triplets:
            pos = self.dataset[
                (self.dataset["gesture_skill"] == anchor["gesture_skill"]) & (self.dataset.index != idx)
                ].sample(1).iloc[0]
            neg = self.dataset[
                (self.dataset["gesture_skill"] != anchor["gesture_skill"]) & (self.dataset.index != idx)
                ].sample(1).iloc[0]
            pos_ims = self.transforms(self.sample_video(pos['video'], pos['gesture'], pos["frame_start"], pos["frame_stop"]))
            neg_ims = self.transforms(self.sample_video(neg['video'], neg['gesture'], neg["frame_start"], neg["frame_stop"]))
        else:
            pos_ims = []
            neg_ims = []

        return {
            'images':        transformed_ims,
            'pos_images':    pos_ims,
            'neg_images':    neg_ims,
            'skill':         anchor["score"],
            'gesture_skill': anchor["gesture_skill"],
            'gesture_id':    self.classes.index(anchor["gesture_skill"]),
            'video':         anchor["video"],
            'frame_start':   anchor["frame_start"]
        }



if __name__ == '__main__':
    vals = {
        'train': 0,
        'test': 0,
        'val': 0
    }
    for experiment in range(8):
        for set in ["train", "test", "val"]:
            try:
                ds = JIGSAWS(set, experiment)
            except ValueError:
                ds = []
            print(experiment, set, len(ds))
            vals[set] += len(ds)
    exit()
    ds = JIGSAWS('test')
    print(len(ds))
    for idx in range(len(ds)):
        anchor = ds.dataset.iloc[idx]
        pos = ds.dataset[
            (ds.dataset["gesture_skill"] == anchor["gesture_skill"]) & (ds.dataset.index != idx)
            ].sample(1)
        print(pos)
