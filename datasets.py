import os, glob
import pandas as pd
import itertools
import numpy as np
import random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms



NUM_TRIPLETS = 25000


class JIGSAWS(Dataset):
    def __init__(self, train_test_val, return_triplets=False, experiment=0):
        self.train_test_val = train_test_val
        self.return_triplets = return_triplets
        self.dataset_loc = "datasets/experiment{:02}_{}.csv".format(experiment, train_test_val)
        self.dataset, self.triplets = self.create_dataset()
        self.classes = ["G1","G2","G3","G4","G5","G6","G7","G8","G9","G10","G11","G12","G13","G14","G15"]



    def create_dataset(self):
        # Read dataset
        df = pd.read_csv(self.dataset_loc)

        # Turn score into binary "expert"/"novice"
        def binarize_score(row):
            # TODO!!!!!!!!!!!!!!!!!!!!
            return row

        # Duplicate items that are right/left hand
        def duplicate_gestures(df):
            dups = {
                "G1": "G12", "G12": "G1",
                "G6": "G7", "G7": "G6",

            }
            df_dups = df[df["gesture"].isin(dups.keys())].copy()
            df_dups["gesture"] = df_dups["gesture"].apply(lambda x: dups[x])
            df_dups["duplicate"] = True
            df["duplicate"] = False
            return pd.concat([df, df_dups], ignore_index=True)

        df = df.apply(binarize_score, axis=1)
        df = duplicate_gestures(df)

        # Sort values so always using same ids
        df = df.sort_values(["frame_start", "video"]).reset_index(drop=True)

        if not self.return_triplets:
            return df, None

        # Create triplets
        triplets = []
        from tqdm import tqdm
        for gesture, gesture_df in tqdm(df.groupby("gesture"), ncols=100, desc='Indexing dataset'):
            pos_indices = list(gesture_df.index)
            neg_indices = list(df[df["gesture"] != gesture].index)
            for anchor in pos_indices:
                other_idxs = [x for x in pos_indices if x != anchor]
                triplets += itertools.product((anchor,), other_idxs, neg_indices)

        return df, triplets




    def __len__(self):
        return NUM_TRIPLETS if self.return_triplets else len(self.dataset)




    def sample_video(self, video, gesture, frame_start, frame_stop):
        # Randomly choose endoscope view
        cap = "_capture1" if np.random.rand() < 0.5 else "_capture2"

        # Get all images
        im_paths = sorted(glob.glob("/home/mike/Projects/Query_By_Example/JIGSAWS/video_frames/{}{}/*".format(video, cap)))

        # Read in all images
        sample_idxs = np.linspace(frame_start, frame_stop, 10, endpoint=True, dtype=int)
        all_ims = [Image.open(im_paths[x]) for x in sample_idxs]

        # Randomly flip horizontally if allowed
        if gesture in ["G2", "G3", "G5", "G8", "G10", "G11", "G15"] and np.random.rand() < 0.5:
            all_ims = [transforms.functional.hflip(im) for im in all_ims]

        return all_ims




    def transforms(self, ims):
        transformed_ims = []
        for im in ims:
            i = transforms.functional.resize(im, (224,224))
            i = transforms.functional.to_tensor(i)
            i = transforms.functional.normalize(i, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transformed_ims.append(i)
        return torch.stack(transformed_ims)




    def __getitem__(self, idx):
        if self.return_triplets:
            # Get row from csv
            triplet = random.choice(self.triplets)
            anchor   = self.dataset.iloc[triplet[0]]
            positive = self.dataset.iloc[triplet[1]]
            negative = self.dataset.iloc[triplet[2]]

            # Get images
            all_ims = []
            for x in [anchor, positive, negative]:
                ims = self.sample_video(x['video'], x['gesture'], x["frame_start"], x["frame_stop"])
                all_ims.append(self.transforms(ims))

            # Get label --> start with gesture only
            labels = [self.classes.index(x["gesture"]) for x in [anchor, positive, negative]]

            return all_ims[0], all_ims[1], all_ims[2], labels

        else:
            anchor = self.dataset.iloc[idx]
            ims = self.sample_video(anchor['video'], anchor['gesture'], anchor["frame_start"], anchor["frame_stop"])
            transformed_ims = self.transforms(ims)
            label = self.classes.index(anchor["gesture"])
            return transformed_ims, label



if __name__ == '__main__':
    ds = JIGSAWS('test')
    print(len(ds.classes))
    print(ds.dataset.head(10))
    ds.__getitem__(4)
