import os, glob
import random
import pandas as pd
import subprocess
from tqdm import tqdm



JIGSAWS = '/home/mike/Projects/Query_By_Example/JIGSAWS'
# JIGSAWS = '/Users/mpeven/Downloads/JIGSAWS'


####################################################################################################
####################################################################################################
### Create full csv
tasks = ['Knot_Tying', 'Suturing', 'Needle_Passing']

all_dfs = []
for task in tasks:
    df = pd.read_csv('{0}/{1}/meta_file_{1}.txt'.format(JIGSAWS, task), sep="\t", header=None,
                     usecols=[0,3], names=["video", "score"])

    for idx, row in df.iterrows():
        t_df = pd.read_csv('{}/{}/transcriptions/{}.txt'.format(JIGSAWS, task, row["video"]), sep=" ",
                           index_col=False, names=["frame_start", "frame_stop", "gesture"])

        # Add score and video labels
        t_df["video"] = row["video"]
        t_df["score"] = row["score"]

        # 1 indexed --> 0 indexed
        t_df["frame_start"] = t_df["frame_start"].apply(lambda x: x-1)
        t_df["frame_stop"] = t_df["frame_stop"].apply(lambda x: x-1)
        all_dfs.append(t_df)

df = pd.concat(all_dfs, ignore_index=True)
df.to_csv("dataset.csv", index=False)

####################################################################################################
####################################################################################################






####################################################################################################
####################################################################################################
### Create leave-one-user-out experiment datasets

df["user"] = df["video"].apply(lambda x: x[-4])
for idx, user in enumerate(df["user"].unique()):
    other_users = [x for x in df["user"].unique() if x != user]
    val_user = random.choice(other_users)

    df_test = df[df["user"] == user]
    df_val = df[df["user"] == val_user]
    df_train = df[df["user"].isin([x for x in other_users if x != val_user])]

    df_train.to_csv("datasets/experiment{:02}_train.csv".format(idx), index=False)
    df_val.to_csv("datasets/experiment{:02}_val.csv".format(idx), index=False)
    df_test.to_csv("datasets/experiment{:02}_test.csv".format(idx), index=False)

####################################################################################################
####################################################################################################






####################################################################################################
####################################################################################################
### Extract video frames
# os.mkdir("{}/video_frames".format(JIGSAWS))
# for vid_path in tqdm(sorted(glob.glob("{}/*/video/*".format(JIGSAWS))), desc="Extracting rgb frames", ncols=100):
#     vid_id = os.path.basename(vid_path).replace(".avi", "")
#     os.mkdir("{}/video_frames/{}".format(JIGSAWS, vid_id))
#     cmd = "ffmpeg -hide_banner -loglevel panic -i {} {}/video_frames/{}/%05d.png".format(vid_path, JIGSAWS, vid_id)
#     subprocess.call(cmd, shell=True)
####################################################################################################
####################################################################################################
