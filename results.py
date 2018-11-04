import os
import pandas as pd
import numpy as np

### Get output csv files
experiment_dir = 'experiment2'
experiments = []
for e in range(8):
    csv = "{}/outputs_{:02}.csv".format(experiment_dir, e)
    if os.path.isfile(csv):
        experiments.append((e, csv))



### Get output
print("Overall accuracy (gesture, skill)")
results = []
for experiment, experiment_file in experiments:
    df = pd.read_csv("{}/outputs_{:02}.csv".format(experiment_dir, experiment))
    df['skill'] = df['skill'].apply(lambda x: int(x.replace("tensor(","").replace(")","")))
    df = df[(df['skill'] > 24) ^ (df['skill'] < 12)]
    if len(df) == 0:
        continue
    results.append({
        'experiment': experiment,
        'both': 100*sum(df['ground_truth'] == df['prediction'])/len(df),
        'gesture': 100*sum(df['ground_truth'].apply(lambda x: x.split("_")[0]) == df['prediction'].apply(lambda x: x.split("_")[0]))/len(df),
        'skill': 100*sum(df['ground_truth'].apply(lambda x: x.split("_")[1]) == df['prediction'].apply(lambda x: x.split("_")[1]))/len(df),
    })


results_df = pd.DataFrame.from_records(results, index='experiment')
pd.options.display.float_format = '{:.1f}'.format
print(results_df)
print("Average (both):    {:.02f}".format(results_df['both'].mean()))
print("Average (gesture): {:.02f}".format(results_df['gesture'].mean()))
print("Average (skill):   {:.02f}".format(results_df['skill'].mean()))
