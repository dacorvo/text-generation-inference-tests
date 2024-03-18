import pandas as pd
import glob

# read all json files from sagemaker directory into pandas df
df = pd.concat([pd.read_json(f,orient = 'index').transpose() for f in glob.glob('sagemaker/*.json')], ignore_index = True)
df.to_csv('sagemaker.csv', index=False)
