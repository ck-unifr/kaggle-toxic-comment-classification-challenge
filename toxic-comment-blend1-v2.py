# The script is modified from
# https://www.kaggle.com/the1owl/toxic-simple-blending-toxic-avenger-spin

import pandas as pd
import numpy as np

gru_v1 = pd.read_csv("submission-pooled-gru-v1.csv")
gru_v2 = pd.read_csv("submission-pooled-gru-v2.csv")
lr_v1 = pd.read_csv("submission-lr-v1.csv")
mlp_v1 = pd.read_csv("submission-mlp-v1.csv")


# gru = pd.read_csv("../input/who09829/submission.csv")
# gruglo = pd.read_csv("../input/pooled-gru-glove-with-preprocessing/submission.csv")
# ave = pd.read_csv("../input/toxic-avenger/submission.csv")
# s9821 = pd.read_csv("../input/toxicfile/sub9821.csv")
# # glove = pd.read_csv('../input/toxic-glove/glove.csv')
# # svm = pd.read_csv("../input/toxic-nbsvm/nbsvm.csv")
# best = pd.read_csv('../input/toxic-hight-of-blending/hight_of_blending.csv')

ble = gru_v1.copy()
col = gru_v1.columns

col = col.tolist()
col.remove('id')

for i in col:
    ble[i] = (4 * gru_v1[i] + 3 * gru_v2[i] + 2 * lr_v1[i] + 2 * mlp_v1[i]) / 11

# submission_ave.csv is downloaded from
# https://www.kaggle.com/the1owl/toxic-simple-blending-toxic-avenger-spin/notebook
ave = pd.read_csv('submission_ave.csv')
sub1 = ble[:]
sub2 = ave[:]
coly = [c for c in ave.columns if c not in ['id','comment_text']]
sub2.columns = [x+'_' if x not in ['id'] else x for x in sub2.columns]
blend = pd.merge(sub1, sub2, how='left', on='id')
for c in coly:
    blend[c] = np.sqrt(blend[c] * blend[c+'_'])
    blend[c] = blend[c].clip(0+1e12, 1-1e12)
blend = blend[sub1.columns]
blend.to_csv('submission.csv', index=False)