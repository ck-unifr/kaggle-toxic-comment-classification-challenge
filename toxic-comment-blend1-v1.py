import pandas as pd


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

b1 = gru_v1.copy()
col = gru_v1.columns

col = col.tolist()
col.remove('id')

for i in col:
    b1[i] = (6 * gru_v1[i] + 5 * gru_v2[i] + 2 * lr_v1[i] + 2 * mlp_v1[i]) / 15

b1.to_csv('submission-blend1-v1', index=False)