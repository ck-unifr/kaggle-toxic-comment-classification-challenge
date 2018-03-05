# The script is modified from
# https://www.kaggle.com/the1owl/toxic-simple-blending-toxic-avenger-spin

import pandas as pd
import numpy as np

gru_v1 = pd.read_csv("submission-pooled-gru-v1.csv")
gru_v2 = pd.read_csv("submission-pooled-gru-v2.csv")
gru_v3 = pd.read_csv("submission-pooled-gru-v3.csv")
gru_v4 = pd.read_csv("submission-pooled-gru-v4.csv")

cnn_lstm_v1 = pd.read_csv("submission_cnn_lstm_v1.csv")
cnn_lstm_v2 = pd.read_csv("submission_cnn_lstm_v2.csv")
cnn_gru_v1 = pd.read_csv("submission_cnn_gru_v1.csv")

lr_v1 = pd.read_csv("submission-lr-v1.csv")
mlp_v1 = pd.read_csv("submission-mlp-v1.csv")

# s9821 = pd.read_csv("sub9821.csv")
# hight = pd.read_csv('hight_of_blending.csv')


# gru = pd.read_csv("../input/who09829/submission.csv")
# gruglo = pd.read_csv("../input/pooled-gru-glove-with-preprocessing/submission.csv")
# ave = pd.read_csv("../input/toxic-avenger/submission.csv")
# s9821 = pd.read_csv("../input/toxicfile/sub9821.csv")
# # glove = pd.read_csv('../input/toxic-glove/glove.csv')
# # svm = pd.read_csv("../input/toxic-nbsvm/nbsvm.csv")
# best = pd.read_csv('../input/toxic-hight-of-blending/hight_of_blending.csv')

# ble = gru_v1.copy()
# col = gru_v1.columns
# col = col.tolist()
# col.remove('id')
# for i in col:
#     ble[i] = (4 * gru_v1[i] + 3 * gru_v2[i] + 2 * gru_v3[i] + 1 * lr_v1[i] + 2 * mlp_v1[i]) / 12

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
ble = gru_v1.copy()
#p_res[label_cols] = (2*p_nbsvm[label_cols] + 3*p_lstm[label_cols] + 4*p_eaf[label_cols]) / 9
ble[label_cols] = (4*gru_v1[label_cols] + 3*gru_v2[label_cols] + 3*gru_v3[label_cols] + 2*gru_v4[label_cols]
                   + 4*cnn_lstm_v1[label_cols] + 5*cnn_lstm_v1[label_cols] + 4*cnn_gru_v1[label_cols]
                   + 1*lr_v1[label_cols] + 2*mlp_v1[label_cols]) / 28
# ble.to_csv('submission-blend1-v1', index=False)


# submission_ave.csv is downloaded from
# https://www.kaggle.com/the1owl/toxic-simple-blending-toxic-avenger-spin/notebook

# ave = pd.read_csv('superblend_1.csv')
# ave = pd.read_csv('submission_ave.csv')
ave = pd.read_csv('submission-lgb-gru-lr-lstm-nb-svm-ave-ensemble.csv')

sub1 = ble[:]
sub2 = ave[:]
coly = [c for c in ave.columns if c not in ['id', 'comment_text']]
sub2.columns = [x+'_' if x not in ['id'] else x for x in sub2.columns]
blend = pd.merge(sub1, sub2, how='left', on='id')
for c in coly:
    blend[c] = np.sqrt(blend[c] * blend[c+'_'])
    blend[c] = blend[c].clip(0+1e12, 1-1e12)
blend = blend[sub1.columns]

submission = pd.read_csv('sample_submission.csv')
submission[label_cols] = blend[label_cols]
submission.to_csv('submission-blend1-v2.csv', index=False)