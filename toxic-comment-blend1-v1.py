import pandas as pd


gru_v1 = pd.read_csv("submission-pooled-gru-v1.csv")
gru_v2 = pd.read_csv("submission-pooled-gru-v2.csv")
gru_v3 = pd.read_csv("submission-pooled-gru-v3.csv")
lr_v1 = pd.read_csv("submission-lr-v1.csv")
mlp_v1 = pd.read_csv("submission-mlp-v1.csv")


# gru = pd.read_csv("../input/who09829/submission.csv")
# gruglo = pd.read_csv("../input/pooled-gru-glove-with-preprocessing/submission.csv")
# ave = pd.read_csv("../input/toxic-avenger/submission.csv")
# s9821 = pd.read_csv("../input/toxicfile/sub9821.csv")
# # glove = pd.read_csv('../input/toxic-glove/glove.csv')
# # svm = pd.read_csv("../input/toxic-nbsvm/nbsvm.csv")
# best = pd.read_csv('../input/toxic-hight-of-blending/hight_of_blending.csv')

# b1 = gru_v1.copy()
# col = gru_v1.columns
# col = col.tolist()
# col.remove('id')
# for i in col:
#     b1[i] = (4 * gru_v1[i] + 3 * gru_v2[i] + 2 * gru_v3[i] + 1 * lr_v1[i] + 2 * mlp_v1[i]) / 12
# b1.to_csv('submission-blend1-v1', index=False)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = gru_v1.copy()
#p_res[label_cols] = (2*p_nbsvm[label_cols] + 3*p_lstm[label_cols] + 4*p_eaf[label_cols]) / 9
p_res[label_cols] = (4*gru_v1[label_cols] + 3*gru_v2[label_cols] + 2*gru_v3[label_cols]
                     + 1*lr_v1[label_cols] + 2*mlp_v1[label_cols]) / 12
p_res.to_csv('submission-blend1-v1.csv', index=False)
