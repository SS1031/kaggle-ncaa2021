import collections
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import torch
import xgboost as xgb
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split

STAGE_1 = True

expname = os.path.basename(__file__)

INPUTDIR = "../data/ncaam-march-mania-2021"
OUTPUTDIR = f"../output/{expname.replace('.py','')}"

if not os.path.exists(OUTPUTDIR):
    os.makedirs(OUTPUTDIR)

MRSCResults = pd.read_csv(INPUTDIR + "/MRegularSeasonCompactResults.csv")

# 勝率
A_w = (
    MRSCResults[MRSCResults.WLoc == "A"]
    .groupby(["Season", "WTeamID"])["WTeamID"]
    .count()
    .to_frame()
    .rename(columns={"WTeamID": "win_A"})
)
N_w = (
    MRSCResults[MRSCResults.WLoc == "N"]
    .groupby(["Season", "WTeamID"])["WTeamID"]
    .count()
    .to_frame()
    .rename(columns={"WTeamID": "win_N"})
)
H_w = (
    MRSCResults[MRSCResults.WLoc == "H"]
    .groupby(["Season", "WTeamID"])["WTeamID"]
    .count()
    .to_frame()
    .rename(columns={"WTeamID": "win_H"})
)
win = A_w.join(N_w, how="outer").join(H_w, how="outer").fillna(0)

H_l = (
    MRSCResults[MRSCResults.WLoc == "A"]
    .groupby(["Season", "LTeamID"])["LTeamID"]
    .count()
    .to_frame()
    .rename(columns={"LTeamID": "lost_H"})
)
N_l = (
    MRSCResults[MRSCResults.WLoc == "N"]
    .groupby(["Season", "LTeamID"])["LTeamID"]
    .count()
    .to_frame()
    .rename(columns={"LTeamID": "lost_N"})
)
A_l = (
    MRSCResults[MRSCResults.WLoc == "H"]
    .groupby(["Season", "LTeamID"])["LTeamID"]
    .count()
    .to_frame()
    .rename(columns={"LTeamID": "lost_A"})
)
lost = A_l.join(N_l, how="outer").join(H_l, how="outer").fillna(0)

win.index = win.index.rename(["Season", "TeamID"])
lost.index = lost.index.rename(["Season", "TeamID"])
wl = win.join(lost, how="outer").reset_index()
wl["win_pct_A"] = wl["win_A"] / (wl["win_A"] + wl["lost_A"])
wl["win_pct_N"] = wl["win_N"] / (wl["win_N"] + wl["lost_N"])
wl["win_pct_H"] = wl["win_H"] / (wl["win_H"] + wl["lost_H"])
wl["win_pct_All"] = (wl["win_A"] + wl["win_N"] + wl["win_H"]) / (
    wl["win_A"] + wl["win_N"] + wl["win_H"] + wl["lost_A"] + wl["lost_N"] + wl["lost_H"]
)

del A_w, N_w, H_w, H_l, N_l, A_l, win, lost

# スコア関連
MRSCResults["relScore"] = MRSCResults.WScore - MRSCResults.LScore

w_scr = MRSCResults.loc[:, ["Season", "WTeamID", "WScore", "WLoc", "relScore"]]
w_scr.columns = ["Season", "TeamID", "Score", "Loc", "relScore"]
l_scr = MRSCResults.loc[:, ["Season", "LTeamID", "LScore", "WLoc", "relScore"]]
l_scr["WLoc"] = l_scr.WLoc.apply(lambda x: "H" if x == "A" else "A" if x == "H" else "N")
l_scr["relScore"] = -1 * l_scr.relScore
l_scr.columns = ["Season", "TeamID", "Score", "Loc", "relScore"]
wl_scr = pd.concat([w_scr, l_scr])

A_scr = (
    wl_scr[wl_scr.Loc == "A"]
    .groupby(["Season", "TeamID"])[["Score", "relScore"]]
    .mean()
    .rename(columns={"Score": "Score_A", "relScore": "relScore_A"})
)
N_scr = (
    wl_scr[wl_scr.Loc == "N"]
    .groupby(["Season", "TeamID"])[["Score", "relScore"]]
    .mean()
    .rename(columns={"Score": "Score_N", "relScore": "relScore_N"})
)
H_scr = (
    wl_scr[wl_scr.Loc == "H"]
    .groupby(["Season", "TeamID"])[["Score", "relScore"]]
    .mean()
    .rename(columns={"Score": "Score_H", "relScore": "relScore_H"})
)
All_scr = (
    wl_scr.groupby(["Season", "TeamID"])[["Score", "relScore"]]
    .mean()
    .rename(columns={"Score": "Score_All", "relScore": "relScore_All"})
)
scr = (
    A_scr.join(N_scr, how="outer")
    .join(H_scr, how="outer")
    .join(All_scr, how="outer")
    .fillna(0)
    .reset_index()
)

del w_scr, l_scr, wl_scr, A_scr, H_scr, N_scr, All_scr

# Details
MRSDetailedResults = pd.read_csv(INPUTDIR + "/MRegularSeasonDetailedResults.csv")

w = MRSDetailedResults.loc[
    :,
    [
        "Season",
        "WTeamID",
        "WFGM",
        "WFGA",
        "WFGM3",
        "WFGA3",
        "WFTM",
        "WFTA",
        "WOR",
        "WDR",
        "WAst",
        "WTO",
        "WStl",
        "WBlk",
        "WPF",
    ],
]
w.columns = [
    "Season",
    "TeamID",
    "FGM",
    "FGA",
    "FGM3",
    "FGA3",
    "FTM",
    "FTA",
    "OR",
    "DR",
    "Ast",
    "TO",
    "Stl",
    "Blk",
    "PF",
]
l = MRSDetailedResults.loc[
    :,
    [
        "Season",
        "LTeamID",
        "LFGM",
        "LFGA",
        "LFGM3",
        "LFGA3",
        "LFTM",
        "LFTA",
        "LOR",
        "LDR",
        "LAst",
        "LTO",
        "LStl",
        "LBlk",
        "LPF",
    ],
]
l.columns = [
    "Season",
    "TeamID",
    "FGM",
    "FGA",
    "FGM3",
    "FGA3",
    "FTM",
    "FTA",
    "OR",
    "DR",
    "Ast",
    "TO",
    "Stl",
    "Blk",
    "PF",
]

detail = pd.concat([w, l])
detail["goal_rate"] = detail.FGM / detail.FGA
detail["3p_goal_rate"] = detail.FGM3 / detail.FGA3
detail["ft_goal_rate"] = detail.FTM / detail.FTA

dt = (
    detail.groupby(["Season", "TeamID"])[
        [
            "FGM",
            "FGA",
            "FGM3",
            "FGA3",
            "FTM",
            "FTA",
            "OR",
            "DR",
            "Ast",
            "TO",
            "Stl",
            "Blk",
            "PF",
            "goal_rate",
            "3p_goal_rate",
            "ft_goal_rate",
        ]
    ]
    .mean()
    .fillna(0)
    .reset_index()
)

del w, l, detail

# MassyOrdinals
MMOrdinals = pd.read_csv(INPUTDIR + "/MMasseyOrdinals.csv")

MOR_127_128 = MMOrdinals[
    (MMOrdinals.SystemName == "MOR")
    & ((MMOrdinals.RankingDayNum == 127) | (MMOrdinals.RankingDayNum == 128))
][["Season", "TeamID", "OrdinalRank"]]
MOR_50_51 = MMOrdinals[
    (MMOrdinals.SystemName == "MOR")
    & ((MMOrdinals.RankingDayNum == 50) | (MMOrdinals.RankingDayNum == 51))
][["Season", "TeamID", "OrdinalRank"]]
# MOR_15_16 = MMOrdinals[
#     (MMOrdinals.SystemName == "MOR")
#     & ((MMOrdinals.RankingDayNum == 15) | (MMOrdinals.RankingDayNum == 16))
# ][["Season", "TeamID", "OrdinalRank"]]

MOR_127_128 = MOR_127_128.rename(columns={"OrdinalRank": "OrdinalRank_127_128"})
MOR_50_51 = MOR_50_51.rename(columns={"OrdinalRank": "OrdinalRank_50_51"})
# MOR_15_16 = MOR_15_16.rename(columns={"OrdinalRank": "OrdinalRank_15_16"})

MOR = MOR_127_128.merge(MOR_50_51, how="left", on=["Season", "TeamID"])

## normalizing Rank values by its season maxium as it varies by seasons
MOR_max = MOR.groupby("Season")[["OrdinalRank_127_128", "OrdinalRank_50_51"]].max().reset_index()
MOR_max.columns = ["Season", "maxRank_127_128", "maxRank_50_51"]

MOR_tmp = MMOrdinals[(MMOrdinals.SystemName == "MOR") & (MMOrdinals.RankingDayNum < 133)]
MOR_stats = (
    MOR_tmp.groupby(["Season", "TeamID"])["OrdinalRank"]
    .agg(["max", "min", "std", "mean"])
    .reset_index()
)
MOR_stats.columns = ["Season", "TeamID", "RankMax", "RankMin", "RankStd", "RankMean"]

MOR = MOR.merge(MOR_max, how="left", on="Season").merge(
    MOR_stats, how="left", on=["Season", "TeamID"]
)
MOR["OrdinalRank_127_128"] = MOR["OrdinalRank_127_128"] / MOR["maxRank_127_128"]  # ランキングの最大値で正規化
MOR["OrdinalRank_50_51"] = MOR["OrdinalRank_50_51"] / MOR["maxRank_50_51"]  # ランキングの最大値で正規化
MOR["RankTrans_50_51_to_127_128"] = MOR["OrdinalRank_127_128"] - MOR["OrdinalRank_50_51"]

MOR.drop(
    ["maxRank_50_51", "maxRank_127_128"],
    axis=1,
    inplace=True,
)

del MOR_127_128, MOR_50_51, MOR_max, MOR_tmp, MOR_stats

wl_1 = wl.loc[:, ["Season", "TeamID", "win_pct_A", "win_pct_N", "win_pct_H", "win_pct_All"]]
wl_1.columns = [
    str(col) + "_1" if col not in ["Season", "TeamID"] else str(col) for col in wl_1.columns
]
wl_2 = wl.loc[:, ["Season", "TeamID", "win_pct_A", "win_pct_N", "win_pct_H", "win_pct_All"]]
wl_2.columns = [
    str(col) + "_2" if col not in ["Season", "TeamID"] else str(col) for col in wl_2.columns
]
scr_1 = scr.copy()
scr_1.columns = [
    str(col) + "_1" if col not in ["Season", "TeamID"] else str(col) for col in scr_1.columns
]
scr_2 = scr.copy()
scr_2.columns = [
    str(col) + "_2" if col not in ["Season", "TeamID"] else str(col) for col in scr_2.columns
]
dt_1 = dt.copy()
dt_1.columns = [
    str(col) + "_1" if col not in ["Season", "TeamID"] else str(col) for col in dt_1.columns
]
dt_2 = dt.copy()
dt_2.columns = [
    str(col) + "_2" if col not in ["Season", "TeamID"] else str(col) for col in dt_2.columns
]
MOR_1 = MOR.copy()
MOR_1.columns = [
    str(col) + "_1" if col not in ["Season", "TeamID"] else str(col) for col in MOR_1.columns
]
MOR_2 = MOR.copy()
MOR_2.columns = [
    str(col) + "_2" if col not in ["Season", "TeamID"] else str(col) for col in MOR_2.columns
]

TCResults = pd.read_csv(INPUTDIR + "/MNCAATourneyCompactResults.csv")

tourney1 = TCResults.loc[:, ["Season", "WTeamID", "LTeamID"]]
tourney1.columns = ["Season", "TeamID1", "TeamID2"]
tourney1["result"] = 1

tourney2 = TCResults.loc[:, ["Season", "LTeamID", "WTeamID"]]
tourney2.columns = ["Season", "TeamID1", "TeamID2"]
tourney2["result"] = 0

tourney = pd.concat([tourney1, tourney2])
del tourney1, tourney2

kenpom = pd.read_csv(f"{INPUTDIR}/MKenpom.csv")
kenpom_cols = [
    "rank",
    "adj_em",
    "adj_o",
    "adj_o_rank",
    "adj_d",
    "adj_d_rank",
    "adj_tempo",
    "adj_tempo_rank",
    "luck",
    "luck_rank",
    "sos_adj_em",
    "sos_adj_em_rank",
    "sos_adj_o",
    "sos_adj_o_rank",
    "sos_adj_d",
    "sos_adj_d_rank",
    "nc_sos_adj_em",
    "nc_sos_adj_em_rank",
]
kenpom_1 = kenpom[["Season", "TeamID"] + kenpom_cols].copy().rename(columns={"TeamID": "TeamID1"})
kenpom_1 = kenpom_1.rename(columns={c: "kenpom_" + c + "_1" for c in kenpom_cols})
kenpom_2 = kenpom[["Season", "TeamID"] + kenpom_cols].copy().rename(columns={"TeamID": "TeamID2"})
kenpom_2 = kenpom_2.rename(columns={c: "kenpom_" + c + "_2" for c in kenpom_cols})
diff_kenpom_cols = [
    "adj_o",
    "adj_o_rank",
    "adj_d",
    "adj_d_rank",
    "adj_tempo",
    "adj_tempo_rank",
    "sos_adj_em",
    "sos_adj_em_rank",
    "sos_adj_o",
    "sos_adj_o_rank",
    "sos_adj_d",
    "sos_adj_d_rank",
]


def merge_data(df):
    df = df.merge(wl_1, how="left", left_on=["Season", "TeamID1"], right_on=["Season", "TeamID"])
    df = df.merge(wl_2, how="left", left_on=["Season", "TeamID2"], right_on=["Season", "TeamID"])
    df = df.drop(["TeamID_x", "TeamID_y"], axis=1)

    df = df.merge(scr_1, how="left", left_on=["Season", "TeamID1"], right_on=["Season", "TeamID"])
    df = df.merge(scr_2, how="left", left_on=["Season", "TeamID2"], right_on=["Season", "TeamID"])
    df = df.drop(["TeamID_x", "TeamID_y"], axis=1)

    df["win_pct_A_diff"] = df["win_pct_A_1"] - df["win_pct_A_2"]
    df["win_pct_N_diff"] = df["win_pct_N_1"] - df["win_pct_N_2"]
    df["win_pct_H_diff"] = df["win_pct_H_1"] - df["win_pct_H_2"]
    df["win_pct_All_diff"] = df["win_pct_All_1"] - df["win_pct_All_2"]

    df["Score_A_diff"] = df["Score_A_1"] - df["Score_A_2"]
    df["Score_N_diff"] = df["Score_N_1"] - df["Score_N_2"]
    df["Score_H_diff"] = df["Score_H_1"] - df["Score_H_2"]
    df["Score_All_diff"] = df["Score_All_1"] - df["Score_All_2"]

    df["relScore_A_diff"] = df["relScore_A_1"] - df["relScore_A_2"]
    df["relScore_N_diff"] = df["relScore_N_1"] - df["relScore_N_2"]
    df["relScore_H_diff"] = df["relScore_H_1"] - df["relScore_H_2"]
    df["relScore_All_diff"] = df["relScore_All_1"] - df["relScore_All_2"]

    df = df.merge(dt_1, how="left", left_on=["Season", "TeamID1"], right_on=["Season", "TeamID"])
    df = df.merge(dt_2, how="left", left_on=["Season", "TeamID2"], right_on=["Season", "TeamID"])

    df = df.drop(["TeamID_x", "TeamID_y"], axis=1)

    df = df.merge(MOR_1, how="left", left_on=["Season", "TeamID1"], right_on=["Season", "TeamID"])
    df = df.merge(MOR_2, how="left", left_on=["Season", "TeamID2"], right_on=["Season", "TeamID"])
    df = df.drop(["TeamID_x", "TeamID_y"], axis=1)

    df["OrdinalRank_127_128_diff"] = df["OrdinalRank_127_128_1"] - df["OrdinalRank_127_128_2"]

    df = df.merge(kenpom_1, on=["Season", "TeamID1"], how="left")
    df = df.merge(kenpom_2, on=["Season", "TeamID2"], how="left")

    for col in diff_kenpom_cols:
        df[f"diff_kenpom_{col}"] = df[f"kenpom_{col}_1"] - df[f"kenpom_{col}_2"]

    df = df.fillna(-1)

    for col in df.columns:
        if (df[col] == np.inf).any() or (df[col] == -np.inf).any():
            df[col][(df[col] == np.inf) | (df[col] == -np.inf)] = -1

    return df


train = merge_data(tourney)
train = train.loc[train.Season >= 2003, :].reset_index(drop=True)

# if STAGE_1:
#     train = train.loc[train.Season < 2015, :]

if STAGE_1:
    MSampleSubmission = pd.read_csv(INPUTDIR + "/MSampleSubmissionStage1.csv")
else:
    MSampleSubmission = pd.read_csv(INPUTDIR + None)  # put stage 2 submission file link here

test1 = MSampleSubmission.copy()
test1["Season"] = test1.ID.apply(lambda x: int(x[0:4]))
test1["TeamID1"] = test1.ID.apply(lambda x: int(x[5:9]))
test1["TeamID2"] = test1.ID.apply(lambda x: int(x[10:14]))

test2 = MSampleSubmission.copy()
test2["Season"] = test2.ID.apply(lambda x: int(x[0:4]))
test2["TeamID1"] = test2.ID.apply(lambda x: int(x[10:14]))
test2["TeamID2"] = test2.ID.apply(lambda x: int(x[5:9]))

test = pd.concat([test1, test2]).drop(["Pred"], axis=1)
test = merge_data(test)

print(train.shape, test.shape)

meta_cols = ["Season", "DayNum", "TeamID1", "TeamID2", "result"]
feat_cols = [c for c in train.columns if c not in meta_cols]

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(train[feat_cols].values)
y = train.result.copy().values
X_trn, X_vld, _, _ = train_test_split(X, y, random_state=42, shuffle=True)

# TabNetPretrainer
unsupervised_model = TabNetPretrainer(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type="sparsemax",  # "sparsemax"
)
max_epochs = 1000
unsupervised_model.fit(
    X_train=X_trn,
    eval_set=[X_vld],
    max_epochs=max_epochs,
    patience=10,
    batch_size=512,
    virtual_batch_size=64,
    num_workers=0,
    drop_last=False,
    pretraining_ratio=0.8,
)

unsupervised_model.save_model(f"{OUTPUTDIR}/pretrain")

preds = train.result.copy().values
scores = []
importances = pd.DataFrame()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (trn_index, vld_index) in enumerate(kfold.split(X, y)):
    X_trn, y_trn = X[trn_index], y[trn_index]
    X_vld, y_vld = X[vld_index], y[vld_index]
    print(f"Fold {fold}, data split")
    print(f"     Shape of train x, y = {X_trn.shape}, {y_trn.shape}")
    print(f"     Shape of valid x, y = {X_vld.shape}, {y_vld.shape}")

    loaded_pretrain = TabNetPretrainer()
    loaded_pretrain.load_model(f"{OUTPUTDIR}/pretrain.zip")
    clf = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-1),
        scheduler_params={"step_size": 10, "gamma": 0.9},  # how to use learning rate scheduler
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="sparsemax",  # This will be overwritten if using pretrain model
    )

    clf.fit(
        X_train=X_trn,
        y_train=y_trn,
        eval_set=[(X_trn, y_trn), (X_vld, y_vld)],
        eval_name=["train", "valid"],
        eval_metric=["logloss"],
        max_epochs=max_epochs,
        patience=100,
        batch_size=128,
        virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False,
        from_unsupervised=loaded_pretrain,
    )

    fold_preds = clf.predict_proba(X_vld).astype(np.float64)[:, 1]
    preds[vld_index] = fold_preds
    scores.append(log_loss(y_vld, fold_preds))
    importances = pd.concat(
        [importances, pd.DataFrame({"feature": feat_cols, "importance": clf.feature_importances_})],
        axis=0,
    )


fig, ax = plt.subplots(figsize=(6, 18))
sns.barplot(
    data=importances,
    x="importance",
    y="feature",
    order=importances.groupby("feature").importance.mean().sort_values(ascending=False).index,
)
plt.show()
np.mean([0.52951, 0.43893, 0.48130, 0.43857, 0.53194, 0.41477])
