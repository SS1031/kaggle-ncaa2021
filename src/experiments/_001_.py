import collections
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import xgboost as xgb
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

import seaborn as sns


expname = os.path.basename(__file__)


INPUTDIR = "../data/ncaam-march-mania-2021"
OUTPUTDIR = f"../output/{expname}"

if not os.path.exists(OUTPUTDIR):
    os.makedirs(OUTPUTDIR)

pd.set_option("display.max_column", 999)

print("Show data list")
print(os.listdir(INPUTDIR))

regular_results = pd.read_csv(f"{INPUTDIR}/MRegularSeasonDetailedResults.csv")
tourney_results = pd.read_csv(f"{INPUTDIR}/MNCAATourneyDetailedResults.csv")
seeds = pd.read_csv(f"{INPUTDIR}/MNCAATourneySeeds.csv")


def prepare_data(df):
    """Teamを
    T1, T2 が [勝つ/負ける] 両方のサンプル

    """
    dfswap = df[
        [
            "Season",
            "DayNum",
            "LTeamID",
            "LScore",
            "WTeamID",
            "WScore",
            "WLoc",
            "NumOT",
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
        ]
    ]

    dfswap.loc[df["WLoc"] == "H", "WLoc"] = "A"
    dfswap.loc[df["WLoc"] == "A", "WLoc"] = "H"
    df.columns.values[6] = "Loc"
    dfswap.columns.values[6] = "Loc"

    df.columns = [
        x if x == "Loc" else x.replace("W", "T1_").replace("L", "T2_") for x in df.columns
    ]
    dfswap.columns = [
        x if x == "Loc" else x.replace("L", "T1_").replace("W", "T2_") for x in dfswap.columns
    ]

    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output.loc[output["Loc"] == "N", "Loc"] = "0"
    output.loc[output["Loc"] == "H", "Loc"] = "1"
    output.loc[output["Loc"] == "A", "Loc"] = "-1"
    output.Loc = output.Loc.astype(int)

    output["PointDiff"] = output["T1_Score"] - output["T2_Score"]

    return output


# Feature Engineering
def regular_season_features(regular_data):
    meta_cols = ["Season", "DayNum", "T1_TeamID", "T2_TeamID", "NumOT", "Loc"]
    boxscore_cols = [col for col in regular_data.columns if col not in meta_cols]

    season_stats = regular_data.groupby(["Season", "T1_TeamID"])[boxscore_cols].agg([np.mean])
    season_stats.columns = ["_".join(col).strip() for col in season_stats.columns.values]
    season_stats = season_stats.reset_index()

    season_stats_T1 = season_stats.copy()
    season_stats_T2 = season_stats.copy()

    season_stats_T1.columns = [
        "T1_" + x.replace("T1_", "").replace("T2_", "oppo_") if x != "Season" else x
        for x in season_stats_T1.columns
    ]
    season_stats_T2.columns = [
        "T2_" + x.replace("T1_", "").replace("T2_", "oppo_") if x != "Season" else x
        for x in season_stats_T2.columns
    ]
    # 最後の14日間の勝率
    last14days_season_stats_T1 = regular_data.loc[regular_data.DayNum > 118].reset_index(drop=True)
    last14days_season_stats_T1["win"] = np.where(last14days_season_stats_T1["PointDiff"] > 0, 1, 0)
    last14days_season_stats_T1 = (
        last14days_season_stats_T1.groupby(["Season", "T1_TeamID"])["win"]
        .mean()
        .reset_index(name="T1_win_pct_last14d")
    )
    last14days_season_stats_T2 = regular_data.loc[regular_data.DayNum > 118].reset_index(drop=True)
    last14days_season_stats_T2["win"] = np.where(last14days_season_stats_T2["PointDiff"] < 0, 1, 0)
    last14days_season_stats_T2 = (
        last14days_season_stats_T2.groupby(["Season", "T2_TeamID"])["win"]
        .mean()
        .reset_index(name="T2_win_pct_last14d")
    )
    season_stats_T1 = season_stats_T1.merge(last14days_season_stats_T1, on=["Season", "T1_TeamID"])
    season_stats_T2 = season_stats_T2.merge(last14days_season_stats_T2, on=["Season", "T2_TeamID"])
    return season_stats_T1, season_stats_T2


def team_quality_features(regular_data, seeds):
    regular_season_effects = regular_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff"]].copy()
    regular_season_effects["T1_TeamID"] = regular_season_effects["T1_TeamID"].astype(str)
    regular_season_effects["T2_TeamID"] = regular_season_effects["T2_TeamID"].astype(str)
    regular_season_effects["win"] = np.where(regular_season_effects["PointDiff"] > 0, 1, 0)
    march_madness = pd.merge(seeds[["Season", "TeamID"]], seeds[["Season", "TeamID"]], on="Season")
    march_madness.columns = ["Season", "T1_TeamID", "T2_TeamID"]
    march_madness.T1_TeamID = march_madness.T1_TeamID.astype(str)
    march_madness.T2_TeamID = march_madness.T2_TeamID.astype(str)
    regular_season_effects = pd.merge(
        regular_season_effects, march_madness, on=["Season", "T1_TeamID", "T2_TeamID"]
    )

    def team_quality(season):
        formula = "win~-1+T1_TeamID+T2_TeamID"
        glm = sm.GLM.from_formula(
            formula=formula,
            data=regular_season_effects.loc[regular_season_effects.Season == season, :],
            family=sm.families.Binomial(),
        ).fit()

        quality = pd.DataFrame(glm.params).reset_index()
        quality.columns = ["TeamID", "quality"]
        quality["Season"] = season
        quality["quality"] = np.exp(quality["quality"])
        quality = quality.loc[quality.TeamID.str.contains("T1_")].reset_index(drop=True)
        quality["TeamID"] = quality["TeamID"].apply(lambda x: x[10:14]).astype(int)
        return quality

    glm_quality = pd.concat([team_quality(year) for year in range(2003, 2020)]).reset_index(
        drop=True
    )

    return glm_quality


def create_seeds_features(tourney_data, seeds):
    seeds_features = tourney_data[["Season", "T1_TeamID", "T2_TeamID"]].copy()
    seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))

    seeds_T1 = seeds[["Season", "TeamID", "seed"]].copy()
    seeds_T2 = seeds[["Season", "TeamID", "seed"]].copy()
    seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
    seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]

    seeds_features = pd.merge(seeds_features, seeds_T1, on=["Season", "T1_TeamID"], how="left")
    seeds_features = pd.merge(seeds_features, seeds_T2, on=["Season", "T2_TeamID"], how="left")

    seeds_features["Seed_diff"] = seeds_features["T1_seed"] - seeds_features["T2_seed"]

    return seeds_features


# Preprocess
regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)

# Feature engineering
season_stats_T1, season_stats_T2 = regular_season_features(regular_data)
tourney_data = tourney_data[["Season", "DayNum", "T1_TeamID", "T1_Score", "T2_TeamID", "T2_Score"]]
tourney_data = pd.merge(tourney_data, season_stats_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, season_stats_T2, on=["Season", "T2_TeamID"], how="left")

# team_quality = team_quality_features(regular_data, seeds)
# team_quality_T1 = team_quality.copy()
# team_quality_T1.columns = ["T1_TeamID", "T1_quality", "Season"]
# team_quality_T2 = team_quality.copy()
# team_quality_T2.columns = ["T2_TeamID", "T2_quality", "Season"]
# tourney_data = pd.merge(tourney_data, team_quality_T1, on=["Season", "T1_TeamID"], how="left")
# tourney_data = pd.merge(tourney_data, team_quality_T2, on=["Season", "T2_TeamID"], how="left")
# tourney_data["T1_quality"] = (
#     tourney_data["T1_quality"].replace([np.inf, -np.inf], np.nan).fillna(0)
# )
# tourney_data["T2_quality"] = (
#     tourney_data["T2_quality"].replace([np.inf, -np.inf], np.nan).fillna(0)
# )

# KenPom, https://www.kaggle.com/paulorzp/kenpom-scraper-2021/output
kenpom = pd.read_csv(f"{INPUTDIR}/MKenpom.csv")
# kenpom["win"] = kenpom.record.str.split("-", expend=True)[0]
# kenpom["lose"] = kenpom.record.str.split("-", expand=False)[1]
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
T1_kenpom = (
    kenpom[["Season", "TeamID"] + kenpom_cols].copy().rename(columns={"TeamID": "T1_TeamID"})
)
T1_kenpom = T1_kenpom.rename(columns={c: "T1_kenpom_" + c for c in kenpom_cols})
T2_kenpom = (
    kenpom[["Season", "TeamID"] + kenpom_cols].copy().rename(columns={"TeamID": "T2_TeamID"})
)
T2_kenpom = T2_kenpom.rename(columns={c: "T2_kenpom_" + c for c in kenpom_cols})

tourney_data = tourney_data.merge(T1_kenpom, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(T2_kenpom, on=["Season", "T2_TeamID"], how="left")

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

for col in diff_kenpom_cols:
    tourney_data[f"diff_kenpom_{col}"] = (
        tourney_data[f"T1_kenpom_{col}"] - tourney_data[f"T2_kenpom_{col}"]
    )

seeds_features = create_seeds_features(tourney_data, seeds)
tourney_data = pd.merge(
    tourney_data, seeds_features, on=["Season", "T1_TeamID", "T2_TeamID"], how="left"
)

meta_cols = ["Season", "DayNum", "T1_TeamID", "T2_TeamID", "T1_Score", "T2_Score"]
feat_cols = [c for c in tourney_data.columns if c not in meta_cols]

X = tourney_data[feat_cols].values
y = (tourney_data["T1_Score"] > tourney_data["T2_Score"]).astype(int).values

preds = (tourney_data["T1_Score"] > tourney_data["T2_Score"]).astype(np.float64).values
scores = []
importances = pd.DataFrame()
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
for fold, (trn_index, vld_index) in enumerate(kfold.split(X, y)):
    X_trn, y_trn = X[trn_index], y[trn_index]
    X_vld, y_vld = X[vld_index], y[vld_index]
    print(f"Fold {fold}, data split")
    print(f"     Shape of train x, y = {X_trn.shape}, {y_trn.shape}")
    print(f"     Shape of valid x, y = {X_vld.shape}, {y_vld.shape}")

    # TabNetPretrainer
    unsupervised_model = TabNetPretrainer(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",  # "sparsemax"
    )
    max_epochs = 1000
    unsupervised_model.fit(
        X_train=X_trn,
        eval_set=[X_vld],
        max_epochs=max_epochs,
        patience=25,
        batch_size=2048,
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False,
        pretraining_ratio=0.8,
    )

    unsupervised_model.save_model(f"{OUTPUTDIR}/test_pretrain_fold{fold}")
    loaded_pretrain = TabNetPretrainer()
    loaded_pretrain.load_model(f"{OUTPUTDIR}/test_pretrain_fold{fold}.zip")

    clf = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-1),
        scheduler_params={"gamma": 0.95},  # how to use learning rate scheduler
        scheduler_fn=torch.optim.lr_scheduler.ExponentialLR,
        mask_type="sparsemax",  # This will be overwritten if using pretrain model
    )

    clf.fit(
        X_train=X_trn,
        y_train=y_trn,
        eval_set=[(X_trn, y_trn), (X_vld, y_vld)],
        eval_name=["train", "valid"],
        eval_metric=["logloss"],
        max_epochs=max_epochs,
        patience=25,
        batch_size=128,
        virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False,
        from_unsupervised=loaded_pretrain,
    )

    fold_preds = clf.predict_proba(X_vld).astype(np.float64)
    preds[vld_index] = fold_preds[:, 1]
    scores.append(log_loss(y_vld, fold_preds[:, 1]))
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
