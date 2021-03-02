# Original from raddar's paris madness https://www.kaggle.com/raddar/paris-madness

import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from scipy.interpolate import UnivariateSpline
import statsmodels.api as sm
import matplotlib.pyplot as plt
import collections


INPUTDIR = "../data/ncaam-march-mania-2021"

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


regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)

# Feature Engineering
meta_cols = ["Season", "DayNum", "T1_TeamID", "T2_TeamID", "NumOT", "Loc"]
boxscore_cols = [col for col in regular_data.columns if col not in meta_cols]

boxscore_cols2 = [
    "T1_Score",
    "T2_Score",
    "T1_FGM",
    "T1_FGA",
    "T1_FGM3",
    "T1_FGA3",
    "T1_FTM",
    "T1_FTA",
    "T1_OR",
    "T1_DR",
    "T1_Ast",
    "T1_TO",
    "T1_Stl",
    "T1_Blk",
    "T1_PF",
    "T2_FGM",
    "T2_FGA",
    "T2_FGM3",
    "T2_FGA3",
    "T2_FTM",
    "T2_FTA",
    "T2_OR",
    "T2_DR",
    "T2_Ast",
    "T2_TO",
    "T2_Stl",
    "T2_Blk",
    "T2_PF",
    "PointDiff",
]


funcs = [np.mean]
season_statistics = regular_data.groupby(["Season", "T1_TeamID"])[boxscore_cols].agg(funcs)
season_statistics.columns = ["_".join(col).strip() for col in season_statistics.columns.values]
season_statistics = season_statistics.reset_index()
print(season_statistics.head())

season_statistics_T1 = season_statistics.copy()
season_statistics_T2 = season_statistics.copy()

season_statistics_T1.columns = [
    "T1_" + x.replace("T1_", "").replace("T2_", "oppo_") if x != "Season" else x
    for x in season_statistics_T1.columns
]
season_statistics_T2.columns = [
    "T2_" + x.replace("T1_", "").replace("T2_", "oppo_") if x != "Season" else x
    for x in season_statistics_T2.columns
]


season_statistics_T1.info()
season_statistics_T2.info()

tourney_data = tourney_data[["Season", "DayNum", "T1_TeamID", "T1_Score", "T2_TeamID", "T2_Score"]]
tourney_data = pd.merge(tourney_data, season_statistics_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, season_statistics_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data.info()

# 最後の14日間の勝率
last14days_stats_T1 = regular_data.loc[regular_data.DayNum > 118].reset_index(drop=True)
last14days_stats_T1["win"] = np.where(last14days_stats_T1["PointDiff"] > 0, 1, 0)
last14days_stats_T1 = (
    last14days_stats_T1.groupby(["Season", "T1_TeamID"])["win"]
    .mean()
    .reset_index(name="T1_win_pct_last14d")
)
last14days_stats_T2 = regular_data.loc[regular_data.DayNum > 118].reset_index(drop=True)
last14days_stats_T2["win"] = np.where(last14days_stats_T2["PointDiff"] < 0, 1, 0)
last14days_stats_T2 = (
    last14days_stats_T2.groupby(["Season", "T2_TeamID"])["win"]
    .mean()
    .reset_index(name="T2_win_pct_last14d")
)
tourney_data = pd.merge(tourney_data, last14days_stats_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, last14days_stats_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data.info()

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


glm_quality = pd.concat(
    [
        team_quality(2010),
        team_quality(2011),
        team_quality(2012),
        team_quality(2013),
        team_quality(2014),
        team_quality(2015),
        team_quality(2016),
        team_quality(2017),
        team_quality(2018),
        team_quality(2019),
    ]
).reset_index(drop=True)

glm_quality_T1 = glm_quality.copy()
glm_quality_T2 = glm_quality.copy()
glm_quality_T1.columns = ["T1_TeamID", "T1_quality", "Season"]
glm_quality_T2.columns = ["T2_TeamID", "T2_quality", "Season"]

tourney_data = pd.merge(tourney_data, glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, glm_quality_T2, on=["Season", "T2_TeamID"], how="left")

seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))
seeds.info()

seeds_T1 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T2 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]

tourney_data = pd.merge(tourney_data, seeds_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, seeds_T2, on=["Season", "T2_TeamID"], how="left")

tourney_data["Seed_diff"] = tourney_data["T1_seed"] - tourney_data["T2_seed"]

y = tourney_data["T1_Score"] - tourney_data["T2_Score"]
print("Objective describe: \n", y.describe())

features = (
    [c for c in season_statistics_T1.columns if c != "Season"]
    + [c for c in season_statistics_T2.columns if c != "Season"]
    + [c for c in seeds_T1.columns if c != "Season"]
    + [c for c in seeds_T2.columns if c != "Season"]
    + [c for c in last14days_stats_T1.columns if c != "Season"]
    + [c for c in last14days_stats_T2.columns if c != "Season"]
    + ["Seed_diff"]
    + ["T1_quality", "T2_quality"]
)

##
# Modeling
##

X = tourney_data[features].values
dtrain = xgb.DMatrix(X, label=y)


def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000
    x = preds - labels
    grad = x / (x ** 2 / c ** 2 + 1)
    hess = -(c ** 2) * (x ** 2 - c ** 2) / (x ** 2 + c ** 2) ** 2
    return grad, hess


param = {}
# param['objective'] = 'reg:linear'
param["eval_metric"] = "mae"
param["booster"] = "gbtree"
param["eta"] = 0.05  # change to ~0.02 for final run
param["subsample"] = 0.35
param["colsample_bytree"] = 0.7
param["num_parallel_tree"] = 3  # recommend 10
param["min_child_weight"] = 40
param["gamma"] = 10
param["max_depth"] = 3

print(param)


##
# best_iterationを取得するために
##
xgb_cv = []
repeat_cv = 3  # recommend 10
for i in range(repeat_cv):
    print(f"Fold repeater {i}")
    xgb_cv.append(
        xgb.cv(
            params=param,
            dtrain=dtrain,
            obj=cauchyobj,
            num_boost_round=3000,
            folds=KFold(n_splits=5, shuffle=True, random_state=i),
            early_stopping_rounds=25,
            verbose_eval=50,
        )
    )

iteration_counts = [np.argmin(x["test-mae-mean"].values) for x in xgb_cv]
val_mae = [np.min(x["test-mae-mean"].values) for x in xgb_cv]
print(iteration_counts, val_mae)

oof_preds = []
for i in range(repeat_cv):
    print(f"Fold repeater {i}")
    preds = y.copy()
    kfold = KFold(n_splits=5, shuffle=True, random_state=i)
    for train_index, val_index in kfold.split(X, y):
        dtrain_i = xgb.DMatrix(X[train_index], label=y[train_index])
        dval_i = xgb.DMatrix(X[val_index], label=y[val_index])
        model = xgb.train(
            params=param, dtrain=dtrain_i, num_boost_round=iteration_counts[i], verbose_eval=50
        )
        preds[val_index] = model.predict(dval_i)
    oof_preds.append(np.clip(preds, -30, 30))

plot_df = pd.DataFrame({"pred": oof_preds[0], "label": np.where(y > 0, 1, 0)})
plot_df["pred_int"] = plot_df["pred"].astype(int)
plot_df = plot_df.groupby("pred_int")["label"].mean().reset_index(name="average_win_pct")
plt.figure()
plt.plot(plot_df.pred_int, plot_df.average_win_pct)

spline_model = []
for i in range(repeat_cv):
    dat = list(zip(oof_preds[i], np.where(y > 0, 1, 0)))
    dat = sorted(dat, key=lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]] = dat[k][1]
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])
    print(f"logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}")

plot_df = pd.DataFrame(
    {"pred": oof_preds[0], "label": np.where(y > 0, 1, 0), "spline": spline_model[0](oof_preds[0])}
)
plot_df["pred_int"] = (plot_df["pred"]).astype(int)
plot_df = plot_df.groupby("pred_int")["spline", "label"].mean().reset_index()

plt.figure()
plt.plot(plot_df.pred_int, plot_df.spline)
plt.plot(plot_df.pred_int, plot_df.label)

spline_model = []
for i in range(repeat_cv):
    dat = list(zip(oof_preds[i], np.where(y > 0, 1, 0)))
    dat = sorted(dat, key=lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]] = dat[k][1]
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])
    spline_fit = np.clip(spline_fit, 0.025, 0.975)
    print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y > 0, 1, 0), spline_fit)}")

spline_model = []
for i in range(repeat_cv):
    dat = list(zip(oof_preds[i], np.where(y > 0, 1, 0)))
    dat = sorted(dat, key=lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]] = dat[k][1]
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])
    spline_fit = np.clip(spline_fit, 0.025, 0.975)
    print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y > 0, 1, 0), spline_fit)}")

print(
    # looking for upsets
    pd.concat(
        [
            tourney_data[
                (tourney_data.T1_seed == 1)
                & (tourney_data.T2_seed == 16)
                & (tourney_data.T1_Score < tourney_data.T2_Score)
            ],
            tourney_data[
                (tourney_data.T1_seed == 2)
                & (tourney_data.T2_seed == 15)
                & (tourney_data.T1_Score < tourney_data.T2_Score)
            ],
            tourney_data[
                (tourney_data.T1_seed == 3)
                & (tourney_data.T2_seed == 14)
                & (tourney_data.T1_Score < tourney_data.T2_Score)
            ],
            tourney_data[
                (tourney_data.T1_seed == 4)
                & (tourney_data.T2_seed == 13)
                & (tourney_data.T1_Score < tourney_data.T2_Score)
            ],
            tourney_data[
                (tourney_data.T1_seed == 16)
                & (tourney_data.T2_seed == 1)
                & (tourney_data.T1_Score > tourney_data.T2_Score)
            ],
            tourney_data[
                (tourney_data.T1_seed == 15)
                & (tourney_data.T2_seed == 2)
                & (tourney_data.T1_Score > tourney_data.T2_Score)
            ],
            tourney_data[
                (tourney_data.T1_seed == 14)
                & (tourney_data.T2_seed == 3)
                & (tourney_data.T1_Score > tourney_data.T2_Score)
            ],
            tourney_data[
                (tourney_data.T1_seed == 13)
                & (tourney_data.T2_seed == 4)
                & (tourney_data.T1_Score > tourney_data.T2_Score)
            ],
        ]
    )
)

val_cv = []
spline_model = []

for i in range(repeat_cv):
    dat = list(zip(oof_preds[i], np.where(y > 0, 1, 0)))
    dat = sorted(dat, key=lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]] = dat[k][1]
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])
    spline_fit = np.clip(spline_fit, 0.025, 0.975)

    val_cv.append(
        pd.DataFrame(
            {"y": np.where(y > 0, 1, 0), "pred": spline_fit, "season": tourney_data.Season}
        )
    )
    print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}")

val_cv = pd.concat(val_cv)
print(val_cv.groupby("season").apply(lambda x: log_loss(x.y, x.pred)))


##
# Create Submission
##
sub = pd.read_csv(f"{INPUTDIR}/MSampleSubmissionStage1.csv")
# sub = pd.read_csv(f"{INPUTDIR}/MSampleSubmissionStage2.csv")
sub["Season"] = 2018
sub["T1_TeamID"] = sub["ID"].apply(lambda x: x[5:9]).astype(int)
sub["T2_TeamID"] = sub["ID"].apply(lambda x: x[10:14]).astype(int)
sub = pd.merge(sub, season_statistics_T1, on=["Season", "T1_TeamID"])
sub = pd.merge(sub, season_statistics_T2, on=["Season", "T2_TeamID"])
sub = pd.merge(sub, glm_quality_T1, on=["Season", "T1_TeamID"])
sub = pd.merge(sub, glm_quality_T2, on=["Season", "T2_TeamID"])
sub = pd.merge(sub, seeds_T1, on=["Season", "T1_TeamID"])
sub = pd.merge(sub, seeds_T2, on=["Season", "T2_TeamID"])
sub = pd.merge(sub, last14days_stats_T1, on=["Season", "T1_TeamID"])
sub = pd.merge(sub, last14days_stats_T2, on=["Season", "T2_TeamID"])
sub["Seed_diff"] = sub["T1_seed"] - sub["T2_seed"]
Xsub = sub[features].values
dtest = xgb.DMatrix(Xsub)

sub_models = []
for i in range(repeat_cv):
    print(f"Fold repeater {i}")
    sub_models.append(
        xgb.train(
            params=param,
            dtrain=dtrain,
            num_boost_round=int(iteration_counts[i] * 1.05),
            verbose_eval=50,
        )
    )

sub_preds = []
for i in range(repeat_cv):
    sub_preds.append(
        np.clip(spline_model[i](np.clip(sub_models[i].predict(dtest), -30, 30)), 0.025, 0.975)
    )

sub["Pred"] = pd.DataFrame(sub_preds).mean(axis=0)
