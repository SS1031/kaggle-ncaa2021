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

team_quality = team_quality_features(regular_data, seeds)
team_quality_T1 = team_quality.copy()
team_quality_T1.columns = ["T1_TeamID", "T1_quality", "Season"]
team_quality_T2 = team_quality.copy()
team_quality_T2.columns = ["T2_TeamID", "T2_quality", "Season"]
tourney_data = pd.merge(tourney_data, team_quality_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, team_quality_T2, on=["Season", "T2_TeamID"], how="left")

seeds_features = create_seeds_features(tourney_data, seeds)
tourney_data = pd.merge(
    tourney_data, seeds_features, on=["Season", "T1_TeamID", "T2_TeamID"], how="left"
)
