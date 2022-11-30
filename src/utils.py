import pandas as pd
import constants as const
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from Dataset import Dataset
from Model import Model

plt.style.use("ggplot")


def get_country_dataset(soccer_df, country_name):

    team_home_df = soccer_df[soccer_df["home_team"] == country_name].rename(
        columns={
            "previous_points_home": "team_prematch_points",
            "previous_points_away": "opponent_prematch_points",
            "total_points_home": "team_postmatch_points",
            "home_score": "team_score",
            "away_score": "opponent_score",
            "away_team": "opponent_team",
            "country": "venu",
            "tournament": "match_status",
        }
    )
    team_home_df["home_adv"] = (~team_home_df["neutral"]).astype(float)
    team_home_df["opponent_home_adv"] = -1 * team_home_df["home_adv"]

    team_away_df = soccer_df[soccer_df["away_team"] == country_name].rename(
        columns={
            "previous_points_away": "team_prematch_points",
            "total_points_away": "team_postmatch_points",
            "previous_points_home": "opponent_prematch_points",
            "away_score": "team_score",
            "home_score": "opponent_score",
            "home_team": "opponent_team",
            "country": "venu",
            "tournament": "match_status",
        }
    )
    team_away_df["opponent_home_adv"] = (~team_away_df["neutral"]).astype(float)
    team_away_df["home_adv"] = -1 * team_away_df["opponent_home_adv"]

    team_df = pd.concat([team_home_df, team_away_df], ignore_index=True).sort_values(
        "date"
    )
    team_df.match_status = (
        team_df.match_status.map(const.MATCH_STATUS_VALUE).fillna(2).astype(float)
    )

    team_df["rating_diffs"] = (
        team_df["team_prematch_points"] - team_df["opponent_prematch_points"]
    )
    return (
        team_df[team_df.date > const.TRAIN_START]
        .drop_duplicates("date")
        .dropna(axis=1, how="any")
        .reset_index(drop=True)
    )


def fit_model(team_df, conf_p=0.95, I=0.7, sigma=0.05):
    dataset = Dataset.from_dataframe(team_df)
    model = Model(dataset, I=I, sigma=sigma, conf_p=conf_p)
    for i in tqdm(range(len(dataset))):
        model.step()
    return model


def predict_match_outcomes(
    country_models, country_dfs, country1, country2, verbose=True, condition=True
):
    lams = []
    for country, opponent in [(country1, country2), (country2, country1)]:
        model = country_models[country]

        country_rating = country_dfs[country]["team_postmatch_points"].iat[-1]
        opponent_rating = country_dfs[opponent]["team_postmatch_points"].iat[-1]
        match_status = country_dfs[country]["match_status"].iat[-1]

        # give all middle eastern countries home adv
        mid_east_countries = ["Qatar"]
        if country in mid_east_countries or opponent in mid_east_countries:
            home_adv = 2 * (country in mid_east_countries) - 1
        else:
            home_adv = 0

        features = np.array(
            [[country_rating - opponent_rating, home_adv, match_status]]
        ).T

        if condition:
            country_vs_opponent_df = country_dfs[country][
                country_dfs[country]["opponent_team"] == opponent
            ]
            model.add_conditioning_data(country_vs_opponent_df)
            for _ in tqdm(range(len(country_vs_opponent_df))):
                model.step()

        lam_country = model.predict(features, preprocess=True)[1]
        lams.append(lam_country)
        if condition:
            model.remove_conditioning_data()

    country1_win_prob = 1 - sp.stats.skellam.cdf(0, *lams)
    country2_win_prob = 1 - sp.stats.skellam.cdf(0, lams[1], lams[0])
    draw_prob = 1 - country1_win_prob - country2_win_prob

    if verbose:
        _plot_match_outcomes(lams, country1, country2)
    return country1_win_prob, country2_win_prob, draw_prob


def _plot_match_outcomes(lams, country1, country2):

    country1_win_prob = 1 - sp.stats.skellam.cdf(0, *lams)
    country2_win_prob = 1 - sp.stats.skellam.cdf(0, lams[1], lams[0])
    draw_prob = 1 - country1_win_prob - country2_win_prob
    print(f"{country1} win prob: {country1_win_prob*100:.3f}%")
    print(f"{country2} win prob: {country2_win_prob*100:.3f}%")
    print(f"Draw prob: {draw_prob*100:.3f}%")

    print(
        f"{country1} win by >= 2 goals: {(1- sp.stats.skellam.cdf(1, *lams))*100:.3f}%"
    )
    print(
        f"{country2} win by >= 2 goals: {(1- sp.stats.skellam.cdf(1, lams[1], lams[0]))*100:.3f}%"
    )

    plt.bar(range(-6, 7), sp.stats.skellam.pmf(range(-6, 7), *lams), color="C1")
    plt.xlabel(f"{country1}'s score - {country2}'s score")
    plt.ylabel("probability")
    plt.title("Predicted score difference")
    plt.show()

    prob_outcomes = np.outer(
        sp.stats.poisson.pmf(range(6), lams[0]),
        sp.stats.poisson.pmf(range(6), lams[1]),
    )
    # top 3 outcomes
    # top_n = np.unravel_index(np.argsort(prob_outcomes.ravel())[-3:], prob_outcomes.shape)
    # labels = np.zeros(prob_outcomes.shape)
    # labels[top_n] = 1
    # labels = labels * prob_outcomes

    sns.heatmap(
        prob_outcomes, annot=True, fmt=".1%", cbar=False, cmap="Blues", vmin=0.08
    )
    plt.ylabel(f"{country1}'s score")
    plt.xlabel(f"{country2}'s score")
    plt.show()
