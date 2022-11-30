import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import path

from src.Model import Model
from src.Dataset import Dataset
import src.constants as const
import src.utils as utils

np.random.seed(12)
np.seterr(divide="raise")
plt.style.use("ggplot")


if __name__ == "__main__":
    # dataset = Dataset.generate_fake_dataset(
    #     100, alpha1=2, alpha2=0.1, sigma=0.04, I=-0.3
    # )

    # print("input variables are standardized.")
    # print("Rank diff\tHome Adv\tMatch Status\tIota\tLambda\tGoals")
    # for i in range(10):
    #     print(
    #         f"{dataset.features[0][i]:.3f}\t\t{dataset.features[1][i]:.3f}\t\t"
    #         + f"{dataset.features[2][i]:.3f}\t\t{dataset.true_iotas[i]:.3f}"
    #         + f"\t{dataset.true_lams[i]:.3f}\t{dataset.goals[i]}"
    #     )

    data_path = "data/soccer_df.csv"
    if path.isfile(data_path):
        soccer_df_path = data_path
    else:
        soccer_df_path = const.SOCCER_DATA_URL
    soccer_df = pd.read_csv(
        const.SOCCER_DATA_URL, index_col=False, parse_dates=["date"]
    )
    soccer_df.to_csv(data_path)
    country_dfs = {
        "Brazil": utils.get_country_dataset(soccer_df, "Brazil"),
        "Argentina": utils.get_country_dataset(soccer_df, "Argentina"),
    }

    country_models = {
        "Brazil": utils.fit_model(
            country_dfs["Brazil"][country_dfs["Brazil"].date > "2021-01-01"]
        ),
        "Argentina": utils.fit_model(
            country_dfs["Argentina"][country_dfs["Argentina"].date > "2021-01-01"]
        ),
    }

    utils.predict_match_outcomes(
        country_models,
        country_dfs,
        "Brazil",
        "Argentina",
        verbose=True,
        condition=True,
    )
