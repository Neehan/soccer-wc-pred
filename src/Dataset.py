import numpy as np
import src.constants as const


class Dataset:
    def __init__(
        self,
        features: np.array,
        feature_names: list,
        goals: np.array,
        dates=None,
        true_iotas=None,
        true_lams=None,
        true_params=None,
    ):
        self.features = self.preprocess(features, save_params=True)
        self.feature_names = feature_names
        self.goals = goals

        if true_params is not None:
            self.true_params = True
            self.true_iotas = true_iotas
            self.true_lams = true_lams
            self.alpha1, self.alpha2, self.sigma, self.I = tuple(true_params)
        else:
            self.true_params = False

        self.dates = (
            list(range(self.features.shape[1])) if dates is None else list(dates)
        )

    def preprocess(self, features, save_params=False):
        # features: rank diff, home adv, match_status
        features = np.copy(features)
        features[0] += 100 * features[1]

        if save_params:
            self.feature_means = np.mean(features, axis=1).reshape(-1, 1)
            self.feature_stds = np.std(features, axis=1).reshape(-1, 1)

        return (features - self.feature_means) / (
            const.FEATURE_STD_SCALE * self.feature_stds
        )

    # def append(self, df, preprocess=True):
    #     feature_names = ["rating_diffs", "home_adv", "match_status"]
    #     features = np.array(df[feature_names]).T
    #     if preprocess:
    #         features = self.preprocess(features, save_params=False)
    #     self.features = np.hstack((self.features, features))
    #     self.goals = np.hstack((self.goals, np.array(df.team_score)))
    #     self.dates += list(df.date)

    # def remove(self, idx):
    #     self.features = self.features[:, :idx]
    #     self.goals = self.goals[:idx]
    #     self.dates = self.dates[:idx]

    def __len__(self):
        return self.features.shape[1]

    @classmethod
    def generate_fake_dataset(cls, dataset_size, alpha1, alpha2, sigma, I):

        # -1: opponent has home adv
        # 0: neutral
        home_adv = np.random.choice([-1, 0, 1], dataset_size, p=[0.3, 0.4, 0.3])
        match_status = np.random.choice([2, 3, 4, 5, 6], dataset_size)
        rating_diffs = np.floor(np.random.normal(0, 400, dataset_size))

        features = np.array([rating_diffs[:], home_adv[:], match_status[:]])
        features[0] += 100 * features[1]

        features = (features - features.mean(axis=1).reshape(-1, 1)) / (
            const.FEATURE_STD_SCALE * features.std(axis=1).reshape(-1, 1)
        )

        iotas = I + np.random.normal(0, sigma, dataset_size).cumsum()
        lams = np.exp(iotas + alpha1 * features[0] + alpha2 * features[2])
        goals = np.random.poisson(lams).astype(float)
        return cls(
            np.array([rating_diffs, home_adv, match_status]),
            ["rating_diffs", "home_adv", "match_status"],
            goals,
            None,
            iotas,
            lams,
            (alpha1, alpha2, sigma, I),
        )

    @classmethod
    def from_dataframe(cls, df):
        feature_names = ["rating_diffs", "home_adv", "match_status"]
        return cls(
            np.array(df[feature_names]).T,
            feature_names,
            np.array(df.team_score),
            np.array(df.date),
        )
