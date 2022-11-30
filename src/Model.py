from functools import partial
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import constants as const
from Dataset import Dataset


class Model:
    def __init__(
        self,
        dataset: Dataset,
        sigma: float,
        I: float,
        iota_range=const.IOTA_RANGE,
        alpha1_range=const.ALPHA1_RANGE,
        alpha2_range=const.ALPHA2_RANGE,
        conf_p=const.CONFIDENCE_LEVEL,
    ):
        self.dataset = dataset
        self.sigma = sigma
        self.I = I
        self.param_ranges = [
            jnp.linspace(*alpha1_range),
            jnp.linspace(*alpha2_range),
            jnp.linspace(*iota_range),
        ]
        self.param_names = ["alpha1", "alpha2", "iota_n"]
        self.conf_p = conf_p

        iota_prior = jsp.stats.norm.logpdf(self.param_ranges[2], I, 0.7)
        alpha1_prior = jsp.stats.norm.logpdf(self.param_ranges[0], 0, 2)
        alpha2_prior = jsp.stats.norm.logpdf(self.param_ranges[1], 0, 2)

        self.log_posterior = (
            alpha1_prior.reshape(-1, 1, 1)  # alpha1 prior
            + alpha2_prior.reshape(1, -1, 1)  # alpha2 prior
            + iota_prior.reshape(1, 1, -1)  # iota0 prior
        )
        self.iota_drift_pdf = jsp.stats.norm.pdf(
            self.param_ranges[2].reshape(-1, 1),
            self.param_ranges[2].reshape(1, -1),
            self.sigma,
        )

        self.index = 0

        self.argmap_estimate = []
        self.arglow_estimate = []
        self.arghigh_estimate = []
        self.next_match_lam_pred = []

    def step(self):
        log_prior = self.log_posterior
        goal = self.dataset.goals[self.index]
        self.log_posterior = self._next_posterior(
            log_prior,
            self.dataset.features[:3, self.index],
            goal,
        )
        self._estimate_params()

        self.index += 1
        if self.index < len(self.dataset):
            lams = self.predict(self.dataset.features[:, self.index : self.index + 1])
            self.next_match_lam_pred.append(lams)

    def append_dataset(self, features, goals, dates):
        self.dataset.append(features, goals, dates, preprocess=True)
        # next match lambda is not predicted for the last match
        # of the training dataset
        lams = self.predict(self.dataset.features[:, self.index : self.index + 1])
        self.next_match_lam_pred.append(lams)

    @partial(jax.jit, static_argnums=(0,))
    def _next_posterior(self, log_prior, features, goal):
        rating_diff, home_adv, match_status = features

        lam = jnp.exp(
            self.param_ranges[2].reshape(1, 1, -1)
            + self.param_ranges[0].reshape(-1, 1, 1) * rating_diff
            + self.param_ranges[1].reshape(1, -1, 1) * match_status
        )
        log_likelihood = jsp.stats.poisson.logpmf(goal, lam)
        posterior = jnp.exp(log_prior + log_likelihood) @ self.iota_drift_pdf
        # normalize so that pdf sums to 1
        posterior /= jnp.sum(posterior)

        # add a small epsilon to prevent overflow in log
        log_posterior = jnp.log(posterior + 1e-500)
        return log_posterior

    @partial(jax.jit, static_argnums=(0, 2))
    def _get_margin_cdf(self, log_posterior, sum_axes):
        margin = jnp.sum(jnp.exp(log_posterior), axis=sum_axes)
        return jnp.cumsum(margin / jnp.sum(margin))

    def _estimate_param(self, log_posterior, sum_axes):
        margin_cdf = self._get_margin_cdf(log_posterior, sum_axes)
        arglow_estimate = np.argmax(margin_cdf > (1 - self.conf_p) / 2)
        arghigh_estimate = (
            np.argmax(margin_cdf > self.conf_p + (1 - self.conf_p) / 2) - 1
        )
        return arglow_estimate, arghigh_estimate

    def _estimate_params(self):
        argmap_estimate = np.unravel_index(
            np.argmax(self.log_posterior, axis=None), self.log_posterior.shape
        )

        iota_range_estimate = self._estimate_param(self.log_posterior, sum_axes=(0, 1))
        alpha1_range_estimate = self._estimate_param(
            self.log_posterior, sum_axes=(1, 2)
        )
        alpha2_range_estimate = self._estimate_param(
            self.log_posterior, sum_axes=(0, 2)
        )

        arglow_estimate, arghigh_estimate = tuple(
            zip(*[alpha1_range_estimate, alpha2_range_estimate, iota_range_estimate])
        )

        self.argmap_estimate.append(argmap_estimate)
        self.arglow_estimate.append(arglow_estimate)
        self.arghigh_estimate.append(arghigh_estimate)

    def predict(self, features, preprocess=False):
        if preprocess:
            features = self.dataset.preprocess(features)

        lams = [
            np.exp(
                self.param_ranges[0][arg[0]] * features[0, 0]  # rating diff
                + self.param_ranges[1][arg[1]] * features[2, 0]  # match status
                + self.param_ranges[2][arg[2]]
            )
            for arg in [
                self.arglow_estimate[-1],
                self.argmap_estimate[-1],
                self.arghigh_estimate[-1],
            ]
        ]
        return lams

    def plot_params(self):
        fig, ax = plt.subplots(4, 1)
        fig.tight_layout(pad=1)

        if self.dataset.true_params:
            true_params = [
                self.dataset.alpha1 * np.ones(self.index),
                self.dataset.alpha2 * np.ones(self.index),
                self.dataset.true_iotas,
            ]
        else:
            true_params = None
        for i, (ax_i, p_range, p_name) in enumerate(
            zip(ax[:-1], self.param_ranges, self.param_names)
        ):
            if self.dataset.true_params:
                ax_i.plot(
                    self.dataset.dates[: self.index],
                    true_params[i][: self.index],
                    "-",
                    label="true param",
                )

            ax_i.plot(
                self.dataset.dates[: self.index],
                [p_range[arg[i]] for arg in self.argmap_estimate],
                "--",
                label=f"map est.",
            )

            ax_i.fill_between(
                self.dataset.dates[: self.index],
                [p_range[arg[i]] for arg in self.arglow_estimate],
                [p_range[arg[i]] for arg in self.arghigh_estimate],
                label=f"{self.conf_p * 100:.2f}% density",
                alpha=0.4,
                color="C1",
            )

            ax_i.set_title(p_name, fontsize=10)

        # plot next match predicted lambda
        x = self.dataset.dates[1 : self.index + 1]

        if self.dataset.true_params:
            ax[3].plot(
                x, self.dataset.true_lams[1 : self.index + 1], label="true param"
            )
        ax[3].plot(
            x, list(zip(*(self.next_match_lam_pred)))[1], "--", label="MLE estimate"
        )
        ax[3].fill_between(
            x,
            list(zip(*(self.next_match_lam_pred)))[0],
            list(zip(*(self.next_match_lam_pred)))[2],
            label=f"{self.conf_p * 100:.2f}% density",
            color="C1",
            alpha=0.4,
        )
        ax[3].set_title("next match lambda_n", fontsize=10)
        ax[3].set_ylim((0, 6))

        fig.set_figheight(8)
        fig.set_figwidth(10)
        fig.legend(*(ax[0].get_legend_handles_labels()), loc="upper left")
        plt.show()
