import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Model import Model
from Dataset import Dataset

np.random.seed(12)
np.seterr(divide="raise")
plt.style.use("ggplot")


if __name__ == "__main__":
    dataset = Dataset.generate_fake_dataset(
        100, alpha1=2, alpha2=0.1, sigma=0.04, I=-0.3
    )

    print("input variables are standardized.")
    print("Rank diff\tHome Adv\tMatch Status\tIota\tLambda\tGoals")
    for i in range(10):
        print(
            f"{dataset.features[0][i]:.3f}\t\t{dataset.features[1][i]:.3f}\t\t{dataset.features[2][i]:.3f}\t\t{dataset.true_iotas[i]:.3f}"
            + f"\t{dataset.true_lams[i]:.3f}\t{dataset.goals[i]}"
        )

    model = Model(dataset, I=0.0, sigma=0.03)
    for i in tqdm(range(len(dataset))):
        model.step()

    model.plot_params()
