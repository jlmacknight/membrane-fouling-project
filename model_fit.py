import numpy as np
import matplotlib.pyplot as plt
from models import (
    BaseModel,
    CakeCompleteModel,
    CakeIntermediateModel,
    CakeStandardModel,
    CompleteStandardModel,
    IntermediateStandardModel,
)
from data_handler import load_experimental_data
from visualization import plot_fitted_models

# Provide a value for the initial volumetric flow rate, J0 (m³/s)
J0 = 0.0008


def fit_and_evaluate_models(tdata: np.ndarray, vdata: np.ndarray, J0: float) -> list:
    """
    Fit all models to the data and calculate their MSE.

    Parameters:
        tdata (numpy.ndarray): Time data.
        vdata (numpy.ndarray): Volume filtered data.
        J0 (float): Initial volumetric flow rate.

    Returns:
        list: A list of fitted models.
    """
    models = [Model(tdata, vdata, J0) for Model in BaseModel.__subclasses__()]

    for model in models:
        model.fit()
        model.mse()
        print(model)

    return models


def main():

    tdata, vdata = load_experimental_data("data.csv")
    models = fit_and_evaluate_models(tdata, vdata, J0)
    plot_fitted_models(tdata, vdata, models)


if __name__ == "__main__":
    main()
