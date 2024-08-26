"""
model_fit.py
--------

Description:
    This script implements and fits five different membrane fouling 
    models to experimental data. The resulting fitted models can then 
    be used to make predictions in similar systems regarding process 
    duration and membrane sizing. The models are adapted from Bolton, 
    LaCasse, and Kuriyel (2006) and are used to analyze flux decay in 
    membrane filtration systems for a constant transmembrane pressure. 
    The fitted models include:
    - Cake-complete
    - Cake-intermediate
    - Cake-standard
    - Complete-standard
    - Intermediate-standard
    Experimental data for the time, volume filtered, and initial 
    volumetric flow rate was obtained from the dataset provided by 
    Mayani et al. (2023) for lentiviral vector clarification.

References:
    - Bolton, G., LaCasse, D., and Kuriyel, R. (2006). Combined models 
      of membrane fouling: Development and application to 
      microfiltration and ultrafiltration of biological fluids. Journal 
      of Membrane Science, 277(1-2). 
      doi:https://doi.org/10.1016/j.memsci.2004.12.053.
    - Mayani, M., Nellimarla, S., Mangalathillam, R., Rao, H., 
      Patarroyo‐White, S., Ma, J., and Figueroa, B. (2023). Depth 
      filtration for clarification of intensified lentiviral vector 
      suspension cell culture. Biotechnology Progress, 40(2). 
      doi:https://doi.org/10.1002/btpr.3409.

Dependencies:
    - numpy
    - matplotlib
    - pandas
    - scipy
    - scikit-learn

Author:
    James Macknight

Date:
    2024-08-02

Usage:
    - Provide data (e.g. a CSV file in the working directory) that 
      contains volume filtered and time series data for a membrane 
      filtration system.
    - Provide a value for the initial volumetric flow rate, J0.
    - Run the script to fit the models and visualize results.
    - Adjust bounds for fit method if the curve fitting does not 
      converge on valid parameters
    - Results will include the parameters for each model and their mean 
      squared errors (MSE).
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Provide a value for the initial volumetric flow rate, J0 (m³/s)
J0: float = 0.0008

# Load the CSV file containing experimental data and extract the time 
# and volume data
data = pd.read_csv('data.csv')

tdata = data[data.filter(like='time').columns[0]].values
vdata = data[data.filter(like='volume').columns[0]].values


@dataclass
class BaseModel:
    """
    Class for membrane fouling models. Provides functionality for 
    fitting models, predicting values, and calculating mean squared 
    error.

    Parameters:
    tdata (numpy.ndarray): Time data.
    vdata (numpy.ndarray): Volume filtered data.
    J0 (float): Initial volumetric flow rate.
    """
    tdata: np.ndarray
    vdata: np.ndarray
    J0: float

    params: Optional[np.ndarray] = None
    pcov: Optional[np.ndarray] = None
    _mse: Optional[float] = None

    param_names: tuple[str] = ()

    def fit(self) -> None:
        """
        Fit the model to the data using curve fitting. Handles runtime 
        errors and other exceptions.
        """
        try:
            bounds = (0, np.inf)
            self.params, self.pcov = curve_fit(
                self.model, self.tdata, self.vdata, bounds=bounds
            )
        except RuntimeError as e:
            print(
                f"RuntimeError fitting model {self.__class__.__name__}: {e}"
            )
        except Exception as e:
            print(
                "An unexpected error occurred for model "
                f"{self.__class__.__name__}: {e}"
            )

    def predict(self, t: np.ndarray) -> np.ndarray:
        """
        Predict values using the fitted model.

        Parameters:
            t (numpy.ndarray): Time data for prediction.

        Returns:
            numpy.ndarray: Predicted volume filtered data.
        """
        if self.params is None:
            raise ValueError(
                f"Model {self.__class__.__name__} is not fitted yet."
            )
        return self.model(t, *self.params)

    def mse(self) -> float:
        """
        Calculate the mean squared error of the model predictions 
        against the actual data.

        Returns:
            float: Mean squared error.
        """
        v_pred = self.predict(self.tdata)
        self._mse = mean_squared_error(self.vdata, v_pred)
        return self._mse

    def __str__(self) -> str:
        params_formatted = {
            name: f"{value:.3e}" 
            for name, value in zip(self.param_names, self.params)
        }

        if self.params is None:
            return f"{self.__class__.__name__} model could not be fitted."

        if self._mse is None:
            return (
                f"Model: {self.__class__.__name__}\n"
                f"Parameters: {params_formatted}\n"
                f"The MSE value for {self.__class__.__name__} was not "
                "calculated"
            )

        return (
            f"Model: {self.__class__.__name__}\n"
            f"Parameters: {params_formatted}\n"
            f"MSE: {self.mse():.3e}"
        )


@dataclass
class CakeCompleteModel(BaseModel):
    """
    Model representing cake filtration, complete blocking.
    """

    param_names: tuple[str] = ("Kb", "Kc")

    def model(self, t: np.ndarray, Kb: float, Kc: float) -> np.ndarray:
        """
        Cake-complete model function.

        Parameters:
            t (numpy.ndarray): Time data.
            Kb (float): Complete blocking parameter.
            Kc (float): Cake filtration parameter.

        Returns:
            numpy.ndarray: Predicted volume filtered data.
        """
        return (self.J0 / Kb) * (
            1 - np.exp(
                (-Kb / (Kc * self.J0**2)) * (
                    np.sqrt(1 + 2 * Kc * (self.J0**2) * t) - 1
                )
            )
        )


@dataclass
class CakeIntermediateModel(BaseModel):
    """
    Model representing cake filtration, intermediate blocking.
    """

    param_names: tuple[str] = ("Kc", "Ki")

    def model(self, t: np.ndarray, Kc: float, Ki: float) -> np.ndarray:
        """
        Cake-intermediate model function.

        Parameters:
            t (numpy.ndarray): Time data.
            Kc (float): Cake filtration parameter.
            Ki (float): Intermediate blocking parameter.

        Returns:
            numpy.ndarray: Predicted volume filtered data.
        """
        return (1 / Ki) * np.log(
            1 + (Ki / (Kc * self.J0)) * (
                (1 + 2 * Kc * (self.J0**2) * t) ** 0.5 - 1
            )
        )


@dataclass
class CakeStandardModel(BaseModel):
    """
    Model representing cake filtration, standard blocking.
    """

    param_names: tuple[str] = ("Kc", "Ks")

    def model(self, t: np.ndarray, Kc: float, Ks: float) -> np.ndarray:
        """
        Cake-standard model function.

        Parameters:
            t (numpy.ndarray): Time data.
            Kc (float): Cake filtration parameter.
            Ks (float): Standard blocking parameter.

        Returns:
            numpy.ndarray: Predicted volume filtered data.
        """
        beta = np.sqrt(
            (4 / 9) + 
            ((4 * Ks) / (3 * Kc * self.J0)) + 
            ((2 * (Ks**2) * t) / (3 * Kc))
        )

        alpha = (
            (8 / (27 * (beta**3))) + 
            ((4 * Ks) / (3 * (beta**3) * Kc * self.J0)) - 
            ((4 * (Ks**2) * t) / (3 * (beta**3) * Kc))
        )

        return (2 / Ks) * (
            beta * np.cos(
                ((2 * np.pi) / 3) - (1 / 3) * np.arccos(alpha)
            ) + (1 / 3)
        )


@dataclass
class CompleteStandardModel(BaseModel):
    """
    Model representing complete blocking, standard blocking.
    """

    param_names: tuple[str] = ("Kb", "Ks")

    def model(self, t: np.ndarray, Kb: float, Ks: float) -> np.ndarray:
        """
        Complete-standard model function.

        Parameters:
            t (numpy.ndarray): Time data.
            Kb (float): Complete blocking parameter.
            Ks (float): Standard blocking parameter.

        Returns:
            numpy.ndarray: Predicted volume filtered data.
        """
        return (self.J0 / Kb) * (
            1 - np.exp(
                (-2 * Kb * t) / (2 + Ks * self.J0 * t)
            )
        )


@dataclass
class IntermediateStandardModel(BaseModel):
    """
    Model representing intermediate blocking, standard blocking.
    """

    param_names: tuple[str] = ("Ki", "Ks")

    def model(self, t: np.ndarray, Ki: float, Ks: float) -> np.ndarray:
        """
        Intermediate-standard model function.

        Parameters:
            t (numpy.ndarray): Time data.
            Ki (float): Intermediate blocking parameter.
            Ks (float): Standard blocking parameter.

        Returns:
            numpy.ndarray: Predicted volume filtered data.
        """
        return (1 / Ki) * np.log(
            1 + ((2 * Ki * self.J0 * t) / (2 + Ks * self.J0 * t))
        )


# Instantiate, fit, and output the results for each model
models: list[BaseModel] = [
    Model(tdata, vdata, J0) for Model in BaseModel.__subclasses__()
]
for model in models:
    model.fit()
    model.mse()
    print(model)


# Plot experimental data and fitted models
plt.figure(figsize=(12, 8))
t_range = np.linspace(min(tdata), max(tdata), 1000)

line_styles = [(0, (3, 5, 1, 5, 1, 5)), "--", "-.", ":", "-"]

for i, model in enumerate(models):
    plt.plot(
        t_range,
        model.predict(t_range),
        label=model.__class__.__name__,
        linestyle=line_styles[i % len(line_styles)],
        color="k",
    )

plt.plot(
    tdata,
    vdata,
    color="black",
    fillstyle="none",
    label="Experimental data",
    marker="D",
    ls="none",
    linewidth=1,
)

plt.xlabel("Time (s)")
plt.ylabel("Volume filtered (m³)")
plt.title("Fitted membrane fouling models")
plt.legend()
plt.grid(True)
plt.show()
