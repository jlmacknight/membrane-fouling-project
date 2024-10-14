from dataclasses import dataclass
from typing import Optional
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import numpy as np


@dataclass
class BaseModel:
    """
    Class for membrane fouling models. Provides functionality for
    fitting models, predicting values, and calculating mean squared
    error.

    Attributes:
        tdata (numpy.ndarray): Experimental time data.
        vdata (numpy.ndarray): Experimental volume filtered data.
        J0 (float): Initial experimental volumetric flow rate.
        params (Optional[numpy.ndarray]): Fitted model parameters after curve fitting (initialized as None).
        pcov (Optional[numpy.ndarray]): Covariance matrix of the parameters after curve fitting (initialized as None).
        _mse (Optional[float]): Mean squared error of the model (initialized as None).
        param_names (tuple[str]): Names of the model parameters.
    
    Methods:
        fit() -> None:
            Fits the model to the experimental data using curve fitting.
            Handles exceptions such as runtime errors.

        predict(t: numpy.ndarray) -> numpy.ndarray:
            Predicts volume filtered data using the fitted model and provided time data.

        mse() -> float:
            Calculates and returns the mean squared error (MSE) of the model's predictions
            against the actual experimental data.
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

        Parameters:
            None

        Returns:
            None
        """
        try:
            bounds = (0, np.inf)
            self.params, self.pcov = curve_fit(
                self.model, self.tdata, self.vdata, bounds=bounds
            )
        except RuntimeError as e:
            print(f"RuntimeError fitting model {self.__class__.__name__}: {e}")
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
            raise ValueError(f"Model {self.__class__.__name__} is not fitted yet.")
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
            name: f"{value:.3e}" for name, value in zip(self.param_names, self.params)
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
            1
            - np.exp(
                (-Kb / (Kc * self.J0**2)) * (np.sqrt(1 + 2 * Kc * (self.J0**2) * t) - 1)
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
            1 + (Ki / (Kc * self.J0)) * ((1 + 2 * Kc * (self.J0**2) * t) ** 0.5 - 1)
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
            (4 / 9) + ((4 * Ks) / (3 * Kc * self.J0)) + ((2 * (Ks**2) * t) / (3 * Kc))
        )

        alpha = (
            (8 / (27 * (beta**3)))
            + ((4 * Ks) / (3 * (beta**3) * Kc * self.J0))
            - ((4 * (Ks**2) * t) / (3 * (beta**3) * Kc))
        )

        return (2 / Ks) * (
            beta * np.cos(((2 * np.pi) / 3) - (1 / 3) * np.arccos(alpha)) + (1 / 3)
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
        return (self.J0 / Kb) * (1 - np.exp((-2 * Kb * t) / (2 + Ks * self.J0 * t)))


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
        return (1 / Ki) * np.log(1 + ((2 * Ki * self.J0 * t) / (2 + Ks * self.J0 * t)))
