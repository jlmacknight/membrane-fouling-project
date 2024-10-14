import numpy as np
import matplotlib.pyplot as plt


def plot_fitted_models(
    tdata: np.ndarray, vdata: np.ndarray, models: list[BaseModel]
) -> None:
    """
    Plot experimental data along with the fitted models.

    Parameters:
        tdata (numpy.ndarray): Experimental time data.
        vdata (numpy.ndarray): Experimental volume data.
        models (list[BaseModel]): List of fitted models.

    Returns:
        None
    """
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
