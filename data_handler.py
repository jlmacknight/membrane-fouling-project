import pandas as pd


def load_experimental_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load time and volume data from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
            tdata (numpy.ndarray): Experimental time data.
            vdata (numpy.ndarray): Experimental volume data.
    """
    data = pd.read_csv(filepath)
    tdata = data[data.filter(like="time").columns[0]].values
    vdata = data[data.filter(like="volume").columns[0]].values
    return tdata, vdata
