import pathlib
import pickle

import pandas as pd

path = pathlib.Path(__file__).parent.resolve()

with open(f"{path}/data_runtimes.pkl", "rb") as f:
    data = pickle.load(f)

frame = pd.DataFrame.from_dict(data)

frame = frame.iloc[:, 3:]  # ignore the first few columns
print()
print(frame)
print()
print(frame.to_latex(float_format="{:.2f}".format))
print()