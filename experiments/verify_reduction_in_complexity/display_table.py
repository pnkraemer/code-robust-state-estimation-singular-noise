import pathlib
import pickle

import pandas as pd

path = pathlib.Path(__file__).parent.resolve()

with open(f"{path}/data_runtimes.pkl", "rb") as f:
    data = pickle.load(f)

frame = pd.DataFrame.from_dict(data)
latex = frame.to_latex(
    column_format="c" * frame.shape[1], index=False, float_format="{:.2f}".format
)


# frame = frame.iloc[:, 3:]  # ignore the first few columns
print()
print(frame)
print()
print(latex)
print()
