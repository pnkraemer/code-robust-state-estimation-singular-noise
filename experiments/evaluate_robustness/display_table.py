import pathlib
import pickle

import pandas as pd

path = pathlib.Path(__file__).parent.resolve()

with open(f"{path}/data_errors.pkl", "rb") as f:
    data = pickle.load(f)


frame = pd.DataFrame.from_dict(data)

latex = frame.to_latex(
    column_format="c" * frame.shape[1], index=False, float_format="{:.1f}".format
)

print(frame)
print()
print(latex)
print()
