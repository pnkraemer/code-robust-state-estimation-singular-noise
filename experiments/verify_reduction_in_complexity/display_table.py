import pathlib
import pickle

import pandas as pd

path = pathlib.Path(__file__).parent.resolve()
with open(f"{path}/data_runtimes.pkl", "wb") as f:
    pickle.dump(data, f)

with open("./experiments/data.pkl", "rb") as f:
    data = pickle.load(f)

frame = pd.DataFrame.from_dict(data)

print(frame.to_latex(float_format="{:.2f}".format))
