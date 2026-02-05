# %%
from pathlib import Path
import pandas as pd
from pybaseball import statcast, cache

cache.enable()
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", None)

# %%
START_DT = "2023-04-01"
END_DT = "2023-10-01"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = RAW_DIR / f"statcast_{START_DT}_{END_DT}.parquet"

# %%
if RAW_PATH.exists():
    df_raw = pd.read_parquet(RAW_PATH)
    print("Loaded cached raw:", RAW_PATH, df_raw.shape)
else:
    df_raw = statcast(START_DT, END_DT).copy()
    df_raw.to_parquet(RAW_PATH, index=False)
    print("Pulled + saved raw:", RAW_PATH, df_raw.shape)

# %%
print(df_raw.columns.tolist())
print(df_raw.head())
#%%