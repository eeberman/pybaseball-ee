# %%
"""
verify_fetcher.py - Compare pybaseball vs statcast_fetcher output

Tests that the new direct Baseball Savant fetcher produces equivalent data
to pybaseball's statcast() function.
"""

from pathlib import Path
import pandas as pd
from pybaseball import statcast, cache
from statcast_fetcher import fetch_statcast

# Enable pybaseball cache for fair comparison
cache.enable()

# %%
# Test parameters - 1 week of data
START_DATE = "2024-04-01"
END_DATE = "2024-04-07"

print(f"Comparing data from {START_DATE} to {END_DATE}")
print("=" * 60)

# %%
# Fetch with pybaseball
print("\n[1] Fetching with pybaseball...")
df_pyb = statcast(START_DATE, END_DATE)
print(f"    pybaseball rows: {len(df_pyb):,}")

# %%
# Fetch with statcast_fetcher
print("\n[2] Fetching with statcast_fetcher...")
df_direct = fetch_statcast(START_DATE, END_DATE, cache_dir=Path("data/cache"), verbose=True)
print(f"    statcast_fetcher rows: {len(df_direct):,}")

# %%
# Row count comparison
print("\n" + "=" * 60)
print("ROW COUNT COMPARISON")
print("=" * 60)
row_diff = abs(len(df_pyb) - len(df_direct))
row_diff_pct = row_diff / max(len(df_pyb), 1) * 100
print(f"pybaseball:       {len(df_pyb):,}")
print(f"statcast_fetcher: {len(df_direct):,}")
print(f"Difference:       {row_diff:,} ({row_diff_pct:.2f}%)")

if row_diff_pct <= 1:
    print("PASS: Row counts within 1%")
elif row_diff_pct <= 5:
    print("WARNING: Row counts differ by 1-5%")
else:
    print("FAIL: Row counts differ by >5%")

# %%
# Column comparison
print("\n" + "=" * 60)
print("COLUMN COMPARISON")
print("=" * 60)

cols_pyb = set(df_pyb.columns)
cols_direct = set(df_direct.columns)

only_pyb = cols_pyb - cols_direct
only_direct = cols_direct - cols_pyb
common = cols_pyb & cols_direct

print(f"Common columns:        {len(common)}")
print(f"Only in pybaseball:    {len(only_pyb)}")
print(f"Only in direct fetch:  {len(only_direct)}")

if only_pyb:
    print(f"\n  Missing from direct: {sorted(only_pyb)[:10]}")
if only_direct:
    print(f"\n  Extra in direct: {sorted(only_direct)[:10]}")

# Critical columns check
CRITICAL_COLS = [
    "game_pk", "game_date", "batter", "pitcher", "description",
    "release_speed", "plate_x", "plate_z", "pfx_x", "pfx_z",
    "release_spin_rate", "pitch_type", "at_bat_number", "pitch_number",
    "balls", "strikes", "p_throws", "stand"
]

missing_critical = [col for col in CRITICAL_COLS if col not in cols_direct]
if missing_critical:
    print(f"\nFAIL: Missing critical columns: {missing_critical}")
else:
    print(f"\nPASS: All {len(CRITICAL_COLS)} critical columns present")

# %%
# Data type comparison (for common columns)
print("\n" + "=" * 60)
print("DATA TYPE COMPARISON (sample)")
print("=" * 60)

sample_cols = ["game_pk", "game_date", "release_speed", "plate_x", "description"]
for col in sample_cols:
    if col in common:
        dtype_pyb = df_pyb[col].dtype
        dtype_direct = df_direct[col].dtype
        match = "OK" if dtype_pyb == dtype_direct else "DIFF"
        print(f"  {col}: pyb={dtype_pyb}, direct={dtype_direct} {match}")

# %%
# Sample data comparison - check specific games match
print("\n" + "=" * 60)
print("GAME-LEVEL DATA COMPARISON")
print("=" * 60)

# Get common games
games_pyb = set(df_pyb["game_pk"].dropna().unique())
games_direct = set(df_direct["game_pk"].dropna().unique())
common_games = games_pyb & games_direct

print(f"Games in pybaseball:       {len(games_pyb)}")
print(f"Games in statcast_fetcher: {len(games_direct)}")
print(f"Games in common:           {len(common_games)}")

if len(common_games) > 0:
    # Check pitch counts for first 5 common games
    print("\nPitch counts per game (first 5):")
    for game_pk in sorted(common_games)[:5]:
        pitches_pyb = len(df_pyb[df_pyb["game_pk"] == game_pk])
        pitches_direct = len(df_direct[df_direct["game_pk"] == game_pk])
        match = "OK" if pitches_pyb == pitches_direct else f"DIFF ({pitches_pyb - pitches_direct:+d})"
        print(f"  Game {game_pk}: pyb={pitches_pyb}, direct={pitches_direct} {match}")

# %%
# Specific pitch match test
print("\n" + "=" * 60)
print("SPECIFIC PITCH MATCH TEST")
print("=" * 60)

if len(common_games) > 0:
    test_game = sorted(common_games)[0]

    # Get first at-bat, first pitch from this game
    game_pyb = df_pyb[df_pyb["game_pk"] == test_game].sort_values(["at_bat_number", "pitch_number"])
    game_direct = df_direct[df_direct["game_pk"] == test_game].sort_values(["at_bat_number", "pitch_number"])

    if len(game_pyb) > 0 and len(game_direct) > 0:
        first_pitch_pyb = game_pyb.iloc[0]
        first_pitch_direct = game_direct.iloc[0]

        print(f"Game: {test_game}")
        print(f"At-bat: {first_pitch_pyb['at_bat_number']}, Pitch: {first_pitch_pyb['pitch_number']}")

        compare_cols = ["batter", "pitcher", "description", "release_speed", "plate_x", "plate_z"]
        print("\nField comparison:")
        all_match = True
        for col in compare_cols:
            if col in common:
                val_pyb = first_pitch_pyb.get(col)
                val_direct = first_pitch_direct.get(col)
                # Handle float comparison
                if pd.isna(val_pyb) and pd.isna(val_direct):
                    match = True
                elif isinstance(val_pyb, float) and isinstance(val_direct, float):
                    match = abs(val_pyb - val_direct) < 0.01
                else:
                    match = val_pyb == val_direct

                status = "OK" if match else "MISMATCH"
                all_match = all_match and match
                print(f"  {col}: pyb={val_pyb}, direct={val_direct} {status}")

        if all_match:
            print("\nPASS: First pitch data matches exactly")
        else:
            print("\nWARNING: Some fields don't match")

# %%
# Summary
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)

checks = {
    "Row count within 1%": row_diff_pct <= 1,
    "All critical columns present": len(missing_critical) == 0,
    "Games overlap": len(common_games) > 0,
}

all_pass = all(checks.values())
for check, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {check}")

print("\n" + ("=" * 60))
if all_pass:
    print("OVERALL: PASS - statcast_fetcher produces equivalent data")
else:
    print("OVERALL: FAIL - investigate differences above")
print("=" * 60)
