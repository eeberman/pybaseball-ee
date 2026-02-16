# %%
"""
statcast_fetcher.py - Direct Statcast data fetching from Baseball Savant

Replaces pybaseball dependency with direct requests to Baseball Savant's public CSV endpoint.
Handles date chunking (required due to ~6 day/30k row limit), caching, and retry logic.
"""

from __future__ import annotations

import io
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_session_with_retries(
    retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: tuple[int, ...] = (500, 502, 503, 504),
) -> requests.Session:
    """
    Create a requests session with automatic retry on transient failures.

    Args:
        retries: Number of retry attempts
        backoff_factor: Exponential backoff multiplier (e.g., 1.0 means 1s, 2s, 4s waits)
        status_forcelist: HTTP status codes to retry on

    Returns:
        Configured requests session
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_statcast(
    start_date: str,
    end_date: str,
    chunk_days: int = 5,
    cache_dir: Path | str | None = None,
    verbose: bool = True,
    timeout: int = 120,
    game_types: str = "",
) -> pd.DataFrame:
    """
    Fetch Statcast data directly from Baseball Savant.

    Baseball Savant limits requests to ~6 days or ~30,000 rows. This function
    automatically chunks the date range and combines results.

    Args:
        start_date: Start date in YYYY-MM-DD format (inclusive)
        end_date: End date in YYYY-MM-DD format (inclusive)
        chunk_days: Number of days per request (default 5, max ~6)
        cache_dir: Optional directory to cache chunks as parquet files
        verbose: Print progress messages
        timeout: Request timeout in seconds
        game_types: Baseball Savant hfGT filter string. Examples:
            "" = all game types (regular season + playoffs)
            "R|" = regular season only
            "F|D|L|W|" = playoffs only (Wild Card, Division, LCS, World Series)

    Returns:
        DataFrame with all Statcast data for the date range

    Raises:
        requests.HTTPError: If Baseball Savant returns an error
        ValueError: If date parsing fails

    Example:
        >>> df = fetch_statcast("2023-04-01", "2023-04-30", cache_dir="data/cache")
    """
    all_chunks = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Ensure cache_dir is a Path if provided
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Create session with retry logic
    session = get_session_with_retries()

    chunk_num = 0
    total_chunks = ((end - current).days // chunk_days) + 1

    while current <= end:
        chunk_num += 1
        # chunk_end is inclusive, so we add chunk_days-1 for full days
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)

        chunk_start_str = current.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")

        # Check cache first
        if cache_dir is not None:
            cache_file = cache_dir / f"statcast_{current:%Y%m%d}_{chunk_end:%Y%m%d}.parquet"
            if cache_file.exists():
                if verbose:
                    print(f"  [{chunk_num}/{total_chunks}] Cached: {chunk_start_str} to {chunk_end_str}")
                all_chunks.append(pd.read_parquet(cache_file))
                current = chunk_end + timedelta(days=1)
                continue

        # Fetch from Baseball Savant
        if verbose:
            print(f"  [{chunk_num}/{total_chunks}] Fetching: {chunk_start_str} to {chunk_end_str}")

        # Baseball Savant endpoint parameters
        # all=true gets all pitches (vs just specific events)
        # type=details gets full pitch-level data
        params = {
            "all": "true",
            "hfPT": "",
            "hfAB": "",
            "hfGT": game_types,
            "hfPR": "",
            "hfZ": "",
            "hfStadium": "",
            "hfBBL": "",
            "hfNewZones": "",
            "hfPull": "",
            "hfC": "",
            "hfSea": "",
            "hfSit": "",
            "player_type": "pitcher",
            "hfOuts": "",
            "hfOpponent": "",
            "pitcher_throws": "",
            "batter_stands": "",
            "hfSA": "",
            "game_date_gt": chunk_start_str,
            "game_date_lt": chunk_end_str,
            "hfMo": "",
            "hfTeam": "",
            "home_road": "",
            "hfRO": "",
            "position": "",
            "hfInfield": "",
            "hfOutfield": "",
            "hfInn": "",
            "hfBBT": "",
            "hfFlag": "",
            "metric_1": "",
            "group_by": "name",
            "min_pitches": "0",
            "min_results": "0",
            "min_pas": "0",
            "sort_col": "pitches",
            "player_event_sort": "api_p_release_speed",
            "sort_order": "desc",
            "chk_stats_sweetspot_speed": "on",
            "type": "details",
        }

        response = session.get(
            "https://baseballsavant.mlb.com/statcast_search/csv",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()

        # Parse CSV response
        content = response.text

        # Check for empty response
        if not content.strip() or content.strip() == "":
            if verbose:
                print(f"    Empty response for {chunk_start_str} to {chunk_end_str}")
            current = chunk_end + timedelta(days=1)
            continue

        try:
            df_chunk = pd.read_csv(io.StringIO(content), low_memory=False)
        except pd.errors.EmptyDataError:
            if verbose:
                print(f"    No data for {chunk_start_str} to {chunk_end_str}")
            current = chunk_end + timedelta(days=1)
            continue

        if len(df_chunk) == 0:
            if verbose:
                print(f"    No rows for {chunk_start_str} to {chunk_end_str}")
            current = chunk_end + timedelta(days=1)
            continue

        if verbose:
            print(f"    Retrieved {len(df_chunk):,} rows")

        all_chunks.append(df_chunk)

        # Cache the chunk
        if cache_dir is not None:
            cache_file = cache_dir / f"statcast_{current:%Y%m%d}_{chunk_end:%Y%m%d}.parquet"
            df_chunk.to_parquet(cache_file, index=False)

        # Rate limiting - be a good citizen
        time.sleep(1.5)

        current = chunk_end + timedelta(days=1)

    # Combine all chunks
    if not all_chunks:
        if verbose:
            print("No data retrieved for the specified date range")
        return pd.DataFrame()

    df = pd.concat(all_chunks, ignore_index=True)

    # Remove duplicates that might occur at chunk boundaries
    if "game_pk" in df.columns and "at_bat_number" in df.columns and "pitch_number" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
        after = len(df)
        if verbose and before != after:
            print(f"  Removed {before - after:,} duplicate rows")

    if verbose:
        print(f"Total: {len(df):,} rows from {start_date} to {end_date}")

    return df


def clear_cache(cache_dir: Path | str, older_than_days: int | None = None) -> int:
    """
    Clear cached statcast parquet files.

    Args:
        cache_dir: Directory containing cached files
        older_than_days: If specified, only clear files older than this many days

    Returns:
        Number of files deleted
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0

    deleted = 0
    cutoff = datetime.now() - timedelta(days=older_than_days) if older_than_days else None

    for f in cache_dir.glob("statcast_*.parquet"):
        if cutoff is None or datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
            f.unlink()
            deleted += 1

    return deleted


# %%
# Test/demo if run directly
if __name__ == "__main__":
    # Quick test: fetch a single day
    print("Testing fetch_statcast...")
    df_test = fetch_statcast(
        "2024-04-01",
        "2024-04-01",
        cache_dir=Path("data/cache"),
        verbose=True,
    )
    print(f"\nTest result: {df_test.shape}")
    if len(df_test) > 0:
        print(f"Columns: {df_test.columns[:10].tolist()}...")
        print(f"Sample game_date values: {df_test['game_date'].unique()[:3]}")
