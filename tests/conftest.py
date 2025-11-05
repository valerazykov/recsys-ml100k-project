import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to PYTHONPATH so `import src.*` works when running pytest from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def small_ratings_df():
    """
    Simple ratings DataFrame in (user_id, item_id, rating, timestamp) format,
    similar to MovieLens u.data.
    """
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3, 3, 3, 4],
            "item_id": [10, 11, 10, 12, 13, 14, 10],
            "rating": [5, 4, 3, 2, 5, 4, 3],
            "timestamp": np.arange(7, dtype=int),
        }
    )
    return df


@pytest.fixture
def tmp_ratings_file(tmp_path, small_ratings_df):
    """Write small_ratings_df to a temporary TSV file and return its path."""
    p = tmp_path / "ratings.tsv"
    # write tab-separated file like u.data (no header)
    small_ratings_df.to_csv(p, sep="\t", header=False, index=False)
    return str(p)
