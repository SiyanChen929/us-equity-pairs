import pandas as pd

from equity_pairs.tradable_lineage import pair_us_listing_start


def test_pair_listing_start_is_pair_specific():
    assert pair_us_listing_start("AME__FERG") == pd.Timestamp("2021-03-11")
    assert pair_us_listing_start("AVY__SW") == pd.Timestamp("2024-07-08")
    assert pair_us_listing_start("BKNG__MGM") is None
