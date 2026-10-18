"""Source-comparison unit checks use only small synthetic traffic-shaped files."""
import pandas as pd
from scripts.verify_unsw_source import (
    attack_counts, compare_columns, row_multiset
)


def _frame(order=None, change=False, id_offset=0):
    cols = {
        "id": [1 + id_offset, 2 + id_offset, 3 + id_offset],
        "dur": [1.0, 2.0, 2.0],
        "proto": ["tcp", "udp", "udp"],
        "attack_cat": ["Normal", "DoS", "DoS"],
        "label": [0, 1, 1]
    }
    f = pd.DataFrame(cols)
    if change:
        f.loc[2, "label"] = 0
    return f if order is None else f.iloc[order].reset_index(drop=True)


def _compare(a,b,drop_id=False):
    columns = [x for x in a.columns if x != "id"] if drop_id else list(a.columns)
    numeric = {"id", "dur", "label"}
    c1,s1 = row_multiset(a, columns, numeric, "fixture A")
    c2,s2 = row_multiset(b, columns, numeric, "fixture B")
    return c1==c2, s1==s2


def test_identical_data():
    assert _compare(_frame(), _frame()) == (True, True)


def test_reordered_data_preserves_multiset_but_not_order():
    assert _compare(_frame(), _frame(order=[2,0,1])) == (True, False)


def test_label_change_is_not_equivalent():
    assert _compare(_frame(), _frame(change=True)) == (False, False)


def test_changed_id_only_matches_excluding_id():
    assert _compare(_frame(), _frame(id_offset=100)) == (False, False)
    assert _compare(_frame(), _frame(id_offset=100), drop_id=True) == (True, True)


def test_numeric_serialization_treated_as_same_parsed_value():
    a, b = _frame(), _frame()
    b["dur"] = b["dur"].astype("int64")
    assert _compare(a, b) == (True, True)


def test_label_distribution_and_schema():
    assert attack_counts(_frame()) == {"normal":1,"dos":2}
    assert compare_columns(_frame(),_frame())["same_column_order"]
