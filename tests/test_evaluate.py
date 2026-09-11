"""Pairing a measurement to a detection is where a silent mistake would corrupt
the headline number, so the name normalising and the matching rules are pinned."""

import numpy as np
import pandas as pd
import pytest

from evaluate import frame_key, match


def test_frame_names_normalise_to_the_same_key():
    """The prediction CSV writes 001, a typed ground truth often keeps .png, and
    a spreadsheet turns 001 into the integer 1."""
    assert frame_key("001") == frame_key("001.png") == frame_key(1) == "001"


def test_non_numeric_names_are_left_alone():
    assert frame_key("bolt_a.jpg") == "bolt_a"


def predictions(rows):
    df = pd.DataFrame(rows, columns=["frame", "x", "y", "z"])
    df["instance"] = df.groupby("frame").cumcount()
    return df


def truths(rows, instance=None):
    df = pd.DataFrame(rows, columns=["frame", "x", "y", "z"])
    df["instance"] = instance if instance is not None else np.nan
    return df


def test_a_frame_with_no_detection_is_reported_not_silently_dropped():
    pairs, unmatched, _ = match(predictions([("001", 0, 0, 1.0)]),
                                truths([("001", 0, 0, 1.0), ("002", 0, 0, 1.0)]))
    assert len(pairs) == 1
    assert unmatched == ["002"]


def test_error_is_the_distance_between_the_pair():
    pairs, _, _ = match(predictions([("001", 0.0, 0.0, 1.0)]),
                        truths([("001", 0.3, 0.4, 1.0)]))
    assert pairs.iloc[0]["error"] == pytest.approx(0.5)


def test_nearest_detection_is_chosen_when_no_instance_is_given():
    pairs, _, used_nearest = match(
        predictions([("001", 5.0, 0, 0), ("001", 0.1, 0, 0)]),
        truths([("001", 0.0, 0, 0)]))
    assert used_nearest is True
    assert pairs.iloc[0]["error"] == pytest.approx(0.1)


def test_an_explicit_instance_column_overrides_nearest_matching():
    """Being explicit must win, even when it picks the further detection."""
    pairs, _, used_nearest = match(
        predictions([("001", 5.0, 0, 0), ("001", 0.1, 0, 0)]),
        truths([("001", 0.0, 0, 0)], instance=[0]))
    assert used_nearest is False
    assert pairs.iloc[0]["error"] == pytest.approx(5.0)


def test_two_measurements_in_one_frame_cannot_claim_the_same_detection():
    pairs, _, _ = match(predictions([("001", 0.0, 0, 0), ("001", 3.0, 0, 0)]),
                        truths([("001", 0.05, 0, 0), ("001", 0.10, 0, 0)]))
    assert len(pairs) == 2
    assert sorted(pairs["instance"]) == [0, 1]
