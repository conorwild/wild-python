import pandas as pd
import numpy as np
from cbspython import TESTS, check_arg_value, test_names, iterable
from collections import OrderedDict
import re


def column_idx_type(df):
    n_levels = len(df.columns.names)
    assert(n_levels < 3)
    assert(n_levels > 0)
    return n_levels


_COMPARATORS = {
    '<': lambda x, y: x < y,
    '>': lambda x, y: x > y,
    '==': lambda x, y: x == y,
    '<=': lambda x, y: x <= y,
    '>=': lambda x, y: x >= y,
    '!=': lambda x, y: x != y
}

def comparison_fn(comp_str):
    check_arg_value('comparator', comp_str, _COMPARATORS.keys())
    return _COMPARATORS[comp_str]


_ABBREV_MAP = {i.abbrev: n for n, i in TESTS.items()}
_ABBREV_MATCHER = re.compile(
    f"\A(?P<abbrev>({')|('.join(_ABBREV_MAP.keys())}))_"
)

def testname_from_abbrev(abbrev):
    check_arg_value('abbrev', abbrev, _ABBREV_MAP.keys())
    return _ABBREV_MAP[abbrev]

def is_a_test_column(column_name):
    return True if _ABBREV_MATCHER.match(column_name) is not None else False

def test_abbrev_from_column(column_name, check=True):
    assert isinstance(column_name, str)
    m = _ABBREV_MATCHER.match(column_name)
    if m is None:
        if check:
            raise TypeError(f"{column_name} does not appear to be a test column")
        else:
            return None
    return m.group('abbrev')


def filter_by_grp_count(df, group_var, comparison, threshold):
    return (
        df.groupby(group_var)
        .filter(
            lambda u: comparison_fn(comparison)(u.shape[0], threshold)
        )
    )


def _force_full_testname(testname):
    if testname in _ABBREV_MAP.keys():
        testname = _ABBREV_MAP[testname]
    return testname


def all_columns_for_test(df, testname):
    testname = _force_full_testname(testname)
    check_arg_value('testname', testname, tests_in_df(df))

    test = TESTS[testname]
    if column_idx_type(df) == 1:
        return [c for c in df.columns if c.startswith(test.abbrev+"_")]
    elif column_idx_type(df) == 2:
        return [c for c in df.columns if c[0] == test.name]
    else:
        raise TypeError


def all_cbs_columns(df):
    """ Finds all the columns in the dataframe that contain CBS-related score
    data. Doesn't matter if the column index is multi or single (abbreviated).

    Args:
        df (Pandas DataFrame): The input dataframe.
    """
    return [
        c for t in tests_in_df(df) for c in all_columns_for_test(df, t)
    ]


def tests_in_df(df):
    if column_idx_type(df) == 1:
        t_ = [test_abbrev_from_column(c, check=False) for c in df.columns]
        tests = [_ABBREV_MAP[t] for t in t_ if t is not None]
    elif column_idx_type(df) == 2:
        tests = [c[0] for c in df.columns if c[0] in test_names()]
    else:
        raise TypeError("Invalid Index Type!")
    
    return list(OrderedDict.fromkeys(tests))


def columns_for_test_feature(df, tests, feature, check=False):
    if tests in ['*', 'all', 'ALL']:
        tests = tests_in_df(df)

    if not iterable(tests):
        tests = [tests]

    columns = []
    for testname in tests:
        testname = _force_full_testname(testname)
        check_arg_value('test', testname, tests_in_df(df))

        test = TESTS[testname]
        if column_idx_type(df) == 1:
            f = f"{test.abbrev}_{feature}"
        else:
            f = (test.name, feature)
        if check:
            check_arg_value('feature', f, all_columns_for_test(df, testname))
        columns.append(f)
    
    if len(columns) == 1:
        columns = columns[0]
    return columns


def filter_tests_by_feature(df, tests, feature, expr, threshold):

    if tests in ['*', 'all', 'ALL']:
        tests = tests_in_df(df)

    if not iterable(tests):
        tests = [tests]

    for testname in tests:
        test_cols = all_columns_for_test(df, testname)
        filter_on = columns_for_test_feature(df, testname, feature, check=True)
        df.loc[
            _COMPARATORS[expr](df[filter_on], threshold),
            test_cols
        ] = np.nan

    return df


def filter_by_worse_than_chance(df, remove_chance=True):
    expr = '<=' if remove_chance else '<'
    for test in tests_in_df(df):
        chance_level = TESTS[test].chance_level
        df = filter_tests_by_feature(df, test, 'accuracy', expr, chance_level)
    return df


def nan_by_test(df):
    """ Given a dataframe with multiple columns (features) per test, we will
        nan all columns beling to the same test if any of those columns are 
        nan.

    Args:
        df (Pandas DataFrame): The input/output datafraem
    """
    for name in tests_in_df(df):
        test_cols = all_columns_for_test(df, name)
        df.loc[df[test_cols].isna().any(axis=1), test_cols] = np.nan
    return df


def filter_by_sds(df, sds = [6,4], subset=None, drop=False):
    if subset is None:
        subset = df.columns
    
    df_ = df[subset].copy()

    outliers = np.full(df_.shape, False)
    for sd in sds:
        stats = df_.agg(['count', 'mean', 'std'])
        oorange = (abs(df_ - stats.loc['mean', :]) > sd*stats.loc['std', :])
        df_[oorange] = np.nan
        outliers = (outliers | oorange.values)
    df[subset] = df_

    if drop:
        df = df[~outliers.any(axis=1)]
    return df
