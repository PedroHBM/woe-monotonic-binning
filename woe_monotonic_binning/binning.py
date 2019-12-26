import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing as mp
from joblib import Parallel, delayed


def unpack_woe(args):
    return woe_binning(*args)


def batch_woe_binning(target, dataset, n_threshold=None, n_occurences=1, p_threshold=0.1):
    from math import ceil

    nprocs = mp.cpu_count()
    columns = dataset.columns[dataset.columns != target]
    if n_threshold is None:
        min_bin_size = ceil(dataset.shape[0] / 20)
    else:
        min_bin_size = n_threshold
    df_list = Parallel(n_jobs=nprocs, verbose=5)(delayed(woe_binning)
                                                 (target, dataset[[column, target]],
                                                  n_threshold=min_bin_size, p_threshold=0.1)
                                                 for column in columns)
    return {i.variable[0]: i for i in df_list}


def woe_binning(target, dataset, n_threshold, n_occurences=1, p_threshold=0.1, sort_overload=None):

    column = dataset.columns[dataset.columns != target][0]
    sorted_dataset = dataset.sort_values(by=[column])
    size = sorted_dataset.shape[0]

    if sorted_dataset[:int(size / 4)][target].sum() > sorted_dataset[int(size * 3 / 4):][target].sum():
        order = True
        interval_end = np.inf
    else:
        order = False
        interval_end = -np.inf

    summary = dataset.dropna().groupby([column]).agg(["mean", "size", "std"])

    summary.columns = summary.columns.droplevel(level=0)

    summary = summary[["mean", "size", "std"]]
    summary = summary.reset_index()

    summary["del_flag"] = 0
    summary["std"] = summary["std"].fillna(0)

    summary = summary.sort_values(by=[column], ascending=(sort_overload or order)).reset_index(drop=True)

    while True:
        i = 0

        summary = summary[summary.del_flag == 0]
        summary = summary.reset_index(drop=True)

        while True:

            j = i + 1

            if j >= len(summary):
                break

            if summary.iloc[j]['mean'] < summary.iloc[i]['mean']:
                i = i + 1
                continue
            else:
                while True:
                    n = summary.iloc[j]['size'] + summary.iloc[i]['size']
                    m = (summary.iloc[j]['size'] * summary.iloc[j]['mean'] +
                         summary.iloc[i]['size'] * summary.iloc[i]['mean']) / n

                    if n == 2:
                        s = np.std([summary.iloc[j]['mean'], summary.iloc[i]['mean']])
                    else:
                        s = np.sqrt((summary.iloc[j]['size'] * ((summary.iloc[j]['std']) ** 2) +
                                     summary.iloc[i]['size'] * ((summary.iloc[i]['std']) ** 2)) / n)

                    summary.loc[i, "size"] = n
                    summary.loc[i, "mean"] = m
                    summary.loc[i, "std"] = s
                    summary.loc[j, "del_flag"] = 1

                    j = j + 1

                    if j >= len(summary):
                        break
                    if summary.loc[j, "mean"] < summary.loc[i, "mean"]:
                        i = j
                        break
            if j >= len(summary):
                break

        dels = np.sum(summary["del_flag"])
        if dels == 0:
            break

    while True:
        summary["next_mean"] = summary["mean"].shift(-1)
        summary["next_size"] = summary["size"].shift(-1)
        summary["next_std"] = summary["std"].shift(-1)

        summary["updated_size"] = summary["next_size"] + summary["size"]
        summary["updated_mean"] = (summary["next_mean"] * summary["next_size"] +
                                   summary["mean"] * summary["size"]) / summary["updated_size"]

        summary["updated_std"] = (summary["next_size"] * summary["next_std"] ** 2 +
                                  summary["size"] * summary["std"] ** 2) / (summary["updated_size"] - 2)

        summary["z_value"] = (summary["mean"] - summary["next_mean"]) / np.sqrt(
            summary["updated_std"] * (1 / summary["size"] + 1 / summary["next_size"]))

        summary["p_value"] = 1 - stats.norm.cdf(summary["z_value"])

        condition = (summary["size"] < n_threshold) | (summary["next_size"] < n_threshold) | (
                summary["mean"] * summary["size"] < n_occurences) | (
                            summary["next_mean"] * summary["next_size"] < n_occurences)

        summary[condition].p_value = summary[condition].p_value + 1

        summary["p_value"] = summary.apply(
            lambda row: row["p_value"] + 1 if (row["size"] < n_threshold) | (row["next_size"] < n_threshold) |
                                              (row["mean"] * row["size"] < n_occurences) |
                                              (row["next_mean"] * row["next_size"] < n_occurences)
            else row["p_value"], axis=1)

        max_p = max(summary["p_value"])
        row_of_maxp = summary['p_value'].idxmax()
        row_delete = row_of_maxp + 1

        if max_p > p_threshold:
            summary = summary.drop(summary.index[row_delete])
            summary = summary.reset_index(drop=True)
        else:
            break

        summary["mean"] = summary.apply(lambda row: row["updated_mean"] if row["p_value"] == max_p else row["mean"],
                                        axis=1)
        summary["size"] = summary.apply(lambda row: row["updated_size"] if row["p_value"] == max_p else row["size"],
                                        axis=1)
        summary["std"] = summary.apply(
            lambda row: np.sqrt(row["updated_std"]) if row["p_value"] == max_p else row["std"], axis=1)

    woe_summary = summary[[column, "size", "mean"]]
    woe_summary.columns = ["interval_start_include", "size", "mean"]
    woe_summary["interval_end_exclude"] = woe_summary.interval_start_include.shift(-1).fillna(interval_end)
    woe_summary.interval_start_include.loc[0] = interval_end * -1
    woe_summary["variable"] = column
    woe_summary = woe_summary[["variable", "interval_start_include", "interval_end_exclude", "size", "mean"]]

    if dataset[column].isna().sum() > 0:
        nan_line = list(
            dataset[dataset[column].isna()].fillna(0).groupby([column]).agg(["size", "mean"]).reset_index(
                drop=True).loc[0].fillna(0).values)
        nan_line = [column, np.nan, np.nan] + nan_line
        woe_summary.loc[woe_summary.index.max() + 1] = nan_line

    woe_summary["bads"] = woe_summary["mean"] * woe_summary["size"]
    woe_summary["goods"] = woe_summary["size"] - woe_summary["bads"]

    total_goods = np.sum(woe_summary["goods"])
    total_bads = np.sum(woe_summary["bads"])

    woe_summary["dist_good"] = woe_summary["goods"] / total_goods
    woe_summary["dist_bad"] = woe_summary["bads"] / total_bads

    woe_summary["woe"] = np.log(woe_summary["dist_bad"] / woe_summary["dist_good"])

    woe_summary["iv_components"] = (woe_summary["dist_bad"] - woe_summary["dist_good"]) * woe_summary["woe"]

    return woe_summary


def apply_bins(dataset, dict_woe, iv_threshold=0.02, bin_threshold=2, is_df=False, remove_100_corr=True):
    df_bin = pd.DataFrame()
    ivs_list = []
    if is_df:
        values = dict_woe.variable.values
    else:
        values = dict_woe.values()
    for df_col in values:
        if is_df:
            df_col = dict_woe[dict_woe.variable == df_col]
        iv_total = df_col.dropna().iv_components.sum()
        if iv_total < iv_threshold or df_col.shape[0] < bin_threshold or iv_total == np.inf:
            continue
        column = df_col.variable.loc[0]
        df_col_dropped = df_col.dropna()
        bin_cuts = list(df_col_dropped.interval_start_include.values) + [
            df_col_dropped.interval_end_exclude.values[-1]]
        labels_woe = list(df_col_dropped.woe.values)
        if bin_cuts[0] > bin_cuts[-1]:
            bin_cuts.reverse()
            labels_woe.reverse()
            include_left = False
            include_right = True
        else:
            include_left = True
            include_right = False
        if remove_100_corr:
            if iv_total not in ivs_list:
                df_bin[column + '_bin'] = pd.to_numeric(
                    pd.cut(dataset[column].fillna(dataset[column].median()), bin_cuts, include_lowest=include_left,
                           right=include_right, labels=labels_woe))
                ivs_list.append(iv_total)
        else:
            df_bin[column + '_bin'] = pd.to_numeric(
                pd.cut(dataset[column].fillna(dataset[column].median()), bin_cuts, include_lowest=include_left,
                       right=include_right, labels=labels_woe))
    return df_bin
