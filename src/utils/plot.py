#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
# plt.style.use(["science", "ieee"])


def plot_records(csv_file, keys, img_name, rolling_window_size=1, average_window_size=1):

    line_styles = ["-", "--", "-.", ":"]
    marker_styles = ["o", "d", "s", "P", "*", "x"]
    line_style_counter = 0
    marker_style_counter = 0

    df = pd.read_csv(csv_file, index_col="epoch")[keys]
    assert any([i == 1 for i in [rolling_window_size, average_window_size]])

    fig, ax = plt.subplots()
    if rolling_window_size > 1:
        df_mean = df.rolling(rolling_window_size, min_periods=1).mean()
        df_std = df.rolling(rolling_window_size, min_periods=1).std()
        for i in range(len(keys)):
            ax.plot(
                df.index.to_numpy(), df_mean[keys[i]], label=keys[i].replace("_", " ").capitalize(),
                linestyle=line_styles[line_style_counter]
            )
            ax.fill_between(
                df.index.to_numpy(), df_mean[keys[i]] - df_std[keys[i]], df_mean[keys[i]] + df_std[keys[i]], alpha=0.4
            )
            line_style_counter += 1
    elif average_window_size > 1:
        df_mean = df.groupby(np.arange(len(df)) // average_window_size).mean()
        df_std = df.groupby(np.arange(len(df)) // average_window_size).std()
        for i in range(len(keys)):
            ax.errorbar(
                df_mean.index.to_numpy(), df_mean[keys[i]], yerr=df_std[keys[i]],
                label=keys[i].replace("_", " ").capitalize(), capsize=3, capthick=1.0, ms=3,
                marker=marker_styles[marker_style_counter], linestyle=line_styles[line_style_counter]
            )
            line_style_counter += 1
            marker_style_counter += 1
    else:
        for i in range(len(keys)):
            ax.plot(
                df.index.to_numpy(), df[keys[i]], label=keys[i].replace("_", " ").capitalize(),
                linestyle=line_styles[line_style_counter]
            )
            line_style_counter += 1

    ax.set_xlabel("#Epoch", fontsize=15)
    ax.set_ylabel(keys[0].split("_")[-1].capitalize(), fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.minorticks_off()
    plt.legend(loc="best", fontsize=15)

    # Save the figure
    fig.tight_layout()
    plt.savefig(os.path.join(os.path.split(csv_file)[0], img_name), bbox_inches="tight", dpi=300)
    # plt.show()
    plt.close("all")


def plot_heatmap(data, img_path, xticks=None, xticklabels=None, yticks=None, yticklabels=None, ticksize=6,
                 xlabel=None, ylabel=None, labelsize=12):

    grid_kws = {"width_ratios": [1, 0.08], "wspace": 0.08}
    data = pd.DataFrame(data=data)
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws)
    ax = sns.heatmap(data, ax=ax, cbar_ax=cbar_ax, cmap="YlGnBu", annot=False, fmt="d", cbar=True)

    # make frame visible
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    for _, spine in cbar_ax.spines.items():
        spine.set_visible(True)

    # set ticks
    ax.tick_params(
        axis="x", which="both", direction="out", top=False, right=False,
        bottom=False, left=False, labelrotation=0, labelsize=ticksize
    )
    ax.tick_params(
        axis="y", which="both", direction="out", top=False, right=False,
        bottom=False, left=False, labelrotation=0, labelsize=ticksize
    )
    if xticks is not None and xticklabels is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    if yticks is not None and yticklabels is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    cbar_ax.tick_params(axis="both", which="both", direction="out", labelsize=ticksize)
    cbar_ax.minorticks_off()

    # set labels
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=labelsize)

    fig.tight_layout()
    plt.savefig(img_path, bbox_inches="tight", dpi=300)
    plt.close("all")


def plot_hist(data, img_path, bins, density=False, ticksize=8, xlabel=None, ylabel=None, labelsize=12):

    fig, ax = plt.subplots()
    ax.hist(data, bins=bins, density=density)

    # set ticks
    ax.tick_params(
        axis="x", which="both", direction="out", top=False, right=False,
        bottom=True, left=False, labelrotation=0, labelsize=ticksize
    )
    ax.tick_params(
        axis="y", which="both", direction="out", top=False, right=False,
        bottom=False, left=True, labelrotation=0, labelsize=ticksize
    )
    ax.minorticks_off()

    # set labels
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=labelsize)

    fig.tight_layout()
    plt.savefig(img_path, bbox_inches="tight", dpi=300)
    plt.close("all")


def plot_images(arrays, img_path):

    n, h, w, c = arrays.shape
    num_rows = 1 if n % 5 != 0 else int(n / 5)
    num_cols = n if num_rows == 1 else 5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8))
    if n == 1:
        if c == 1:
            axes.imshow(arrays.squeeze(), cmap="gray")
        else:
            axes.imshow(arrays.squeeze())
        axes.axis("off")
        fig.tight_layout()
        plt.savefig(img_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close("all")
    else:
        counter = 0
        for i in range(num_rows):
            for j in range(num_cols):
                array = arrays[counter]
                if c == 1:
                    axes[i, j].imshow(array.squeeze(), cmap="gray")
                else:
                    axes[i, j].imshow(array)
                axes[i, j].axis("off")
                counter += 1
        fig.tight_layout()
        plt.savefig(img_path, bbox_inches="tight", dpi=300)
        plt.close("all")


def plot_image_grid(arrays_list, img_path):

    n_rows = len(arrays_list)
    n_cols, h, w, c = arrays_list[0].shape
    fig, axes = plt.subplots(n_rows, n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            if c == 1:
                axes[i, j].imshow(arrays_list[i][j].squeeze(), cmap="gray")
            else:
                axes[i, j].imshow(arrays_list[i][j])
            axes[i, j].axis("off")
    fig.tight_layout()
    plt.savefig(img_path, bbox_inches="tight", dpi=300)
    plt.close("all")


