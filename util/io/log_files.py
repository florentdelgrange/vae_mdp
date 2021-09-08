import os
import sys
from collections import namedtuple
from typing import Collection, List, Optional, Dict, Tuple, Union
import glob
import numpy as np
from scipy.stats import iqr
from tensorflow.python.framework import tensor_util
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def get_event_dataframe(
        log_dir: str,
        tags: Optional[Collection[str]] = None,
        exclude_pattern: Optional[str] = None,
        tags_renaming: Optional[Dict[str, str]] = None,
        run_name: Optional[str] = None,
        event_name: Optional[str] = None,
        value_dtype: Optional = np.float32,
        smooth: Optional[Collection[str]] = None,
        smooth_window: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    if tags_renaming is None:
        tags_renaming = {}
    if smooth is None:
        smooth = []
    if smooth_window is None:
        smooth_window = {}

    TaggedEvent = namedtuple('TaggedEvent', ['x_axis', 'y_axis', 'tags'])
    df = None

    files = glob.glob(log_dir, recursive=True)
    if exclude_pattern is not None:
        files = set(files) - set(glob.glob(exclude_pattern, recursive=True))
    if len(files) == 0:
        raise IOError("No such file or directory:",
                      "{}".format(log_dir) + (" \\ {}".format(exclude_pattern)
                                              if exclude_pattern is not None else ""))

    for event_file in files:
        tagged_events = TaggedEvent(x_axis=[], y_axis=[], tags=[])

        for event in summary_iterator(event_file):
            for value in event.summary.value:
                if tags is not None:
                    if value.tag in tags:
                        tagged_events.x_axis.append(event.step)
                        tagged_events.y_axis.append(tensor_util.MakeNdarray(value.tensor))
                        tagged_events.tags.append(tags_renaming.get(value.tag, value.tag))
                else:
                    tagged_events.x_axis.append(event.step)
                    tagged_events.y_axis.append(tensor_util.MakeNdarray(value.tensor))
                    tagged_events.tags.append(tags_renaming.get(value.tag, value.tag))

        data = pd.DataFrame(
            {'step': np.array(tagged_events.x_axis, dtype=np.int64),
             'value': (tagged_events.y_axis if value_dtype is None else
                       np.array(tagged_events.y_axis, dtype=value_dtype)),
             'tag': tagged_events.tags, })

        if not data.empty:
            for tag in set(smooth) & (set(tags) if tags is not None else set(smooth)):
                moving_mean = (
                    data[data['tag'] == tag]
                        .rolling(window=smooth_window.get(tag, 10), min_periods=1, on='step')
                        .mean()
                        .rename(columns={'value': 'smooth'}))

                data = data.join(moving_mean.set_index('step'), on='step')

            df = data if df is None else df.append(data)

    if run_name is not None:
        df['run'] = run_name
    if event_name is not None:
        df['event'] = event_name

    return df


def get_interquantile_range(df: pd.DataFrame):
    return iqr(np.array([df[column].values for column in df.columns]), axis=0)


def plot_elbo_evaluation(
        df: pd.DataFrame,
        compare_environments: bool = False,
        compare_experience_replay: bool = False,
        relplot: bool = False,
        eval_elbo_tag: str = 'eval_elbo',
        aspect: float = 2.5,
        estimator: str = 'mean',
        ci: float = 0.9,
):
    if estimator == 'median':
        estimator = np.median
    df = df[df['tag'] == eval_elbo_tag]

    sns.set_theme(style="darkgrid")

    if compare_experience_replay or compare_environments and relplot:
        return sns.relplot(
            data=df.rename(columns={
                "value": "ELBO",
                "run": "experience replay",
                "event": "environment"}),
            x='step',
            y='ELBO',
            row='experience replay' if compare_experience_replay else None,
            col='environment' if compare_environments else None,
            aspect=aspect,
            kind='line',
            facet_kws=dict(sharex=False, sharey=False),
            estimator=estimator,
            ci=ci * 100 if estimator != 'mean' else 'sd')
    else:
        if compare_environments:
            hue = 'environment'
        elif compare_experience_replay:
            hue = 'experience replay'
        else:
            hue = None
        return sns.lineplot(
            data=df.rename(columns={
                "value": "ELBO",
                "run": "experience replay",
                "event": "environment"}),
            x='step',
            y='ELBO',
            legend='brief',
            hue=hue,
            estimator=estimator,
            ci=ci * 100 if estimator != 'mean' else 'sd')


def plot_histograms_per_step(
        df: pd.DataFrame,
        num_x_ticks: int = 5,
        num_y_ticks: int = 4,
        use_math_text: bool = False,
        aspect: float = 2.5,
        col: str = 'run',
        display_ylabel: bool = True
):
    df = df.sort_values(by='step')
    tick = ticker.ScalarFormatter(useOffset=True, useMathText=use_math_text)
    tick.set_powerlimits((0, 0))

    # we assume that all buckets have the same range, according to the way tf summaries records histograms
    if col == 'run':
        nrows = len(df['event'].unique()) if 'event' in df else 1
        ncols = len(df['run'].unique()) if 'run' in df else 1
    elif col == 'event':
        ncols = len(df['event'].unique()) if 'event' in df else 1
        nrows = len(df['run'].unique()) if 'run' in df else 1
    else:
        raise ValueError("The col parameter should be either 'event' or 'run'")

    if col == 'run':
        fig, axs = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, aspect * nrows))
    else:
        fig, axs = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, aspect * nrows), sharey=(col == 'event'))
        fig.tight_layout()

    # fig, axs = plt.subplots(nrows, ncols, figsize=(2.75 * ncols, .75 * aspect * nrows))
    def plot_histogram(df, ax, display_x_ticks=True, display_y_ticks=True, ax_title=None):
        buckets = df['value'].head(1).to_numpy()[0][..., :2]

        data = np.stack(df['value'].to_numpy())[..., 2]

        def round_to_base(x, base, upper: bool = False):
            if upper:
                return base * np.ceil(x / base)
            else:
                return base * np.round(x / base)

        xticks = np.linspace(0, df['step'].unique().max(), num=data.shape[0], dtype=np.int32)
        xticks = round_to_base(xticks, base=df['step'].unique().max() // (num_x_ticks * 2))
        if use_math_text:
            xticks = [u"${}$".format(tick.format_data(x) if x != 0. else str(0))
                      for x in np.flipud(xticks).astype(float)]
        else:
            xticks = [tick.format_data(x) if x != 0. else str(0) for x in np.flipud(xticks).astype(float)]
        power2 = np.power(2, np.ceil(np.log(buckets.flatten().max()) / np.log(2)))
        yticks = np.array([round_to_base(power2 / len(buckets) * (bucket + 1), power2 // num_y_ticks)
                           for bucket in range(len(buckets))],
                          dtype=np.int32)

        _df = pd.DataFrame(
            data=np.flipud(data.transpose()),
            columns=np.flipud(np.array(xticks)),
            index=np.flipud(yticks))

        yticks_interval = int(round_to_base(len(buckets), num_y_ticks, upper=True)) // num_y_ticks
        ax = sns.heatmap(
            _df,
            cmap='Blues',
            xticklabels=data.shape[0] // num_x_ticks if display_x_ticks else display_x_ticks,
            yticklabels=yticks_interval if display_y_ticks else display_y_ticks,
            cbar=False,
            ax=ax)

        if display_x_ticks:
            ax.set_xlabel("step")
        if display_y_ticks and col == 'run':
            ax.set_ylabel("latent state")
        if ax_title is not None:
            ax.set_title(ax_title)

        for label in ax.get_yticklabels():
            if display_y_ticks:
                label.set_rotation(0)

        return ax

    if nrows == 1 and ncols == 1:
        plot_histogram(df, axs)
    else:
        if nrows == 1:
            _axs = np.expand_dims(axs, 0)
        elif ncols == 1:
            _axs = np.expand_dims(axs, 1)
        else:
            _axs = axs
        for row, axs_row in enumerate(_axs):
            for column, ax in enumerate(axs_row):
                _row = 'event' if col == 'run' else 'run'
                _col = 'run' if col == 'run' else 'event'
                _df = df[df[_row] == df[_row].unique()[row]] if _row in df else df
                _df = _df[_df[_col] == _df[_col].unique()[column]] if _col in _df else _df
                plot_histogram(
                    _df,
                    ax,
                    display_x_ticks=(row == nrows - 1),
                    display_y_ticks=(column == 0),
                    ax_title=('experience replay = {}'.format(df['run'].unique()[column if col == 'run' else row])
                              if ('run' in df and row == 0 and _row == 'event') or
                                 ('run' in df) else None))

    # plt.tight_layout()
    # fig.supylabel("latent state")
    if display_ylabel and col == 'event':
        fig.text(0.05 if col == 'run' else -0.02, 0.5, 'latent state', va='center', rotation='vertical')
    plt.subplots_adjust(hspace=0.05 if col == 'run' else 0.15, wspace=0.05)
    return fig


def plot_policy_evaluation(
        df: pd.DataFrame,
        plot_best: bool = False,
        original_policy_expected_rewards: Optional[Union[float, Dict[str, float]]] = None,
        compare_experience_replay: bool = False,
        compare_environments: bool = False,
        relplot: bool = False,
        original_policy_as_label: bool = True,
        policy_evaluation_avg_rewards_tag: str = 'policy_evaluation_avg_rewards',
        aspect: float = 2.5,
        estimator: str = 'mean',
        ci=0.9,
        font_scale=1.35,
        environment_hue: bool = True,
        hide_title: bool = False,
):
    if estimator == 'median':
        estimator = np.median

    df = df[df['tag'] == policy_evaluation_avg_rewards_tag]

    sns.set_theme(style="darkgrid")

    df = df.assign(policy='distilled')

    if plot_best:
        _df = None
        for env in df['event'].unique():
            __df = df[df['event'] == env]
            __df = __df.append(__df.assign(policy='distilled (best)', value=__df['value'].max()))
            _df = __df if _df is None else _df.append(__df)
        df = _df

    if original_policy_expected_rewards is not None and not original_policy_as_label and not compare_environments:
        N = 100
        g = plt.plot(
            np.linspace(0, df['step'].max(), N, dtype=np.int64),
            np.ones(N) * original_policy_expected_rewards,
            linestyle='--',
            color='tab:green',
            alpha=0.5,
            linewidth=1)

        plt.text(0, 200, '$\pi$  ', ha='right', color='tab:green', )
        return g

    elif original_policy_expected_rewards is not None:
        if isinstance(original_policy_expected_rewards, dict):
            _df = None
            for env in df['event'].unique():
                __df = df[df['event'] == env]
                if env in original_policy_expected_rewards:
                    __df = __df.append(__df.assign(policy='original', value=original_policy_expected_rewards[env]))
                _df = __df if _df is None else _df.append(__df)
            df = _df
        else:
            df = df.append(df.assign(policy='original', value=original_policy_expected_rewards))

    if compare_experience_replay or compare_environments and relplot:
        sns.set_context('paper')
        sns.set(font_scale=font_scale)
        g = sns.relplot(
            data=df.rename(columns={
                "value": "avg. rewards",
                "run": "experience replay",
                "event": "environment"}),
            x='step',
            y='avg. rewards',
            hue='environment' if environment_hue else None,
            style='policy' if original_policy_as_label else 'experience replay',
            row='experience replay' if compare_experience_replay else None,
            col='environment' if compare_environments else None,
            aspect=aspect,
            estimator=estimator,
            ci=ci * 100 if estimator != 'mean' else 'sd',
            facet_kws=dict(sharex=False, sharey=False),
            kind='line',
            legend='brief',
            style_order=['distilled', 'original', 'distilled (best)'])

        if hide_title:
            for ax in g.axes.flatten():
                ax.set_title('')
        return g
    else:
        if compare_environments:
            hue = 'environment'
        elif compare_experience_replay:
            hue = 'experience replay'
        else:
            hue = None
        sns.set(font_scale=font_scale)
        return sns.lineplot(
            data=df.rename(columns={
                "value": "avg. rewards",
                "run": "experience replay",
                "event": "environment"}),
            x='step',
            y='avg. rewards',
            hue=hue,
            estimator=estimator,
            ci=ci * 100 if estimator != 'mean' else 'sd',
            style='policy' if (original_policy_as_label or plot_best) else 'experience replay', )
