import os
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
        regex: str = '**',
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

    for filename in glob.glob(os.path.join(log_dir, regex)):
        event_file = os.path.join(log_dir, filename)
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
        compare_experience_replay: bool = False,
        relplot: bool = False,
        eval_elbo_tag: str = 'eval_elbo'
):
    df = df[df['tag'] == eval_elbo_tag]

    sns.set_theme(style="darkgrid")

    if compare_experience_replay and relplot:
        return sns.relplot(
            data=df.rename(columns={"value": "ELBO", "run": "experience replay"}),
            x='step',
            y='ELBO',
            row='experience replay',
            aspect=2.5,
            kind='line',
            facet_kws=dict(sharey=False))
    else:
        return sns.lineplot(
            data=df.rename(columns={"value": "ELBO", "run": "experience replay"}),
            x='step',
            y='ELBO',
            hue='experience replay' if compare_experience_replay else None)


def plot_histograms_per_step(
        df: pd.DataFrame,
        num_x_ticks: int = 5,
        num_y_ticks: int = 5,
        use_math_text: bool = False,
):

    tick = ticker.ScalarFormatter(useOffset=True, useMathText=use_math_text)
    tick.set_powerlimits((0, 0))

    # we assume that all buckets have the same range, according to the way tf summaries records histograms
    buckets = df['value'].head(1).to_numpy()[0][..., :2]
    nrows = len(df['event'].unique()) if 'event' in df else 1
    ncols = len(df['run'].unique()) if 'run' in df else 1

    fig, axs = plt.subplots(nrows, ncols)

    def plot_histogram(df, ax, display_x_ticks=True, display_y_ticks=True, ax_title=None):

        data = np.stack(df['value'].to_numpy())[..., 2]

        def round_to_base(x, base, upper: bool = False):
            if upper:
                return base * np.ceil(x / base)
            else:
                return base * np.round(x / base)

        base = round_to_base(buckets.flatten().max() // 20, 50)
        xticks = np.linspace(0, df['step'].unique().max(), num=data.shape[0], dtype=np.int32)
        xticks = round_to_base(xticks, base=df['step'].unique().max() // (num_x_ticks * 2))
        if use_math_text:
            xticks = [u"${}$".format(tick.format_data(x) if x != 0. else str(0))
                      for x in np.flipud(xticks).astype(float)]
        else:
            xticks = [tick.format_data(x) if x != 0. else str(0) for x in np.flipud(xticks).astype(float)]
        yticks = np.array([round_to_base(bucket.mean(), base) for bucket in buckets], dtype=np.int32)

        _df = pd.DataFrame(
            data=np.flipud(data.transpose()),
            columns=np.flipud(np.array(xticks)),
            index=np.flipud(yticks))

        ax = sns.heatmap(
            _df,
            cmap='Blues',
            xticklabels=data.shape[0] // num_x_ticks if display_x_ticks else display_x_ticks,
            yticklabels=len(buckets) // num_y_ticks if display_y_ticks else display_y_ticks,
            cbar=False,
            ax=ax)

        if display_x_ticks:
            ax.set_xlabel("step")
        if display_y_ticks:
            ax.set_ylabel("state")
        if ax_title is not None:
            ax.set_title(ax_title)

        for label in ax.get_yticklabels():
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
                _df = df[df['event'] == df['event'].unique()[row]] if 'event' in df else df
                _df = df[df['run'] == df['run'].unique()[column]] if 'run' in df else df
                plot_histogram(
                    _df,
                    ax,
                    display_x_ticks=(row == nrows - 1),
                    display_y_ticks=(column == 0),
                    ax_title='experience replay = {}'.format(df['run'].unique()[column])
                             if 'run' in df and row == 0 else None)

    return fig

def plot_policy_evaluation(
        df: pd.DataFrame,
        original_policy_expected_rewards: Optional[float] = None,
        compare_experience_replay: bool = False,
        relplot: bool = False,
        original_policy_as_label: bool = True,
        policy_evaluation_avg_rewards_tag: str = 'policy_evaluation_avg_rewards'
):
    df = df[df['tag'] == policy_evaluation_avg_rewards_tag]

    sns.set_theme(style="darkgrid")

    if original_policy_expected_rewards is not None and not original_policy_as_label:
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
        df = df.assign(policy='distilled')
        df = df.append(df.assign(policy='original', value=200.))

    if compare_experience_replay and relplot:
        return sns.relplot(
            data=df.rename(columns={"value": "rewards", "run": "experience replay"}),
            x='step',
            y='rewards',
            # hue='experience replay',
            style='policy' if original_policy_as_label else 'experience replay',
            row='experience replay',
            aspect=2.5,
            facet_kws=dict(sharey=False),
            kind='line')
    else:
        return sns.lineplot(
            data=df.rename(columns={"value": "rewards", "run": "experience replay"}),
            x='step',
            y='rewards',
            hue='experience replay' if compare_experience_replay else None,
            style='policy' if original_policy_as_label else 'experience replay', )