import os
from collections import namedtuple
from typing import Collection, List, Optional, Dict, Tuple
import glob
import numpy as np
from scipy.stats import iqr
from tensorflow.python.framework import tensor_util
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_event_arrays(
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


def plot_event(
        log_dir: str,
        tags: Collection[str],
        tags_renaming: Optional[Dict[str, str]] = None,
        scale_by_tag: Optional[Dict[str, Tuple[float, float]]] = None,
        dir_regex: Optional[str] = None,
):
    if tags_renaming is None:
        tags_renaming = {}
    if scale_by_tag is None:
        scale_by_tag = {}

    df = None

    if dir_regex is None:
        log_dirs = [log_dir]
    else:
        log_dirs = glob.glob(os.path.join(log_dir, dir_regex))

    for i, log_dir in enumerate(log_dirs):
        if df is None:
            df = get_event_arrays(log_dir, tags, name='run_{:d}'.format(i))
        else:
            df = df.append(get_event_arrays(log_dir, tags, name='run_{:d}'.format(i)))

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
        sns.relplot(
            data=df.rename(columns={"value": "ELBO", "run": "experience replay"}),
            x='step',
            y='ELBO',
            row='experience replay',
            aspect=2.5,
            kind='line',
            facet_kws=dict(sharey=False))
    else:
        sns.lineplot(
            data=df.rename(columns={"value": "ELBO", "run": "experience replay"}),
            x='step',
            y='ELBO',
            hue='experience replay' if compare_experience_replay else None)


def plot_histograms_per_step(
        df: pd.DataFrame,
        compare_experience_replay: bool = False,
        cbar: bool = False,
):
    sns.set_theme(style="darkgrid")

    def gen_mean_bucket_values(bucket):
        return np.repeat(
            np.floor(bucket[..., :2].mean()), bucket[..., 2].astype(int, casting='unsafe')
        ).astype(np.int32)

    _df = None

    for run in df['run'].unique():
        hist_run = df[df['run'] == run]

        data = np.array(
            [np.concatenate([gen_mean_bucket_values(bucket) for bucket in value], axis=-1)
             for value in hist_run['value']])

        data = pd.DataFrame({
            'state': data.flatten(),
            # 'step': hist_run['step'].repeat(data.shape[-1]),
            'step': np.array(
                [[step] * len(states) for step, states in zip(hist_run["step"], data)],
                dtype=np.int32
            ).flatten(),
        })
        data['run'] = run

        _df = data if _df is None else _df.append(data)

    sns.displot(
        data=_df.rename(columns={'run': 'experience replay'}),
        x='step',
        y='state',
        bins=30,
        cbar=cbar,
        # hue='experience replay' if compare_experience_replay else None,
        kind='hist',
        col='experience replay' if compare_experience_replay else None,
        common_bins=False)


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
        plt.plot(
            np.linspace(0, df['step'].max(), N, dtype=np.int64),
            np.ones(N) * original_policy_expected_rewards,
            linestyle='--',
            color='tab:green',
            alpha=0.5,
            linewidth=1)

        plt.text(0, 200, '$\pi$  ', ha='right', color='tab:green', )

    elif original_policy_expected_rewards is not None:
        df = df.assign(policy='distilled')
        df = df.append(df.assign(policy='original', value=200.))

    if compare_experience_replay and relplot:
        sns.relplot(
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
        sns.lineplot(
            data=df.rename(columns={"value": "rewards", "run": "experience replay"}),
            x='step',
            y='rewards',
            hue='experience replay' if compare_experience_replay else None,
            style='policy' if original_policy_as_label else 'experience replay', )
