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


def get_event_arrays(
        log_dir: str,
        tags: Optional[Collection[str]] = None,
        regex: str = '**',
        tags_renaming: Optional[Dict[str, str]] = None,
        run_name: Optional[str] = None,
        event_name: Optional[str] = None
) -> pd.DataFrame:
    if tags_renaming is None:
        tags_renaming = {}

    TaggedEvent = namedtuple('TaggedEvent', ['x_axis', 'y_axis', 'tags'])
    events = TaggedEvent(x_axis=[], y_axis=[], tags=[])

    for filename in glob.glob(os.path.join(log_dir, regex)):
        event_file = os.path.join(log_dir, filename)
        for event in summary_iterator(event_file):
            for value in event.summary.value:
                if tags is not None:
                    if value.tag in tags:
                        events.x_axis.append(event.step)
                        events.y_axis.append(tensor_util.MakeNdarray(value.tensor))
                        events.tags.append(tags_renaming.get(value.tag, value.tag))
                else:
                    events.x_axis.append(event.step)
                    events.y_axis.append(tensor_util.MakeNdarray(value.tensor))
                    events.tags.append(tags_renaming.get(value.tag, value.tag))

    data = {'step': np.array(events.x_axis),
            'value': events.y_axis,
            'tag': events.tags, }

    if run_name is not None:
        data['run'] = run_name
    if event_name is not None:
        data['event'] = event_name

    return pd.DataFrame(data)


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


def add_moving_mean(df: pd.DataFrame) -> pd.DataFrame:
    pass

if __name__ == '__main__':
    cartpole = get_event_arrays('/home/florent/workspace/logs/05-07-21/CartPole-v0/',
                         regex='*PER*/**',
                         tags=['eval_elbo', 'policy_evaluation_avg_rewards', ],
                         tags_renaming={'eval_elbo': 'ELBO'},
                         run_name='prioritized replay',
                         event_name='Cartpole-v0')
    cartpole = cartpole.append(
        get_event_arrays('/home/florent/workspace/logs/05-07-21/CartPole-v0/',
                         regex='*[!PER]*/**',
                         tags=['eval_elbo', 'policy_evaluation_avg_rewards', ],
                         tags_renaming={'eval_elbo': 'ELBO'},
                         run_name='uniform replay',
                         event_name='CartPole-v0'))
    import matplotlib.pyplot as plt

    eval_elbo = cartpole[cartpole['tag'] == 'ELBO']
    policy_eval_avg_rew = cartpole[cartpole['tag'] == 'policy_evaluation_avg_rewards']
    distortion = cartpole[cartpole['tag']]

    # fig, ax = plt.subplots()
    #  eval_elbo.plot(title='eval elbo', ax=ax, grid=True)
    #  policy_eval_avg_rew.plot(title='eval policy', ax=ax, grid=True)
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=eval_elbo, x='step', y='value', style='run', hue='run')
    plt.show()
    sns.lineplot(data=policy_eval_avg_rew, x='step', y='value', hue='run', style='run')
    plt.show()
    sns.relplot(data=x, x='step', y='value', hue='run', style='run', col='tag', kind='line')