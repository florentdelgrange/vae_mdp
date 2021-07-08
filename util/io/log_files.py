import os
from collections import namedtuple
from typing import Collection, List, Optional, Dict, Tuple
import glob
import numpy as np
import pandas
from scipy.stats import iqr
from tensorflow.python.framework import tensor_util
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import seaborn as sns


def get_event_arrays(
        log_dir: str,
        tags: Collection[str],
        regex: str = '**',
) -> pd.DataFrame:

    TaggedEvent = namedtuple('TaggedEvent', ['x_axis', 'y_axis', 'tags'])
    events = TaggedEvent(x_axis=[], y_axis=[], tags=[])

    for filename in glob.glob(os.path.join(log_dir, regex)):
        event_file = os.path.join(log_dir, filename)
        for event in summary_iterator(event_file):
            for value in event.summary.value:
                if value.tag in tags:
                    # event_dict[value.tag][event.step] = tensor_util.MakeNdarray(value.tensor)
                    events.x_axis.append(event.step)
                    events.y_axis.append(tensor_util.MakeNdarray(value.tensor))
                    events.tags.append(value.tag)

    return pd.DataFrame(data={'step': np.array(events.x_axis),
                              'value': np.array(events.y_axis),
                              'tag': events.tags, })


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


if __name__ == '__main__':
    x = get_event_arrays('/home/florent/workspace/logs/05-07-21/CartPole-v0/',
                         regex='*PER*/**',
                         tags=['eval_elbo', 'policy_evaluation_avg_rewards', ],)
    import matplotlib.pyplot as plt

    #  x: pd.DataFrame = plot_event(log_dir='/home/florent/workspace/logs/05-07-21/CartPole-v0/',
    #                               dir_regex='*PER*',
    #                               tags=['eval_elbo', 'policy_evaluation_avg_rewards', ])
    eval_elbo = x[x['tag'] == 'eval_elbo']
    policy_eval_avg_rew = x[x['tag'] == 'policy_evaluation_avg_rewards']

    # fig, ax = plt.subplots()
    #  eval_elbo.plot(title='eval elbo', ax=ax, grid=True)
    #  policy_eval_avg_rew.plot(title='eval policy', ax=ax, grid=True)
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=eval_elbo, x='step', y='value', hue='tag')
    plt.show()
    sns.lineplot(data=policy_eval_avg_rew, x='step', y='value', style='tag')
    plt.show()
