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
        name: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    event_dict = {tag: {} for tag in tags}

    if name is None:
        name = os.path.join(log_dir, regex)

    for filename in glob.glob(os.path.join(log_dir, regex)):
        event_file = os.path.join(log_dir, filename)
        for event in summary_iterator(event_file):
            for value in event.summary.value:
                if value.tag in tags and event.step not in event_dict[value.tag]:
                    event_dict[value.tag][event.step] = tensor_util.MakeNdarray(value.tensor)

    df_by_tag = {}
    for tag, event in event_dict.items():
        df_by_tag[tag] = pd.DataFrame(data=np.array(list(event.values())),
                                      index=np.array(list(event.keys())),
                                      columns=[name])

    return df_by_tag


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

    df_by_tag = {tag: None for tag in tags}

    if dir_regex is None:
        log_dirs = [log_dir]
    else:
        log_dirs = glob.glob(os.path.join(log_dir, dir_regex))

    for i, log_dir in enumerate(log_dirs):
        for tag, df in get_event_arrays(log_dir, tags, name='run_{:d}'.format(i)).items():
            if df_by_tag[tag] is None:
                df_by_tag[tag] = df
            else:
                df_by_tag[tag] = pd.concat([df_by_tag[tag], df], axis='columns')

    for tag in tags:
        df_by_tag[tag] = df_by_tag[tag].rename_axis("step", axis='index')
        df_by_tag[tag] = df_by_tag[tag].rename_axis("run", axis='columns')
        for column in df_by_tag[tag].columns:
            df_by_tag[tag][column].fillna(df_by_tag[tag][column].median(), inplace=True)

    return df_by_tag


def get_interquantile_range(df: pd.DataFrame):
    return iqr(np.array([df[column].values for column in df.columns]), axis=0)


if __name__ == '__main__':
    # x = get_event_arrays('/home/florent/workspace/logs/05-07-21/CartPole-v0/',
    #                      regex='*seed=1111*PER*/*',
    #                      tags=['eval_elbo', 'policy_evaluation_avg_rewards', ],
    #                      name='test')
    import matplotlib.pyplot as plt

    x: Dict[str, pd.DataFrame] = plot_event(log_dir='/home/florent/workspace/logs/05-07-21/CartPole-v0/',
                                            dir_regex='*PER*',
                                            tags=['eval_elbo', 'policy_evaluation_avg_rewards', ])
    eval_elbo = x['eval_elbo']
    policy_eval_avg_rew = x['policy_evaluation_avg_rewards']

    fig, ax = plt.subplots()
    #  eval_elbo.plot(title='eval elbo', ax=ax, grid=True)
    #  policy_eval_avg_rew.plot(title='eval policy', ax=ax, grid=True)
    print(eval_elbo)
    # sns.lineplot(data=eval_elbo, x='step', style='run')

    # plt.show()

    from IPython.display import display

    display(eval_elbo)
    display(policy_eval_avg_rew)
