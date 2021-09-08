import os
import numpy as np
import seaborn as sns
from absl import flags, app
from matplotlib import pyplot as plt
from log_files import get_event_dataframe, plot_policy_evaluation, plot_histograms_per_step


def main(argv):
    del argv
    params = FLAGS.flag_values_dict()
    if params['plot_histograms'] and params['uniform_logdir'] is None:
        raise ValueError("Uniform log directories should be provided")
    log_dirs = params['logdir']
    event_names = params['event_name']
    estimator = {'mean': 'mean', 'median': np.median}
    estimator = estimator[params['estimator']]
    uniform_log_dirs = params['uniform_logdir']
    enforce_legend = params['enforce_legend']

    for i, dir in enumerate(log_dirs):
        log_dirs[i] = os.path.join(dir, '*.v2')

    if params['rl_policy_return'] is not None:
        original_policy_expected_rewards = {
            event_name: rewards for event_name, rewards in zip(
                event_names[:len(params['rl_policy_return'])],
                params['rl_policy_return'])
        }
    else:
        original_policy_expected_rewards = None

    if event_names is None:
        event_names = []
    if len(event_names) < len(log_dirs):
        event_names += log_dirs[len(event_names):]
    if params['plot_histograms'] and len(uniform_log_dirs) < len(log_dirs):
        raise ValueError("A uniform replay log directory should be provided for each (PER) log directory.")

    sns.set(rc={"figure.dpi": 150, 'savefig.dpi': 300})
    sns.set_context('paper')

    if not os.path.exists(os.path.join('evaluation', 'plots')):
        os.makedirs(os.path.join('evaluation', 'plots'))

    print("Generating dataframes...")
    logs = {event_name: log_dir for event_name, log_dir in zip(event_names, log_dirs)}

    df = None

    for env, log_dir in logs.items():
        _df = get_event_dataframe(
            log_dir,
            tags=['eval_elbo', 'local_reward_loss', 'local_probability_loss', 'encoder_entropy',
                  'state_encoder_entropy', 'policy_evaluation_avg_rewards', 'action_encoder_entropy',
                  'eval_rate', 'eval_distortion'],
            run_name='prioritized replay',
            event_name=env, )
        df = _df if df is None else df.append(_df)

    if not params['plot_histograms']:
        print("Done.\n")
        print("Plotting Distortion/Rate/ELBO...")
        _df = df[df['tag'] == 'eval_distortion']
        _df = _df.append(df[df['tag'] == 'eval_rate'])
        _df = _df.append(df[df['tag'] == 'eval_elbo'])
        # remove outliers
        _df = _df.drop(_df[np.abs(_df['value']) > 100].index)

        sns.set_context('paper')
        sns.set(font_scale=1.8)
        g = sns.relplot(
            data=_df.replace(
                {'tag': {'eval_distortion': 'distortion', 'eval_rate': 'rate', 'eval_elbo': 'ELBO'}}
            ).rename(columns={'tag': 'metric', 'event': 'environment'})[_df['step'] <= int(1e6)],
            x='step',
            y='value',
            hue='environment',
            col='metric',
            estimator=estimator,
            ci=85 if estimator != 'mean' else 'sd',
            legend=enforce_legend,
            facet_kws=dict(sharey=False),
            kind='line',
            seed=42,
            aspect=1.25, )

        g.set(ylabel=None)
        plt.savefig(os.path.join('evaluation', 'plots', 'distortion_rate_elbo.pdf'), bbox_inches='tight')
        print("Done.\n")

        print("Plotting PAC local losses bounds...")
        _df = df[df['tag'] == 'local_probability_loss']
        _df = _df.append(df[df['tag'] == 'local_reward_loss'])
        _df = _df[_df['value'] <= 1]
        _df = _df.replace({
            'tag': {'local_probability_loss': 'transition',
                    'local_reward_loss': 'reward'}}
        ).rename(columns={
            'tag': 'local loss',
            'value': 'PAC bound',
            'event': 'environment'})
        _df = _df[_df['step'] <= int(1e6)]

        plt.figure(figsize=(8, 6.5))
        sns.lineplot(
            data=_df,
            x='step',
            y='PAC bound',
            hue='environment',
            style='local loss',
            estimator=estimator,
            ci=90 if estimator != 'mean' else 'sd',
            legend=False)

        plt.savefig(os.path.join("evaluation", "plots", "local_losses_2.pdf"), bbox_inches='tight')
        print("Done.\n")

        print("Plotting policy evaluation...")
        plot_policy_evaluation(
            df[df['step'] <= int(1e6)],
            plot_best=True,
            original_policy_expected_rewards=original_policy_expected_rewards,
            compare_environments=True,
            aspect=1.0197503069995908,
            relplot=True,
            estimator=estimator,
            ci=0.9 if estimator != 'mean' else 'sd',
            font_scale=1.8,
            environment_hue=True,
            hide_title=True)

        plt.savefig(os.path.join('evaluation', 'plots', "eval_policy.pdf"))
        print("Done.\n")
    else:
        for key in event_names:
            logs[key + '_PER'] = logs[key]
        for key, value in zip(event_names, uniform_log_dirs):
            logs[key + '_uniform'] = value

        hist = None
        for environment in event_names:
            _df = get_event_dataframe(
                logs[environment + '_PER'],
                tags=['state_frequency'],
                run_name='prioritized',
                event_name=environment,
                value_dtype=None)
            _df = _df.append(
                get_event_dataframe(
                    os.path.join(logs[environment + '_uniform'], '**', '*.v2'),
                    exclude_pattern=os.path.join(logs[environment + '_uniform'], '**', '*PER*', '**', '*.v2'),
                    tags=['state_frequency'],
                    run_name='uniform',
                    event_name=environment,
                    value_dtype=None))
            hist = _df if hist is None else hist.append(_df)

        hist = hist[hist['step'] <= int(1e6)]
        print("Done.\n")

        plt.rcParams['figure.dpi'] = 300

        print("Plotting latent space histograms...")
        for environment in event_names:
            plot_histograms_per_step(hist[hist['event'] == environment], num_x_ticks=5, num_y_ticks=4, aspect=2)
            plt.savefig(
                os.path.join('evaluation', 'plots', '{}_histogram.pdf'.format(environment)), bbox_inches='tight')
        print("Done.")


if __name__ == '__main__':
    flags.DEFINE_multi_string(
        name='logdir',
        default=None,
        required=True,
        help="Directory containing logs (TF events) to be plotted."
             "(Glob) regex are authorized.")
    flags.DEFINE_multi_string(
        name='event_name',
        default=None,
        required=True,
        help="Name of the event to be plotted (should be provided in the same order than log directories).")
    flags.DEFINE_enum(
        name='estimator',
        default='median',
        enum_values=['median', 'mean'],
        help='Whether to use the median or the mean to draw solid lines.'
             'Standard deviation is drawn as shaded are if using the mean,'
             'and a confidence interval if using the median')
    flags.DEFINE_bool(
        name='plot_histograms',
        default=False,
        help='Whether to plot replay buffer histograms or not.'
             'If set, log directories for uniform replay buffers should be provided')
    flags.DEFINE_multi_string(
        name='uniform_logdir',
        default=None,
        help='Log directories containing logs of instances using uniform replay buffers.'
             'Should be provided in the same order than the other log directories.')
    flags.DEFINE_bool(
        name='enforce_legend',
        default=False,
        help="Enforce to display the legend on every plots."
    )
    flags.DEFINE_multi_float(
        name='rl_policy_return',
        default=None,
        help='Returns achieved by the original RL policies (should be provided in the same order than log directories)'
    )

    FLAGS = flags.FLAGS
    app.run(main)
