import numpy as np
import tensorflow as tf
import optuna
import h5py
from pathlib import Path
import importlib

import policies
import reinforcement_learning
import train
import variational_action_discretizer
import variational_mdp


def optimize_hyperparameters(study_name, optimize_trial, storage='sqlite:///hpopt.db', n_trials=100):
    sqlite_timeout = 300
    storage = optuna.storages.RDBStorage(
        storage,
        engine_kwargs={'connect_args': {'timeout': sqlite_timeout}}
    )
    study = optuna.create_study(study_name=study_name,
                                storage=storage,
                                load_if_exists=True,
                                direction='maximize')
    study.optimize(optimize_trial, n_trials=n_trials)


def search(
        environment_name: str,
        environment_suite_name: str,
        state_shape,
        action_shape,
        reward_shape,
        label_shape,
        fixed_parameters: dict,
        policy_path: str,
        num_steps: int = 1e6,
        study_name='study',
        n_trials=100):
    environment_suite = None
    try:
        environment_suite = importlib.import_module('tf_agents.environments.' + environment_suite_name)
    except BaseException as err:
        serr = str(err)
        print("An error occurred when loading the module '" + environment_suite_name + "': " + serr)

    specs = train.get_environment_specs(environment_suite, environment_name, latent_policy=True)

    def suggest_hp(trial):

        defaults = algo_args
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512])
        collect_steps_per_iteration = trial.suggest_categorical('collect_steps_per_iteration', [1, 4, 8, 16, 64, 128])
        latent_state_size = trial.suggest_int('latent_state_size', 8, 20)
        number_of_discrete_actions = trial.suggest_int('number_of_discrete_actions', 2, 16)
        one_output_per_action = trial.suggest_categorical('one_output_per_action', [True, False])
        relaxed_state_encoder_temperature = trial.suggest_float('relaxed_state_encoder_temperature', 0.1, 0.99)
        relaxed_state_prior_temperature = trial.suggest_float('relaxed_state_prior_temperature', 0.1, 0.99)
        encoder_temperature = trial.suggest_float('encoder_temperature', 0.1, 0.99)
        prior_temperature = trial.suggest_float('prior_temperature', 0.1, 0.99)
        kl_annealing_growth_rate = trial.suggest_float('kl_annealing_growth_rate', 1e-5, 1e-2, log=True)
        entropy_regularizer_decay_rate = trial.suggest_float('entropy_regularizer_decay_rate', 1e-5, 1e-2, log=True)
        prioritized_experience_replay = trial.suggest_categorical('prioritized_experience_replay', [True, False])
        priority_exponent = trial.suggest_float('priority_exponent', 1e-1, 1.)
        importance_sampling_exponent = trial.suggest_float('importance_sampling_exponent', 1e-1, 1.)
        importance_sampling_exponent_growth_rate = trial.suggest_float(
            'importance_sampling_exponent_growth_rate', 1e-5, 1e-2, log=True)
        neurons = trial.suggest_int('neurons', 16, 512, step=16)
        hidden = trial.suggest_int('hidden', 1, 5)
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu'])

        for attr in ['learning_rate', 'batch_size', 'collect_steps_per_iteration', 'latent_state_size',
                     'number_of_discrete_actions', 'one_output_per_action', 'state_encoder_temperature',
                     'state_prior_temperature', 'action_encoder_temperature', 'action_prior_temperature',
                     'kl_annealing_growth_rate', 'entropy_regularizer_decay_rate', 'prioritized_experience_replay',
                     'priority_exponent', 'importance_sampling_exponent', 'importance_sampling_exponent_growth_rate',
                     'neurons', 'hidden', 'activation']:
            defaults[attr] = locals()[attr]

        return defaults

    def optimize_trial(trial):
        hyperparameters = suggest_hp(trial)

        for component_name in ['encoder', 'transition', 'label_transition', 'reward', 'decoder', 'discrete_policy']:
            hyperparameters[component_name + '_layers'] = hyperparameters['hidden'] * [hyperparameters['neurons']]
        q, p_t, p_l_t, p_r, p_decode, latent_policy = train.generate_network_components(hyperparameters, name='state')

        tf.random.set_seed(hyperparameters['seed'])

        vae_mdp = variational_mdp.VariationalMarkovDecisionProcess(
            state_shape=specs.state_shape, action_shape=specs.action_shape,
            reward_shape=specs.reward_shape, label_shape=specs.label_shape,
            encoder_network=q, transition_network=p_t, label_transition_network=p_l_t,
            reward_network=p_r, decoder_network=p_decode,
            latent_policy_network=latent_policy,
            latent_state_size=hyperparameters['latent_state_size'],
            mixture_components=fixed_parameters['mixture_components'],
            encoder_temperature=hyperparameters['relaxed_state_encoder_temperature'],
            prior_temperature=hyperparameters['relaxed_state_prior_temperature'],
            encoder_temperature_decay_rate=0.,
            prior_temperature_decay_rate=0.,
            entropy_regularizer_scale_factor=fixed_parameters['entropy_regularizer_scale_factor'],
            entropy_regularizer_decay_rate=hyperparameters['entropy_regularizer_decay_rate'],
            entropy_regularizer_scale_factor_min_value=fixed_parameters['entropy_regularizer_scale_factor_min_value'],
            marginal_entropy_regularizer_ratio=fixed_parameters['marginal_entropy_regularizer_ratio'],
            kl_scale_factor=fixed_parameters['kl_annealing_scale_factor'],
            kl_annealing_growth_rate=hyperparameters['kl_annealing_growth_rate'],
            multivariate_normal_full_covariance=fixed_parameters['full_covariance'],
            full_optimization=True,
            importance_sampling_exponent=hyperparameters['importance_sampling_exponent'],
            importance_sampling_exponent_growth_rate=hyperparameters['importance_sampling_exponent_growth_rate'],
            evaluation_window_size=10)

        q, p_t, p_l_t, p_r, p_decode, latent_policy = train.generate_network_components(hyperparameters, name='action')
        vae_mdp = variational_action_discretizer.VariationalActionDiscretizer(
            vae_mdp=vae_mdp,
            number_of_discrete_actions=hyperparameters['number_of_discrete_actions'],
            action_encoder_network=q, transition_network=p_t, action_label_transition_network=p_l_t,
            reward_network=p_r, action_decoder_network=p_decode,
            latent_policy_network=latent_policy,
            encoder_temperature=hyperparameters['encoder_temperature'],
            prior_temperature=hyperparameters['prior_temperature'],
            encoder_temperature_decay_rate=0.,
            prior_temperature_decay_rate=0.,
            one_output_per_action=hyperparameters['one_output_per_action'],
            relaxed_state_encoding=True,
            full_optimization=True,
            reconstruction_mixture_components=1, )

        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate'])

        environment = vae_mdp.initialize_environment(
            environment_suite=environment_suite,
            env_name=environment_name,
            parallel_environments=fixed_parameters['parallel_env'] > 0,
            num_parallel_environments=fixed_parameters['parallel_env'],
            environment_seed=fixed_parameters['seed'],
            use_prioritized_replay_buffer=hyperparameters['prioritized_experience_replay'], )

        dataset_handler = vae_mdp.initialize_dataset_components(
            env=environment,
            policy=policies.SavedTFPolicy(fixed_parameters['policy_path'], specs.time_step_spec, specs.action_spec),
            labeling_function=reinforcement_learning.labeling_functions[environment_name],
            batch_size=hyperparameters['batch_size'],
            manager=None,
            use_prioritized_replay_buffer=hyperparameters['prioritized_experience_replay'],
            priority_exponent=hyperparameters['priority_exponent'],
            buckets_based_priorities=hyperparameters['buckets_based_priorities'],
            discrete_action_space=False,
            collect_steps_per_iteration=hyperparameters['collect_steps_per_iteration'],
            initial_collect_steps=int(1e4),
            replay_buffer_capacity=int(1e6))

        policy_evaluation_driver = vae_mdp.initialize_policy_evaluation_driver(
            environment_suite=environment_suite,
            env_name=environment_name,
            labeling_function=reinforcement_learning.labeling_functions[environment_name],
            policy_evaluation_num_episodes=30)

        initial_training_steps = int(1e5)
        training_steps_per_iteration = int(1e4)

        vae_mdp.train_from_policy(
            policy=None,
            environment_suite=environment_suite,
            env_name=environment_name,
            labeling_function=reinforcement_learning.labeling_functions[environment_name],
            num_iterations=initial_training_steps,
            use_prioritized_replay_buffer=hyperparameters['prioritized_experience_replay'],
            global_step=global_step,
            optimizer=optimizer,
            eval_steps=0,
            save_directory=None,
            policy_evaluation_num_episodes=30,
            environment=environment,
            dataset_handler=dataset_handler,
            policy_evaluation_driver=policy_evaluation_driver,
            close_at_the_end=False)

        for _ in range(initial_training_steps, num_steps, training_steps_per_iteration):
            intermediate_value = vae_mdp.train_from_policy(
                policy=None,
                environment_suite=environment_suite,
                env_name=environment_name,
                labeling_function=reinforcement_learning.labeling_functions[environment_name],
                num_iterations=training_steps_per_iteration,
                use_prioritized_replay_buffer=hyperparameters['prioritized_experience_replay'],
                global_step=global_step,
                optimizer=optimizer,
                eval_steps=0,
                save_directory=None,
                policy_evaluation_num_episodes=30,
                environment=environment,
                dataset_handler=dataset_handler,
                policy_evaluation_driver=policy_evaluation_driver,
                close_at_the_end=False)

            # Report intermediate objective value.
            trial.report(intermediate_value, global_step.numpy())

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

        return intermediate_value

    optimize_hyperparameters(study_name,
                             optimize_trial,
                             n_trials=n_trials)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', required=True, type=str)
    parser.add_argument('--env', default='Pendulum-v0', type=str)
    parser.add_argument('--hp', default=None, type=str)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--eval-freq', default=np.inf, type=float)
    parser.set_defaults(optimize=False)

    args, unk_args = parser.parse_known_args()
    algo_args = get_config(args.algo, args.env, parse_cmdline_kwargs(unk_args), config_path=args.hp)
    try:
        timesteps = algo_args.pop('timesteps')
    except KeyError:
        print('no timesteps provided, using default 1e6')
        timesteps = 1e6

    optimize_agent(
        args.algo,
        args.env,
        algo_args,
        timesteps=timesteps,
        study_name=f'{args.env}_{args.algo}',
        n_trials=1000,
        runs=3,
        eval_freq=args.eval_freq)
