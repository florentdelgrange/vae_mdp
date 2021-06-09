import logging
import os
import sys
import time
from typing import Optional

import tensorflow as tf
import optuna
import importlib

import policies
import reinforcement_learning
from train import get_environment_specs, generate_network_components
import variational_action_discretizer
import variational_mdp


def optimize_hyperparameters(study_name, optimize_trial, storage=None, n_trials=100):
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    if storage is None:
        if not os.path.exists('studies'):
            os.makedirs('studies')
        storage = 'sqlite:///studies/{}.db'.format(study_name)

    sqlite_timeout = 300
    storage = optuna.storages.RDBStorage(
        storage,
        engine_kwargs={'connect_args': {'timeout': sqlite_timeout}})
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize')

    return study.optimize(optimize_trial, n_trials=n_trials)


def search(
        fixed_parameters: dict,
        num_steps: int = 1e6,
        study_name='study',
        n_trials=100,
        wall_time: Optional[str] = None,
):
    start_time = time.time()

    environment_suite_name = fixed_parameters['env_suite']
    environment_name = fixed_parameters['environment']
    environment_suite = None
    try:
        environment_suite = importlib.import_module('tf_agents.environments.' + environment_suite_name)
    except BaseException as err:
        serr = str(err)
        print("An error occurred when loading the module '" + environment_suite_name + "': " + serr)

    specs = get_environment_specs(
        environment_suite=environment_suite,
        environment_name=environment_name,
        discrete_action_space=not fixed_parameters['action_discretizer'],
        allows_for_parallel_environment=fixed_parameters['parallel_env'] > 1)

    def suggest_hyperparameters(trial):

        defaults = {}
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512])
        neurons = trial.suggest_int('neurons', 16, 512, step=16)
        hidden = trial.suggest_int('hidden', 1, 5)
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu'])
        latent_state_size = trial.suggest_int(
            'latent_state_size', specs.label_shape[0] + 2, max(20, specs.label_shape[0] + 8))
        relaxed_state_encoder_temperature = trial.suggest_float('relaxed_state_encoder_temperature', 1e-6, 1.)
        relaxed_state_prior_temperature = trial.suggest_float('relaxed_state_prior_temperature', 1e-6, 1.)
        kl_annealing_growth_rate = trial.suggest_float('kl_annealing_growth_rate', 1e-5, 1e-2, log=True)
        entropy_regularizer_decay_rate = trial.suggest_float('entropy_regularizer_decay_rate', 1e-5, 1e-2, log=True)
        prioritized_experience_replay = trial.suggest_categorical('prioritized_experience_replay', [True, False])
        if prioritized_experience_replay:
            collect_steps_per_iteration = trial.suggest_int(
                'prioritized_experience_replay_collect_steps_per_iteration', 1, batch_size // 8)
            buckets_based_priorities = trial.suggest_categorical('buckets_based_priorities', [True, False])
            priority_exponent = trial.suggest_float('priority_exponent', 1e-1, 1.)
            importance_sampling_exponent = trial.suggest_float('importance_sampling_exponent', 1e-1, 1.)
            importance_sampling_exponent_growth_rate = trial.suggest_float(
                'importance_sampling_exponent_growth_rate', 1e-5, 1e-2, log=True)
        else:
            collect_steps_per_iteration = trial.suggest_int(
                'uniform_replay_buffer_collect_steps_per_iteration', 1, batch_size)
            # default values
            buckets_based_priorities = False
            priority_exponent = 1.
            importance_sampling_exponent = 1.
            importance_sampling_exponent_growth_rate = 1.

        if fixed_parameters['action_discretizer']:
            number_of_discrete_actions = trial.suggest_int(
                'number_of_discrete_actions', 2, fixed_parameters['number_of_discrete_actions'])
            encoder_temperature = trial.suggest_float(
                'encoder_temperature', 1e-6, 1. / (number_of_discrete_actions - 1))
            prior_temperature = trial.suggest_float(
                'prior_temperature', 1e-6, 1. / (number_of_discrete_actions - 1))
            one_output_per_action = trial.suggest_categorical('one_output_per_action', [True, False])

        for attr in ['learning_rate', 'batch_size', 'collect_steps_per_iteration', 'latent_state_size',
                     'relaxed_state_encoder_temperature', 'relaxed_state_prior_temperature',
                     'kl_annealing_growth_rate', 'entropy_regularizer_decay_rate', 'prioritized_experience_replay',
                     'neurons', 'hidden', 'activation', 'priority_exponent', 'importance_sampling_exponent',
                     'importance_sampling_exponent_growth_rate',
                     'buckets_based_priorities'] + [
                        'encoder_temperature', 'prior_temperature', 'number_of_discrete_actions',
                        'one_output_per_action'] if fixed_parameters['action_discretizer'] else []:
            defaults[attr] = locals()[attr]

        return defaults

    def optimize_trial(trial: optuna.Trial):
        hyperparameters = suggest_hyperparameters(trial)

        print("Suggested hyperparameters")
        for key in hyperparameters.keys():
            print("{}={}".format(key, hyperparameters[key]))

        for component_name in ['encoder', 'transition', 'label_transition', 'reward', 'decoder', 'discrete_policy']:
            hyperparameters[component_name + '_layers'] = hyperparameters['hidden'] * [hyperparameters['neurons']]
        q, p_t, p_l_t, p_r, p_decode, latent_policy = generate_network_components(
            hyperparameters, name='variational_mdp')

        tf.random.set_seed(fixed_parameters['seed'])

        evaluation_window_size = fixed_parameters['evaluation_window_size']
        vae_mdp = variational_mdp.VariationalMarkovDecisionProcess(
            state_shape=specs.state_shape, action_shape=specs.action_shape,
            reward_shape=specs.reward_shape, label_shape=specs.label_shape,
            encoder_network=q, transition_network=p_t, label_transition_network=p_l_t,
            reward_network=p_r, decoder_network=p_decode,
            latent_policy_network=(latent_policy if fixed_parameters['latent_policy'] else None),
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
            evaluation_window_size=evaluation_window_size,
            evaluation_criterion=variational_mdp.EvaluationCriterion.MAX)

        if fixed_parameters['action_discretizer']:
            q, p_t, p_l_t, p_r, p_decode, latent_policy = generate_network_components(
                hyperparameters, name='variational_action_discretizer')
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
            parallel_environments=fixed_parameters['parallel_env'] > 1,
            num_parallel_environments=fixed_parameters['parallel_env'],
            collect_steps_per_iteration=hyperparameters['collect_steps_per_iteration'],
            environment_seed=fixed_parameters['seed'],
            use_prioritized_replay_buffer=hyperparameters['prioritized_experience_replay'], )

        policy = policies.SavedTFPolicy(fixed_parameters['policy_path'], specs.time_step_spec, specs.action_spec)
        dataset_components = vae_mdp.initialize_dataset_components(
            env=environment,
            policy=policy,
            labeling_function=reinforcement_learning.labeling_functions[environment_name],
            batch_size=hyperparameters['batch_size'],
            manager=None,
            use_prioritized_replay_buffer=hyperparameters['prioritized_experience_replay'],
            priority_exponent=hyperparameters['priority_exponent'],
            buckets_based_priorities=hyperparameters['buckets_based_priorities'],
            discrete_action_space=not fixed_parameters['action_discretizer'],
            collect_steps_per_iteration=hyperparameters['collect_steps_per_iteration'],
            initial_collect_steps=int(1e4),
            replay_buffer_capacity=int(1e6))

        policy_evaluation_driver = vae_mdp.initialize_policy_evaluation_driver(
            environment_suite=environment_suite,
            env_name=environment_name,
            labeling_function=reinforcement_learning.labeling_functions[environment_name],
            policy_evaluation_num_episodes=30)

        initial_training_steps = evaluation_window_size * num_steps // 100
        training_steps_per_iteration = num_steps // 100

        def train_model(training_steps):
            return vae_mdp.train_from_policy(
                policy=policy,
                environment_suite=environment_suite,
                env_name=environment_name,
                labeling_function=reinforcement_learning.labeling_functions[environment_name],
                training_steps=training_steps,
                logs=True,
                log_dir=os.path.join('studies', 'logs'),
                log_name='{:d}'.format(trial._trial_id),
                use_prioritized_replay_buffer=hyperparameters['prioritized_experience_replay'],
                global_step=global_step,
                optimizer=optimizer,
                eval_steps=1000,
                annealing_period=fixed_parameters['annealing_period'],
                start_annealing_step=fixed_parameters['start_annealing_step'],
                eval_and_save_model_interval=training_steps_per_iteration,
                save_directory=None,
                policy_evaluation_num_episodes=30,
                environment=environment,
                dataset_components=dataset_components,
                policy_evaluation_driver=policy_evaluation_driver,
                close_at_the_end=False,
                display_progressbar=fixed_parameters['display_progressbar'],
                start_time=start_time,
                wall_time=wall_time,
                memory_limit=fixed_parameters['memory'] if fixed_parameters['memory'] > 0. else None)

        result = train_model(initial_training_steps)
        score = result['score']

        if result['continue']:
            for step in range(initial_training_steps, num_steps, training_steps_per_iteration):

                try:
                    result = train_model(step + training_steps_per_iteration)
                except:
                    print("The training has stopped prematurely due to an error.")
                    result['continue'] = False

                score = result['score']
                print("Step {} intermediate score: {}".format(step + training_steps_per_iteration, score))

                # Report intermediate objective value.
                trial.report(float(score), step=step + training_steps_per_iteration)

                # Handle pruning based on the intermediate value.
                if fixed_parameters['prune_trials'] and trial.should_prune():
                    raise optuna.TrialPruned()

                if not result['continue']:
                    break

        dataset_components.close_fn()

        #  for key, value in vae_mdp.loss_metrics.items():
        #      trial.set_user_attr(key, float(value.result()))

        return float(score)

    return optimize_hyperparameters(study_name, optimize_trial, n_trials=n_trials)
