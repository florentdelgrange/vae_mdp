import os
import tensorflow as tf
import optuna
import importlib

import policies
import reinforcement_learning
from train import get_environment_specs, generate_network_components
import variational_action_discretizer
import variational_mdp


def optimize_hyperparameters(study_name, optimize_trial, storage=None, n_trials=100):
    if storage is None:
        if not os.path.exists('studies'):
            os.makedirs('studies')
        storage = 'sqlite:///studies/{}.db'.format(study_name)

    sqlite_timeout = 300
    storage = optuna.storages.RDBStorage(
        storage,
        engine_kwargs={'connect_args': {'timeout': sqlite_timeout}}
    )
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
        n_trials=100):
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
        discrete_action_space=not fixed_parameters['action_discretizer'])

    def suggest_hyperparameters(trial):

        defaults = {}
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512])
        collect_steps_per_iteration = trial.suggest_categorical(
            'collect_steps_per_iteration', [1, 4, 8, 16, 32, 64, 128])
        latent_state_size = trial.suggest_int('latent_state_size', specs.label_shape[0] + 1, 20)
        relaxed_state_encoder_temperature = trial.suggest_float('relaxed_state_encoder_temperature', 0.1, 0.99)
        relaxed_state_prior_temperature = trial.suggest_float('relaxed_state_prior_temperature', 0.1, 0.99)
        kl_annealing_growth_rate = trial.suggest_float('kl_annealing_growth_rate', 1e-5, 1e-2, log=True)
        entropy_regularizer_decay_rate = trial.suggest_float('entropy_regularizer_decay_rate', 1e-5, 1e-2, log=True)
        prioritized_experience_replay = trial.suggest_categorical('prioritized_experience_replay', [True, False])
        buckets_based_priorities = trial.suggest_categorical('buckets_based_priorities', [True, False])
        priority_exponent = trial.suggest_float('priority_exponent', 1e-1, 1.)
        importance_sampling_exponent = trial.suggest_float('importance_sampling_exponent', 1e-1, 1.)
        importance_sampling_exponent_growth_rate = trial.suggest_float(
            'importance_sampling_exponent_growth_rate', 1e-5, 1e-2, log=True)
        neurons = trial.suggest_int('neurons', 16, 512, step=16)
        hidden = trial.suggest_int('hidden', 1, 5)
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu'])
        if fixed_parameters['action_discretizer']:
            encoder_temperature = trial.suggest_float('encoder_temperature', 0.1, 0.99)
            prior_temperature = trial.suggest_float('prior_temperature', 0.1, 0.99)
            number_of_discrete_actions = trial.suggest_int('number_of_discrete_actions', 2, 16)
            one_output_per_action = trial.suggest_categorical('one_output_per_action', [True, False])

        for attr in ['learning_rate', 'batch_size', 'collect_steps_per_iteration', 'latent_state_size',
                     'relaxed_state_encoder_temperature', 'relaxed_state_prior_temperature',
                     'kl_annealing_growth_rate', 'entropy_regularizer_decay_rate', 'prioritized_experience_replay',
                     'priority_exponent', 'importance_sampling_exponent', 'importance_sampling_exponent_growth_rate',
                     'buckets_based_priorities', 'neurons', 'hidden', 'activation'] + [
                     'encoder_temperature', 'prior_temperature', 'number_of_discrete_actions',
                     'one_output_per_action'] if fixed_parameters['action_discretizer'] else []:
            defaults[attr] = locals()[attr]

        return defaults

    def optimize_trial(trial):
        hyperparameters = suggest_hyperparameters(trial)

        for component_name in ['encoder', 'transition', 'label_transition', 'reward', 'decoder', 'discrete_policy']:
            hyperparameters[component_name + '_layers'] = hyperparameters['hidden'] * [hyperparameters['neurons']]
        q, p_t, p_l_t, p_r, p_decode, latent_policy = generate_network_components(hyperparameters, name='state')

        tf.random.set_seed(fixed_parameters['seed'])

        hyperparameters['collect_steps_per_iteration'] = min(
            hyperparameters['collect_steps_per_iteration'],
            hyperparameters['batch_size'] // 2)

        evaluation_window_size = 5
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
            evaluation_window_size=evaluation_window_size,
            evaluation_criterion=variational_mdp.EvaluationCriterion.MEAN)

        if fixed_parameters['action_discretizer']:
            q, p_t, p_l_t, p_r, p_decode, latent_policy = generate_network_components(
                hyperparameters, name='action')
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

        initial_training_steps = evaluation_window_size * int(1e4)
        training_steps_per_iteration = int(1e4)

        train_model = lambda training_steps: vae_mdp.train_from_policy(
            policy=policy,
            environment_suite=environment_suite,
            env_name=environment_name,
            labeling_function=reinforcement_learning.labeling_functions[environment_name],
            num_iterations=training_steps,
            logs=False,
            use_prioritized_replay_buffer=hyperparameters['prioritized_experience_replay'],
            global_step=global_step,
            optimizer=optimizer,
            eval_steps=0,
            save_directory=None,
            policy_evaluation_num_episodes=30,
            environment=environment,
            dataset_components=dataset_components,
            policy_evaluation_driver=policy_evaluation_driver,
            close_at_the_end=False,
            display_progressbar=fixed_parameters['display_progressbar'])

        score = train_model(initial_training_steps)

        for _ in range(initial_training_steps, num_steps, training_steps_per_iteration):
            score = train_model(training_steps_per_iteration)

            # Report intermediate objective value.
            trial.report(score, global_step.numpy())

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

        dataset_components.close_fn()

        return score

    return optimize_hyperparameters(study_name, optimize_trial, n_trials=n_trials)
