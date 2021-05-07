import functools
import importlib
import os

import tensorflow as tf
import tf_agents
from absl import app
from absl import flags
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tf_agents.environments import tf_py_environment
from tf_agents.policies import policy_saver
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step

import policies
import reinforcement_learning
import variational_action_discretizer
import variational_mdp
from util.io import dataset_generator

flags.DEFINE_string(
    "dataset_path",
    help="Path of the directory containing the dataset files in hdf5 format.",
    default='')
flags.DEFINE_integer("batch_size", default=128, help="Batch size.")
flags.DEFINE_integer(
    "mixture_components",
    default=1,
    help="Number of gaussian mixture components used to model the reconstruction distributions.")
flags.DEFINE_integer(
    "action_mixture_components",
    default=0,
    help="Number of gaussian mixture components used to model the action reconstruction distribution (optional). If not"
         "set, all mixture distributions take the same value obtained via --mixture_components.")
flags.DEFINE_bool(
    "full_covariance",
    default=False,
    help="If set, the states and rewards reconstruction distributions will use a full covariance matrix instead of"
         "a diagonal matrix."
)
flags.DEFINE_string(
    "activation",
    # default="leaky_relu",
    default="relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer("latent_size", default=17, help='Number of bits used for the discrete latent state space.')
flags.DEFINE_float(
    "max_state_decoder_variance",
    default="0.",
    help='Maximum variance allowed for the state decoder.'
)
flags.DEFINE_float(
    "encoder_temperature",
    default=-1.,
    help="Temperature of the relaxation of the discrete encoder distribution."
)
flags.DEFINE_float(
    "prior_temperature",
    default=-1.,
    help="Temperature of relaxation of the discrete prior distribution over latent variables."
)
flags.DEFINE_float(
    "relaxed_state_encoder_temperature",
    default=-1.,
    help="Temperature of the binary concrete relaxation encoder distribution over latent states."
)
flags.DEFINE_float(
    "relaxed_state_prior_temperature",
    default=-1.,
    help="Temperature of the binary concrete relaxation prior distribution over latent states."
)
flags.DEFINE_bool(
    "latent_policy",
    default=False,
    help="If set, VAEs for state discretization will learn a abstraction of the input policy conditioned on"
         "latent states."
         "Remark 1: only works for environment with discrete actions."
         "Remark 2: Action discretizer VAEs always learn a latent policy."
)
flags.DEFINE_float(
    "encoder_temperature_decay_rate",
    default=1e-6,
    help="Decay rate used to anneal the temperature of the encoder distribution over latent states."
)
flags.DEFINE_float(
    "prior_temperature_decay_rate",
    default=2e-6,
    help="Decay rate used to anneal the temperature of the prior distribution over latent states."
)
flags.DEFINE_float(
    "entropy_regularizer_scale_factor",
    default=0.,
    help="Entropy regularizer scale factor."
)
flags.DEFINE_float(
    "entropy_regularizer_decay_rate",
    default=0.,
    help="Decay rate of the scale factor of the entropy regularizer."
)
flags.DEFINE_float(
    "entropy_regularizer_scale_factor_min_value",
    default=0.,
    help="Minimum value that can take the scale factor of the entropy regularizer."
)
flags.DEFINE_float(
    "marginal_entropy_regularizer_ratio",
    default=0.,
    lower_bound=0.,
    upper_bound=0.5,
    help="Indicates the ratio of the entropy regularizer focusing on enforcing a high marginal state encoder entropy"
         "(experimental)."
)
flags.DEFINE_float(
    "kl_annealing_scale_factor",
    default=1.,
    help='Scale factor of the KL terms of the ELBO.'
)
flags.DEFINE_float(
    "kl_annealing_growth_rate",
    default=0.,
    help='Annealing growth rate of the ELBO KL terms scale factor.'
)
flags.DEFINE_integer(
    "start_annealing_step",
    default=int(1e4),
    help="Step from which temperatures and scale factors start to be annealed."
)
flags.DEFINE_integer(
    "max_steps",
    default=int(1e6),
    help="Maximum number of training steps."
)
flags.DEFINE_string(
    "save_dir",
    default=".",
    help="Checkpoints and models save directory."
)
flags.DEFINE_bool(
    "display_progressbar",
    default=False,
    help="Display progressbar."
)
flags.DEFINE_bool(
    "action_discretizer",
    default=False,
    help="Discretize the action space via a VAE already trained. Require the flag --load_vae to be set."
)
flags.DEFINE_bool(
    "one_output_per_action",
    default=False,
    help="Set whether discrete action networks use one output per action or use the latent action as input."
)
flags.DEFINE_boolean(
    "do_not_eval",
    default=False,
    help="Set this flag to not perform an evaluation of the ELBO (using discrete latent variables) during training."
)
flags.DEFINE_bool(
    "full_vae_optimization",
    default=False,
    help='Set whether the ELBO is optimized over the whole VAE or if the optimization is only focused on the'
         'state or action discretizer part of the VAE.'
)
flags.DEFINE_bool(
    'relaxed_state_encoding',
    default=True,
    help='Use a relaxed encoding of states to optimize the action discretizer part of the VAE.'
)
flags.DEFINE_integer(
    "number_of_discrete_actions",
    default=16,
    help='Number of discrete actions per latent state to learn.'
)
flags.DEFINE_string(
    "load_vae",
    default='',
    help='Path of a VAE model already trained to load (saved via the tf.saved_model function).'
)
flags.DEFINE_multi_integer(
    "encoder_layers",
    default=[256, 256],
    help='Number of units to use for each layer of the encoder.'
)
flags.DEFINE_multi_integer(
    "decoder_layers",
    default=[256, 256],
    help='Number of units to use for each layer of the decoder.'
)
flags.DEFINE_multi_integer(
    "transition_layers",
    default=[256, 256],
    help='Number of units to use for each layer of the transition network.'
)
flags.DEFINE_multi_integer(
    "label_transition_layers",
    default=[256, 256],
    help='Number of units to use for each layer of the label transition network.'
)
flags.DEFINE_multi_integer(
    "reward_layers",
    default=[256, 256],
    help='Number of units to use for each layer of the reward network.'
)
flags.DEFINE_multi_integer(
    "discrete_policy_layers",
    default=[256, 256],
    help="Number of units to use for each layer of the simplified policy network."
)
flags.DEFINE_string(
    "policy_path",
    default='',
    help="Path of a policy in tf.saved_model format."
)
flags.DEFINE_string(
    "environment",
    default='',
    help="Name of the agent's environment."
)
flags.DEFINE_string(
    "env_suite",
    default='suite_gym',
    help='Name of the tf_agents environment suite.'
)
flags.DEFINE_integer(
    "parallel_env",
    default=1,
    help='Number of parallel environments to be used during training.'
)
flags.DEFINE_float(
    'state_scaling',
    default=1.,
    help='Scaler for the input states of the environment.'
)
flags.DEFINE_integer(
    'annealing_period',
    default=1,
    help='annealing period'
)
flags.DEFINE_bool(
    'aggressive_training',
    default=False,
    help='Set whether to perform aggressive inference optimizations.'
)
flags.DEFINE_integer(
    'initial_collect_steps',
    default=int(1e4),
    help='Number of frames to be collected in the replay buffer before starting the training.'
)
flags.DEFINE_float(
    'seed', help='set seed', default=42
)
flags.DEFINE_bool(
    'logs',
    default=True,
    help="Enable logging training metrics to the logs directory."
)
flags.DEFINE_bool(
    'checkpoint',
    default=True,
    help='Enable to save/load checkpoints to/from the save directory.'
)
flags.DEFINE_float(
    'epsilon_greedy',
    default=0.,
    help='Epsilon value used for training the model via epsilon-greedy with the input policy.'
)
flags.DEFINE_bool(
    'decompose_training',
    default=False,
    help='Decompose the VAE training in two phases: 1) state space abstraction, 2) action space + policy abstraction.'
)
flags.DEFINE_bool(
    'prioritized_experience_replay',
    default=False,
    help='Use a prioritized experience replay buffer'
)
flags.DEFINE_float(
    'priority_exponent',
    default=.6,
    help='Exponent parameter for the priority experience replay'
)
flags.DEFINE_float(
    'importance_sampling_exponent',
    default=0.4,
    help='Exponent parameter of the importance sampling weights used with the prioritized experience replay buffer'
)
flags.DEFINE_float(
    'importance_sampling_exponent_growth_rate',
    default=1e-5,
    help='Growth rate used for annealing the weighted importance sampling exponent parameter when using a prioritized'
         'experience replay buffer.'
)
flags.DEFINE_bool(
    'buckets_based_priority',
    default=True,
    help='If set, the priority of the replay buffer use a bucket based priority scheme (where each bucket corresponds'
         'to a discrete latent state). If not, the loss is used '
)
FLAGS = flags.FLAGS


def main(argv):
    del argv
    params = FLAGS.flag_values_dict()

    tf.random.set_seed(params['seed'])

    def check_missing_argument(name: str):
        if params[name] == '':
            raise RuntimeError('Missing argument: --{}'.format(name))

    if params['dataset_path'] == '':
        for param in ('policy_path', 'environment'):
            check_missing_argument(param)

    relaxed_state_encoder_temperature = params['relaxed_state_encoder_temperature']
    relaxed_state_prior_temperature = params['relaxed_state_prior_temperature']
    if params['encoder_temperature'] < 0.:
        if params['action_discretizer']:
            params['encoder_temperature'] = 1. / (params['number_of_discrete_actions'] - 1)
        else:
            params['encoder_temperature'] = 0.99
    if params['prior_temperature'] < 0.:
        if params['action_discretizer']:
            params['prior_temperature'] = params['encoder_temperature'] / 1.5
        else:
            params['prior_temperature'] = 0.95
    if relaxed_state_encoder_temperature < 0:
        relaxed_state_encoder_temperature = params['encoder_temperature']
    if relaxed_state_prior_temperature < 0:
        relaxed_state_prior_temperature = params['prior_temperature']

    environment_name = params['environment']

    batch_size = params['batch_size']
    mixture_components = params['mixture_components']
    latent_state_size = params['latent_size']  # depends on the number of bits reserved for labels

    if params['load_vae'] != '':
        name_list = params['load_vae'].split(os.path.sep)
        if 'models' in name_list and name_list.index('models') < len(name_list) - 1:
            base_model_name = os.path.join(*name_list[name_list.index('models') + 1:])
        else:
            base_model_name = os.path.split(params['load_vae'])[-1]

        if params['load_vae'][-1] == os.path.sep:
            params['load_vae'] = params['load_vae'][:-1]
    else:
        base_model_name = ''

    if params['policy_path'] != '' and params['policy_path'][-1] == os.path.sep:
        params['policy_path'] = params['policy_path'][:-1]

    vae_name = ''
    if not params['action_discretizer'] or params['full_vae_optimization'] or params['decompose_training']:
        vae_name = 'vae_LS{}_MC{}_ER{}-decay={:g}-min={:g}_KLA{}-growth={:g}_TD{:.2f}-{:.2f}_{}-{}_seed={:d}'.format(
            latent_state_size,
            mixture_components,
            params['entropy_regularizer_scale_factor'],
            params['entropy_regularizer_decay_rate'],
            params['entropy_regularizer_scale_factor_min_value'],
            params['kl_annealing_scale_factor'],
            params['kl_annealing_growth_rate'],
            relaxed_state_encoder_temperature,
            relaxed_state_prior_temperature,
            params['encoder_temperature_decay_rate'],
            params['prior_temperature_decay_rate'],
            int(params['seed']))
    if params['action_discretizer']:
        if vae_name != '':
            base_model_name = vae_name
        vae_name = os.path.join(
            base_model_name,
            os.path.split(params['policy_path'])[-1],
            'action_discretizer',
            'LA{}_MC{}_ER{}-decay={:g}-min={:g}_KLA{}-growth={:g}_TD{:.2f}-{:.2f}_{}-{}'.format(
                params['number_of_discrete_actions'],
                mixture_components,
                params['entropy_regularizer_scale_factor'],
                params['entropy_regularizer_decay_rate'],
                params['entropy_regularizer_scale_factor_min_value'],
                params['kl_annealing_scale_factor'],
                params['kl_annealing_growth_rate'],
                params['encoder_temperature'],
                params['prior_temperature'],
                params['encoder_temperature_decay_rate'],
                params['prior_temperature_decay_rate']
            )
        )
    if params['prioritized_experience_replay']:
        vae_name += '_PER-priority_exponent={:g}-WIS_exponent={:g}-WIS_growth_rate={:g}'.format(
            params['priority_exponent'],
            params['importance_sampling_exponent'],
            params['importance_sampling_exponent_growth_rate'])
        if params['buckets_based_priority']:
            vae_name += '_bucket_based_priorities'
        else:
            vae_name += 'loss_based_priorities'
    if params['max_state_decoder_variance'] > 0:
        vae_name += '_max_state_decoder_variance={:g}'.format(params['max_state_decoder_variance'])
    if params['epsilon_greedy'] > 0:
        vae_name += '_epsilon_greedy={:g}'.format(params['epsilon_greedy'])
    if params['marginal_entropy_regularizer_ratio'] > 0:
        vae_name += '_marginal_state_entropy_ratio={:g}'.format(params['marginal_entropy_regularizer_ratio'])
    if params['state_scaling'] != 1.:
        vae_name += '_state_scaling={:g}'.format(params['state_scaling'])

    additional_parameters = [
        'one_output_per_action',
        'full_vae_optimization',
        'relaxed_state_encoding',
        'full_covariance',
        'latent_policy',
        'decompose_training',
    ]
    nb_additional_params = sum(
        map(lambda x: params[x], additional_parameters))
    if nb_additional_params > 0:
        vae_name += ('_params={}' + '-{}' * (nb_additional_params - 1)).format(
            *filter(lambda x: params[x], additional_parameters))

    activation = getattr(tf.nn, params["activation"])

    def generate_network_components(name=''):

        network_components = []
        for component_name in ['encoder', 'transition', 'label_transition', 'reward', 'decoder', 'discrete_policy']:
            x = Sequential(name="{}_{}_network_body".format(name, component_name))
            for i, units in enumerate(params[component_name + '_layers']):
                x.add(Dense(units, activation=activation, name="{}_{}_{}".format(name, component_name, i)))
            network_components.append(x)

        return network_components

    if params['env_suite'] != '':
        try:
            environment_suite = importlib.import_module('tf_agents.environments.' + params['env_suite'])
        except BaseException as err:
            serr = str(err)
            print("An error occurred when loading the module '" + params['env_suite'] + "': " + serr)
    else:
        environment_suite = None

    environment = tf_py_environment.TFPyEnvironment(
        tf_agents.environments.parallel_py_environment.ParallelPyEnvironment(
            [lambda: environment_suite.load(environment_name)]))

    state_shape, action_shape, reward_shape, label_shape = (
        shape if shape != () else (1,) for shape in (
        environment.observation_spec().shape,
        environment.action_spec().shape,
        environment.time_step_spec().reward.shape,
        tuple(reinforcement_learning.labeling_functions[environment_name](
            environment.reset().observation).shape[1:])
    )
    )

    time_step_spec = tensor_spec.from_spec(environment.time_step_spec())
    action_spec = tensor_spec.from_spec(environment.action_spec())
    if params['latent_policy']:
        # one hot encoding
        action_shape = (environment.action_spec().maximum + 1,)

    environment.close()
    del environment

    def build_vae_model():
        if params['load_vae'] == '':
            q, p_t, p_l_t, p_r, p_decode, latent_policy = generate_network_components(name='state')
            return variational_mdp.VariationalMarkovDecisionProcess(
                state_shape=state_shape, action_shape=action_shape, reward_shape=reward_shape, label_shape=label_shape,
                encoder_network=q, transition_network=p_t, label_transition_network=p_l_t,
                reward_network=p_r, decoder_network=p_decode,
                latent_policy_network=(latent_policy if params['latent_policy'] else None),
                latent_state_size=latent_state_size,
                mixture_components=mixture_components,
                encoder_temperature=relaxed_state_encoder_temperature,
                prior_temperature=relaxed_state_prior_temperature,
                encoder_temperature_decay_rate=params['encoder_temperature_decay_rate'],
                prior_temperature_decay_rate=params['prior_temperature_decay_rate'],
                entropy_regularizer_scale_factor=params['entropy_regularizer_scale_factor'],
                entropy_regularizer_decay_rate=params['entropy_regularizer_decay_rate'],
                entropy_regularizer_scale_factor_min_value=params['entropy_regularizer_scale_factor_min_value'],
                marginal_entropy_regularizer_ratio=params['marginal_entropy_regularizer_ratio'],
                kl_scale_factor=params['kl_annealing_scale_factor'],
                kl_annealing_growth_rate=params['kl_annealing_growth_rate'],
                multivariate_normal_full_covariance=params['full_covariance'],
                max_decoder_variance=(
                    None if params['max_state_decoder_variance'] == 0. else params['max_state_decoder_variance']
                ),
                full_optimization=not params['decompose_training'] and params['latent_policy'],
                importance_sampling_exponent=params['importance_sampling_exponent'],
                importance_sampling_exponent_growth_rate=params['importance_sampling_exponent_growth_rate']
            )
        else:
            vae = variational_mdp.load(params['load_vae'])
            vae.encoder_temperature = relaxed_state_encoder_temperature
            vae.prior_temperature = relaxed_state_prior_temperature
            return vae

    def build_action_discretizer_vae_model(vae_mdp_model, full_optimization=True):
        if params['full_vae_optimization'] and params['load_vae'] != '':
            vae = variational_action_discretizer.load(params['load_vae'], full_optimization=True)
        else:
            q, p_t, p_l_t, p_r, p_decode, latent_policy = generate_network_components(name='action')
            vae = variational_action_discretizer.VariationalActionDiscretizer(
                vae_mdp=vae_mdp_model,
                number_of_discrete_actions=params['number_of_discrete_actions'],
                action_encoder_network=q, transition_network=p_t, action_label_transition_network=p_l_t,
                reward_network=p_r, action_decoder_network=p_decode,
                latent_policy_network=latent_policy,
                encoder_temperature=params['encoder_temperature'],
                prior_temperature=params['prior_temperature'],
                encoder_temperature_decay_rate=params['encoder_temperature_decay_rate'],
                prior_temperature_decay_rate=params['prior_temperature_decay_rate'],
                one_output_per_action=params['one_output_per_action'],
                relaxed_state_encoding=params['relaxed_state_encoding'],
                full_optimization=full_optimization,
                reconstruction_mixture_components=(
                    mixture_components if params['action_mixture_components'] == 0
                    else params['action_mixture_components']
                ),
            )
            vae.kl_scale_factor = params['kl_annealing_scale_factor']
            vae.kl_growth_rate = params['kl_annealing_growth_rate']
            vae.entropy_regularizer_scale_factor = params['entropy_regularizer_scale_factor']
            vae.entropy_regularizer_decay_rate = params['entropy_regularizer_decay_rate']
        return vae

    models = [build_vae_model()]
    if params['action_discretizer']:
        if not params['decompose_training']:
            models[0] = build_action_discretizer_vae_model(models[0], full_optimization=params['full_vae_optimization'])
        else:
            models.append(build_action_discretizer_vae_model(models[0], full_optimization=False))
    else:
        if params['decompose_training']:
            models.append(models[0])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    step = tf.Variable(0, trainable=False, dtype=tf.int64)

    for phase, vae_mdp_model in enumerate(models):
        checkpoint_directory = os.path.join(
            params['save_dir'], 'saves', environment_name, 'training_checkpoints', vae_name)
        if params['checkpoint']:
            print("checkpoint path:", checkpoint_directory)
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=vae_mdp_model, step=step)
            manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_directory, max_to_keep=1)
        else:
            checkpoint = manager = None

        if phase == 1 and not params['action_discretizer'] and params['latent_policy']:
            vae_mdp_model.latent_policy_training_phase = True

        #  if base_model_name != '':
        #      vae_name_list = vae_name.split(os.path.sep)
        #      vae_name_list[0] = '_'.join(base_model_name.split(os.path.sep))
        #      vae_name = os.path.join(*vae_name_list)

        policy = policies.SavedTFPolicy(params['policy_path'], time_step_spec, action_spec)

        vae_mdp_model.train_from_policy(policy=policy,
                                        environment_suite=environment_suite,
                                        env_name=environment_name,
                                        labeling_function=reinforcement_learning.labeling_functions[environment_name],
                                        epsilon_greedy=params['epsilon_greedy'] if phase == 0 else 0.,
                                        batch_size=batch_size, optimizer=optimizer, checkpoint=checkpoint,
                                        manager=manager, log_name=vae_name,
                                        start_annealing_step=(
                                            params['start_annealing_step'] + params['max_steps'] // 2
                                            if phase == 1 and params['action_discretizer'] else
                                            params['start_annealing_step']),
                                        reset_kl_scale_factor=(
                                            params['kl_annealing_scale_factor'] if phase == 1 and
                                                                                   (params['action_discretizer'] or
                                                                                    params['latent_policy']) else None),
                                        reset_entropy_regularizer=(
                                            params['entropy_regularizer_scale_factor'] if phase == 1 and
                                                                                          (params[
                                                                                               'action_discretizer'] or
                                                                                           params[
                                                                                               'latent_policy']) else None),
                                        logs=params['logs'],
                                        num_iterations=(
                                            params['max_steps'] if not params['decompose_training'] or phase == 1
                                            else params['max_steps'] // 2),
                                        display_progressbar=params['display_progressbar'],
                                        save_directory=params['save_dir'],
                                        parallel_environments=params['parallel_env'] > 1,
                                        num_parallel_environments=params['parallel_env'],
                                        eval_steps=int(1e3) if not params['do_not_eval'] else 0,
                                        policy_evaluation_num_episodes=(
                                            0 if not (params['action_discretizer'] or params['latent_policy'])
                                                 or (phase == 0 and len(models) > 1) else 30),
                                        annealing_period=params['annealing_period'],
                                        aggressive_training=params['aggressive_training'],
                                        initial_collect_steps=params['initial_collect_steps'],
                                        discrete_action_space=(
                                                not params['action_discretizer'] and params['latent_policy']),
                                        use_prioritized_replay_buffer=params['prioritized_experience_replay'],
                                        priority_exponent=params['priority_exponent'],
                                        buckets_based_priorities=params['buckets_based_priority'])

    return 0


if __name__ == '__main__':
    tf_agents.system.multiprocessing.handle_main(functools.partial(app.run, main))
    # app.run(main)
