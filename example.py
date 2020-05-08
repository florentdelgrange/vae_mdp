from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, MaxPooling2D
from tensorflow.keras.utils import plot_model
import numpy as np
import variational_mdp as vae_mdp
from variational_mdp import VariationalMDPStateAbstraction

if __name__ == '__main__':
    # Example
    state_dim = (4,)
    # state_dim = (128, 128, 3,)
    action_dim = (2,)
    reward_dim = (1,)
    label_dim = (3,)

    #  input_pre_processing_network = Input(shape=state_dim)
    #  pre_processing_network = Conv2D(filters=16, kernel_size=5, activation='relu', strides=(2, 2)) \
    #      (input_pre_processing_network)
    #  pre_processing_network = Conv2D(filters=32, kernel_size=3, activation='relu', strides=(2, 2)) \
    #      (pre_processing_network)
    #  pre_processing_network = BatchNormalization()(pre_processing_network)
    #  pre_processing_network = MaxPooling2D()(pre_processing_network)
    #  pre_processing_network = Flatten()(pre_processing_network)
    #  pre_processing_network = Model(inputs=input_pre_processing_network, outputs=pre_processing_network)

    # Encoder body
    encoder_input = Input(shape=(np.prod(state_dim) * 2 + np.prod(action_dim) + np.prod(reward_dim),))
    # x = Input(shape=(np.prod(tuple(filter(lambda dim: dim is not None, pre_processing_network.output.shape))) * 2 +
    #                 np.prod(action_dim) + np.prod(reward_dim),))
    q = Dense(32, activation='relu')(encoder_input)
    q = Dense(64, activation='relu')(q)
    q = Model(inputs=encoder_input, outputs=q, name="encoder_network_body")

    # Transition network body
    transition_input = Input(shape=(256,))
    p_t = Dense(128, activation='relu')(transition_input)
    p_t = Dense(128, activation='relu')(p_t)
    p_t = Model(inputs=transition_input, outputs=p_t, name="transition_network_body")

    # Reward network body
    p_r_input = Input(shape=(384,))
    p_r = Dense(128, activation='relu')(p_r_input)
    p_r = Dense(64, activation='relu')(p_r)
    p_r = Model(inputs=p_r_input, outputs=p_r, name="reward_network_body")

    # Decoder network body
    p_decoder_input = Input(shape=(128,))
    p_decode = Dense(64, activation='relu')(p_decoder_input)
    p_decode = Dense(32, activation='relu')(p_decode)
    p_decode = Model(inputs=p_decoder_input, outputs=p_decode, name="decoder_body")

    #  p_deconv_input = Input(tuple(filter(lambda dim: dim is not None, p_decode.output.shape)))
    #  p_deconv = Dense(units=np.prod(tuple(filter(lambda dim: dim is not None, pre_processing_network.output.shape))))\
    #      (p_deconv_input)
    #  p_deconv = Reshape((32, 32, 32))(p_deconv)
    #  p_deconv = \
    #      Conv2DTranspose(filters=32, kernel_size=3, activation='relu', strides=(2, 2), padding='SAME')(p_deconv)
    #  p_deconv = \
    #      Conv2DTranspose(filters=16, kernel_size=5, activation='relu', strides=(2, 2), padding='SAME')(p_deconv)
    #  p_deconv = \
    #      Conv2DTranspose(filters=3, kernel_size=3, padding='SAME')(p_deconv)
    #  p_deconv = Model(inputs=p_deconv_input, outputs=p_deconv)

    action_processor = Sequential(name='action_processor')
    action_processor.add(Dense(32, activation='sigmoid', name="process_action"))
    action_processor.add(Dense(128, activation='sigmoid', name="process_action"))
    model = VariationalMDPStateAbstraction(state_dim, action_dim, reward_dim, label_dim, q, p_t, p_r, p_decode,
                                           action_pre_processing_network=action_processor)
    # state_pre_processing_network=pre_processing_network,
    # state_post_processing_network=p_deconv)
    plot_model(model.vae, dpi=300, expand_nested=True, show_shapes=True)
