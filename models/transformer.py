from tensorflow.keras.layers import Dense, Flatten


def get_transformer_output(inputs):
    return Dense(17)(Flatten()(inputs[0]))
