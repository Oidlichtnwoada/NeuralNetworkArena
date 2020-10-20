from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.models.reformer.reformer import EncoderBlock, EncoderDecoderBlock


def ContinuousReformer(input_vocab_size,
             output_vocab_size=None,
             d_model=512,
             d_ff=2048,
             n_encoder_layers=6,
             n_decoder_layers=6,
             n_heads=8,
             dropout=0.1,
             max_len=2048,
             ff_activation=tl.Relu,
             ff_dropout=None,
             mode='train'):
    """Reversible transformer encoder-decoder model.

    This model expects an input pair: target, source.

    At the moment, this model supports dot-product attention only. For the
    attention types in the Reformer paper, see ReformerLM.

    Args:
      input_vocab_size: int: vocab size of the source.
      output_vocab_size: int (optional): vocab size of the target. If None, the
        source and target are assumed to have the same vocab.
      d_model: int:  depth of embedding
      d_ff: int: depth of feed-forward layer
      n_encoder_layers: int: number of encoder layers
      n_decoder_layers: int: number of decoder layers
      n_heads: int: number of attention heads
      dropout: float: dropout rate (how much to drop out)
      max_len: int: maximum symbol length for positional encoding
      ff_activation: the non-linearity in feed-forward layer
      ff_dropout: float: (optional) separate dropout rate at feed-forward
        nonlinearity. This is called relu_dropout in T2T.
      mode: str: 'train' or 'eval'

    Returns:
      A Reformer model as a layer that maps from a target, source pair to
      activations over a vocab set.
    """

    def PositionalEncoder(vocab_size, mode):  # tokens --> vectors
        # TODO(kitaev): axial positional encoding is better for very long sequences.
        positional_encoding = tl.PositionalEncoding(
            max_len=max_len, dropout=dropout, mode=mode)
        return [
            tl.Embedding(vocab_size, d_model),
            tl.Dropout(rate=dropout, shared_axes=[-2], mode=mode),
            positional_encoding,
        ]

    # Mode 'predict' means that the decoder should be run one token at a time.
    # The encoder only ever runs over full sequences, which is why it's switched
    # to 'eval' mode instead.
    in_encoder = PositionalEncoder(
        input_vocab_size, mode='eval' if mode == 'predict' else mode)
    if output_vocab_size is None:
        output_vocab_size = input_vocab_size
    out_encoder = PositionalEncoder(output_vocab_size, mode)

    # pylint: disable=g-complex-comprehension
    encoder_blocks = [
        EncoderBlock(
            d_model, d_ff, n_heads, tl.SelfAttention, dropout, ff_activation,
            ff_dropout, mode=mode)
        for _ in range(n_encoder_layers)]
    # pylint: enable=g-complex-comprehension

    encoder = tl.Serial([
        in_encoder,
        tl.Dup(),
        tl.ReversibleSerial(encoder_blocks),
        tl.Fn('XYAvg', lambda x, y: (x + y) / 2.0),
        tl.LayerNorm(),
    ])
    if mode == 'predict':
        encoder = tl.Cache(encoder)

    encoder_decoder_blocks = [
        EncoderDecoderBlock(
            d_model, d_ff, n_heads, dropout, ff_activation, ff_dropout, mode)
        for _ in range(n_decoder_layers)]

    # Assemble and return the model.
    return tl.Serial(
        # Input: encoder_side_tokens, decoder_side_tokens
        # Copy decoder tokens for use in loss.
        tl.Select([0, 1, 1]),  # tok_e tok_d tok_d
        tl.Branch([], [tl.PaddingMask(),
                       tl.Fn('Squeeze',
                             lambda x: jnp.squeeze(x, (1, 2)), n_out=1)]),
        #                                     # tok_e mask  tok_d .....

        # Encode.
        encoder,  # vec_e  mask tok_d .....

        # Decode.
        tl.Select([2, 0, 1]),  # tok_d vec_e mask .....
        tl.ShiftRight(mode=mode),  # tok_d vec_e mask .....
        out_encoder,  # vec_d vec_e mask .....
        tl.Dup(),  # vec_d1 vec_d2 vec_e mask .....
        tl.ReversibleSerial(encoder_decoder_blocks),
        tl.Fn('XYAvg',
              lambda x, y: (x + y) / 2.0),  # vec_d vec_e mask .....
        tl.LayerNorm(),  # vec_d vec_e mask .....

        # Map to output vocab.
        tl.Select([0], n_in=3),  # vec_d .....
        tl.Dense(output_vocab_size),  # vec_d .....
        tl.LogSoftmax(),  # vec_d .....
    )
