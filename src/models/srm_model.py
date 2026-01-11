import tensorflow as tf
import numpy as np

class SRMLayer(tf.keras.layers.Layer):
    """
    Fixed Forensic Layer using Depthwise Convolution.
    1. Applies SRM filters to R, G, B separately (No summation).
    2. Truncates large values (Edges) to focus on Texture/Noise.
    3. Returns absolute residuals (Energy).
    """
    def __init__(self, **kwargs):
        super(SRMLayer, self).__init__(**kwargs)
        # Depthwise Conv ensures R, G, B residuals are kept separate
        # We have 3 kernels (KV, Spam, MinMax)
        # Depth multiplier = 3 means: Input R -> R_KV, R_Spam, R_MinMax
        self.conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=5,
            strides=1,
            padding='same',
            depth_multiplier=3,  # 3 Filters per input channel
            use_bias=False,
            trainable=False,  # Locked weights
            name='srm_depthwise'
        )
        self.truncate_max = 3.0  # Standard SRM truncation threshold

    def build(self, input_shape):
        # 1. Define Kernels (5x5)
        # KV Kernel (Residuals)
        q_kv = np.array([[-1, 2, -2, 2, -1],
                         [2, -6, 8, -6, 2],
                         [-2, 8, -12, 8, -2],
                         [2, -6, 8, -6, 2],
                         [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0

        # Spam11 (Edge) - Padded to 5x5
        q_spam = np.array([[-1, 2, -1],
                           [2, -4, 2],
                           [-1, 2, -1]], dtype=np.float32)
        q_spam = np.pad(q_spam, ((1, 1), (1, 1)), 'constant') / 4.0

        # MinMax (Center) - Padded to 5x5
        q_minmax = np.array([[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]], dtype=np.float32)
        q_minmax = np.pad(q_minmax, ((1, 1), (1, 1)), 'constant') / 8.0

        # 2. Stack for Depthwise Format: (Kernel_H, Kernel_W, In_Channels, Depth_Multiplier)
        # We need shape (5, 5, 3, 3)
        # Channel 1 (Red) gets [KV, Spam, MinMax]
        # Channel 2 (Green) gets [KV, Spam, MinMax]...
        
        filters = np.stack([q_kv, q_spam, q_minmax], axis=-1) # Shape (5, 5, 3)
        
        # In DepthwiseConv2D, weights are (K_H, K_W, In_C, Depth_Mult)
        # We replicate the filters for each input channel
        filters = np.expand_dims(filters, axis=2) # Shape (5, 5, 1, 3)
        filters = np.tile(filters, (1, 1, 3, 1))  # Shape (5, 5, 3, 3)

        # 3. Set Weights
        self.conv.build(input_shape)
        self.conv.set_weights([filters])
        super(SRMLayer, self).build(input_shape)

    def call(self, inputs):
        # 1. Convolve (Get Residuals)
        x = self.conv(inputs)
        
        # 2. Truncate (Optional but recommended for SRM stability)
        # Clips "strong edges" so they don't drown out "weak texture noise"
        x = tf.clip_by_value(x, -self.truncate_max, self.truncate_max)
        
        # 3. Absolute Value (Energy)
        # We want to detect presence of noise, regardless of sign
        return tf.abs(x)