"""
SRM Layer V3 - Enhanced Forensic Feature Extraction
====================================================
Optimized for detecting both GAN and Diffusion model artifacts.

Key improvements:
1. Extended filter bank (5 forensic filters instead of 3)
2. Bayar constrained convolution for learned forensic features
3. Multi-scale noise analysis
4. Improved normalization for training stability
"""

import tensorflow as tf
import numpy as np


class SRMLayerV3(tf.keras.layers.Layer):
    """
    Enhanced Forensic Layer V3 with extended filter bank.
    
    Applies 5 fixed SRM/forensic filters:
    1. KV (Ker-Vass) - Second order residuals
    2. SPAM (Subtractive Pixel Adjacency Matrix) - Edge detection  
    3. MinMax - Center-surround contrast
    4. Bayar - Constrained high-pass filter
    5. Laplacian - Second derivative for texture
    
    Output: 15 channels (5 filters × 3 RGB channels)
    """
    
    def __init__(self, truncate_max=3.0, **kwargs):
        super(SRMLayerV3, self).__init__(**kwargs)
        self.truncate_max = truncate_max
        self.num_filters = 5
        
    def build(self, input_shape):
        # ============================================
        # DEFINE ALL 5x5 FORENSIC KERNELS
        # ============================================
        
        # 1. KV Kernel (Second-order residuals) - Classic SRM
        q_kv = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0

        # 2. SPAM11 (Edge detector) - Padded to 5x5
        q_spam_3x3 = np.array([
            [-1,  2, -1],
            [ 2, -4,  2],
            [-1,  2, -1]
        ], dtype=np.float32) / 4.0
        q_spam = np.pad(q_spam_3x3, ((1, 1), (1, 1)), 'constant')

        # 3. MinMax (Center-surround) - Padded to 5x5  
        q_minmax_3x3 = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32) / 8.0
        q_minmax = np.pad(q_minmax_3x3, ((1, 1), (1, 1)), 'constant')

        # 4. Bayar Constrained Filter - High-pass with center=-1
        # This is particularly effective for GAN artifacts
        q_bayar_3x3 = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        # Normalize so sum = 0 (high-pass property)
        q_bayar_3x3 = q_bayar_3x3 - q_bayar_3x3.mean()
        q_bayar_3x3 = q_bayar_3x3 / (np.abs(q_bayar_3x3).sum() + 1e-7)
        q_bayar = np.pad(q_bayar_3x3, ((1, 1), (1, 1)), 'constant')

        # 5. Laplacian of Gaussian approximation - Padded to 5x5
        # Good for detecting diffusion model smoothing artifacts
        q_log_3x3 = np.array([
            [ 0,  1,  0],
            [ 1, -4,  1],
            [ 0,  1,  0]
        ], dtype=np.float32) / 4.0
        q_log = np.pad(q_log_3x3, ((1, 1), (1, 1)), 'constant')

        # ============================================
        # BUILD DEPTHWISE CONV KERNEL
        # ============================================
        # Stack all filters: (5, 5, 5)
        filters = np.stack([q_kv, q_spam, q_minmax, q_bayar, q_log], axis=-1)
        
        # Expand for depthwise: (5, 5, 1, 5)
        filters = np.expand_dims(filters, axis=2)
        
        # Tile for 3 input channels: (5, 5, 3, 5)
        filters = np.tile(filters, (1, 1, 3, 1))
        
        # Create depthwise conv layer
        self.conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=5,
            strides=1,
            padding='same',
            depth_multiplier=self.num_filters,
            use_bias=False,
            trainable=False,
            name='srm_depthwise_v3'
        )
        
        # Build and set weights
        self.conv.build(input_shape)
        self.conv.set_weights([filters.astype(np.float32)])
        
        super(SRMLayerV3, self).build(input_shape)

    def call(self, inputs):
        # Apply forensic filters
        x = self.conv(inputs)
        
        # Truncate strong edges to focus on subtle noise
        x = tf.clip_by_value(x, -self.truncate_max, self.truncate_max)
        
        # Return absolute energy (sign-agnostic noise detection)
        return tf.abs(x)
    
    def get_config(self):
        config = super(SRMLayerV3, self).get_config()
        config.update({'truncate_max': self.truncate_max})
        return config


class BayarConv2D(tf.keras.layers.Layer):
    """
    Learnable Bayar Constrained Convolution.
    
    Constraint: Center weight = -sum(surrounding weights)
    This ensures the filter is always high-pass (DC = 0).
    
    This layer LEARNS forensic features while maintaining
    the high-pass constraint, making it adaptive to new forgery types.
    """
    
    def __init__(self, filters=3, kernel_size=5, **kwargs):
        super(BayarConv2D, self).__init__(**kwargs)
        self.num_filters = filters
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        in_channels = input_shape[-1]
        k = self.kernel_size
        center = k // 2
        
        # Initialize weights (excluding center)
        # Shape: (k, k, in_channels, filters) but center will be computed
        self.kernel_weights = self.add_weight(
            name='bayar_weights',
            shape=(k * k - 1, in_channels, self.num_filters),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Store indices for reconstruction
        self.center_idx = center * k + center
        self.k = k
        self.in_channels = in_channels
        
        super(BayarConv2D, self).build(input_shape)
    
    def call(self, inputs):
        k = self.k
        
        # Get surrounding weights
        surrounding = self.kernel_weights  # (k*k-1, in_c, filters)
        
        # Compute center weight as negative sum of surrounding
        center_weight = -tf.reduce_sum(surrounding, axis=0, keepdims=True)  # (1, in_c, filters)
        
        # Reconstruct full kernel by inserting center weight
        # Split surrounding at center position
        before_center = surrounding[:self.center_idx]  # (center_idx, in_c, filters)
        after_center = surrounding[self.center_idx:]   # (k*k-1-center_idx, in_c, filters)
        
        # Concatenate with center in the middle
        full_kernel = tf.concat([before_center, center_weight, after_center], axis=0)
        
        # Reshape to conv kernel shape: (k, k, in_c, filters)
        full_kernel = tf.reshape(full_kernel, (k, k, self.in_channels, self.num_filters))
        
        # Apply convolution
        return tf.nn.conv2d(inputs, full_kernel, strides=[1, 1, 1, 1], padding='SAME')
    
    def get_config(self):
        config = super(BayarConv2D, self).get_config()
        config.update({
            'filters': self.num_filters,
            'kernel_size': self.kernel_size
        })
        return config


class ForensicStreamV3(tf.keras.layers.Layer):
    """
    Complete Forensic Stream combining fixed and learnable filters.
    
    Architecture:
    1. Fixed SRM filters (5 filters × 3 channels = 15 channels)
    2. Learnable Bayar filters (8 filters)
    3. Fusion and feature extraction
    
    Output: Feature vector ready for gating mechanism
    """
    
    def __init__(self, gate_dim=1280, **kwargs):
        super(ForensicStreamV3, self).__init__(**kwargs)
        self.gate_dim = gate_dim
        
    def build(self, input_shape):
        # Fixed forensic filters
        self.srm = SRMLayerV3(name='srm_fixed')
        
        # Learnable constrained filters
        self.bayar = BayarConv2D(filters=8, kernel_size=5, name='bayar_learnable')
        
        # Normalization
        self.bn_srm = tf.keras.layers.BatchNormalization(name='bn_srm')
        self.bn_bayar = tf.keras.layers.BatchNormalization(name='bn_bayar')
        
        # Feature extraction convolutions
        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', name='forensic_conv1')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='forensic_conv2')
        self.bn_feat = tf.keras.layers.BatchNormalization(name='bn_feat')
        
        # Pooling and projection
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name='forensic_gap')
        self.dense = tf.keras.layers.Dense(self.gate_dim, activation='sigmoid', name='forensic_gate')
        
        super(ForensicStreamV3, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Fixed SRM path
        srm_out = self.srm(inputs)
        srm_out = self.bn_srm(srm_out, training=training)
        
        # Learnable Bayar path
        bayar_out = self.bayar(inputs)
        bayar_out = tf.abs(bayar_out)  # Energy
        bayar_out = self.bn_bayar(bayar_out, training=training)
        
        # Concatenate forensic features
        combined = tf.concat([srm_out, bayar_out], axis=-1)  # 15 + 8 = 23 channels
        
        # Feature extraction
        x = self.conv1(combined)
        x = self.conv2(x)
        x = self.bn_feat(x, training=training)
        
        # Generate gate vector
        x = self.gap(x)
        gate = self.dense(x)
        
        return gate
    
    def get_config(self):
        config = super(ForensicStreamV3, self).get_config()
        config.update({'gate_dim': self.gate_dim})
        return config


# Backward compatibility alias
SRMLayer = SRMLayerV3