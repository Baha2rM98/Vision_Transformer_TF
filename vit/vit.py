import tensorflow as tf


# Layer to extract patches from input images
class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, num_subpatch, subpatch_dim, num_channels):
        super(PatchExtractor, self).__init__()
        self.num_subpatch = num_subpatch  # Number of patches per image
        self.subpatch_dim = subpatch_dim  # Dimension of each patch
        self.num_channels = num_channels  # Number of channels in the input image

    def call(self, images):
        # Extract patches from images using TensorFlow's extract_patches function
        batch = tf.shape(images)[0]  # Batch size
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.subpatch_dim, self.subpatch_dim, 1],  # Patch size
            strides=[1, self.subpatch_dim, self.subpatch_dim, 1],  # Stride same as patch size (non-overlapping)
            rates=[1, 1, 1, 1],  # Dilation rate (no dilation)
            padding="VALID",  # No padding, valid patches only
        )
        # Reshape the patches into the desired shape [batch_size, num_patches, patch_size * num_channels]
        return tf.reshape(patches,
                          [batch, self.num_subpatch, self.subpatch_dim * self.subpatch_dim * self.num_channels])


# Layer to encode extracted patches into patch embeddings with positional embeddings and a class token
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches  # Total number of patches
        self.projection_dim = projection_dim  # Dimension of the output patch embeddings

        # Initialize the class token as a zero vector with trainable=True
        w_init = tf.zeros_initializer()
        class_token = w_init(shape=(1, projection_dim), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, trainable=True)

        # Dense layer to project patch to a higher-dimensional space
        self.projection = tf.keras.layers.Dense(units=projection_dim)

        # Learnable positional embedding layer for patches + class token
        self.position_embedding = tf.keras.layers.Embedding(input_dim=num_patches + 1, output_dim=projection_dim)

    def call(self, patch):
        batch = tf.shape(patch)[0]  # Batch size

        # Reshape and replicate the class token for the batch size
        class_token = tf.tile(self.class_token, multiples=[batch, 1])
        class_token = tf.reshape(class_token, (batch, 1, self.projection_dim))

        # Project patches to a higher-dimensional space
        patches_embed = self.projection(patch)

        # Concatenate class token at the beginning of the patch embeddings
        patches_embed = tf.concat([patches_embed, class_token], axis=1)

        # Generate positional embeddings for patches + class token
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        positions_embed = self.position_embedding(positions)

        # Add positional embeddings to patch embeddings
        return patches_embed + positions_embed


# Feed-forward layer within each transformer block
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, hidden_features, out_features, dropout_rate):
        super(FeedForward, self).__init__()
        # Two-layer MLP with GELU activation in between
        self.dense1 = tf.keras.layers.Dense(hidden_features, activation='gelu')
        self.dense2 = tf.keras.layers.Dense(out_features)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # Forward pass through MLP with dropout applied after each layer
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return self.dropout(x)


# Transformer block consisting of multi-head self-attention and feed-forward layers with residual connections
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, projection_dim, num_heads, dropout_rate):
        super(TransformerBlock, self).__init__()
        # Layer normalization for input to attention layer
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Multi-head self-attention layer
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim // num_heads,
                                                       dropout=dropout_rate)
        # Layer normalization for input to feed-forward layer
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Feed-forward network
        self.ff = FeedForward(projection_dim * 4, projection_dim, dropout_rate)

    def call(self, x):
        # Self-attention with residual connection
        x1 = self.norm1(x)
        attention_output = self.attn(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, x])

        # Feed-forward with residual connection
        x3 = self.norm2(x2)
        x3 = self.ff(x3)
        return tf.keras.layers.Add()([x3, x2])


# Transformer encoder consisting of multiple transformer blocks
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, projection_dim, num_heads, num_blocks, dropout_rate):
        super(TransformerEncoder, self).__init__()
        # Stack of transformer blocks
        self.blocks = [TransformerBlock(projection_dim, num_heads, dropout_rate) for _ in range(num_blocks)]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # Final layer normalization
        self.dropout = tf.keras.layers.Dropout(dropout_rate)  # Final dropout layer

    def call(self, x):
        # Pass through each transformer block
        for block in self.blocks:
            x = block(x)
        # Apply final normalization and dropout
        x = self.norm(x)
        return self.dropout(x)


# Optional feed-forward head for task-specific output (classification or regression)
class FeedForwardHead(tf.keras.layers.Layer):
    def __init__(self, hidden_neurons, dropout_rate, activation, num_classes):
        super(FeedForwardHead, self).__init__()
        # Multi-layer feed-forward network with Tanh activations
        self.dense1 = tf.keras.layers.Dense(hidden_neurons, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(hidden_neurons // 2, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(hidden_neurons // 4, activation='tanh')

        # Output layer depends on task (classification or regression)
        if activation == 'softmax':
            self.dense4 = tf.keras.layers.Dense(num_classes, activation=activation)
        if activation == 'linear':
            self.dense4 = tf.keras.layers.Dense(1, activation=activation)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # Forward pass through feed-forward layers with dropout
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)
        x = self.dropout(x)
        return self.dense4(x)


# Vision Transformer model class
class VisionTransformer:
    def __init__(self, input_shape, model_variant, dropout_rate=0.1, task='classification'):
        self.input_shape = input_shape  # Shape of input images
        self.model_variant = model_variant  # Model configuration (e.g., hidden size, heads, layers, etc.)
        self.dropout_rate = dropout_rate  # Dropout rate for all layers
        self.num_channels = input_shape[-1]  # Number of image channels (e.g., 3 for RGB)
        self.subpatch_dim = self.model_variant['subpatch_dim']  # Dimension of each patch
        self.num_subpatch = int((input_shape[1] / self.subpatch_dim) ** 2)  # Number of patches per image
        self.projection_dim = self.model_variant['hidden_size']  # Dimensionality of patch embedding
        self.num_heads = self.model_variant['heads']  # Number of attention heads
        self.num_blocks = self.model_variant['layers']  # Number of transformer blocks
        self.mlp_head = self.model_variant['mlp_head']  # Size of MLP head
        self.task = task  # Task type ('classification' or 'regression')
        self.activation = None  # Activation function for output layer

    def __build_encoder(self):
        # Input layer
        input_layer = tf.keras.layers.Input(shape=self.input_shape, name='image_input')

        # Extract patches from input images
        patches = PatchExtractor(self.num_subpatch, self.subpatch_dim, self.num_channels)(input_layer)

        # Encode patches with positional embeddings and class token
        patches_embed = PatchEncoder(self.num_subpatch, self.projection_dim)(patches)

        # Pass encoded patches through the transformer encoder
        representation = TransformerEncoder(self.projection_dim, self.num_heads, self.num_blocks, self.dropout_rate)(
            patches_embed)

        # Output logic based on task type
        if self.task == 'classification':
            # Classification output uses the [CLS] token
            self.activation = 'softmax'
            output = representation[:, 0, :]
            return tf.keras.models.Model(inputs=input_layer, outputs=output, name='VisionTransformer')

        if self.task == 'regression':
            # Regression output uses Global Average Pooling
            self.activation = 'linear'
            output = tf.keras.layers.GlobalAveragePooling1D()(representation)
            return tf.keras.models.Model(inputs=input_layer, outputs=output, name='VisionTransformer')

        raise ValueError(f"The task {self.task} is not supported.")

    def build_model(self, include_top=True, num_classes=1000):
        # Build the base transformer encoder
        base_model = self.__build_encoder()

        # If the top head is not included, return base model only
        if not include_top:
            return base_model

        # Add feed-forward head for final classification or regression
        fc = FeedForwardHead(self.mlp_head, self.dropout_rate, self.activation, num_classes)(base_model.output)
        return tf.keras.models.Model(inputs=base_model.input, outputs=fc,
                                     name=f"VisionTransformer_{self.task.capitalize()}")
