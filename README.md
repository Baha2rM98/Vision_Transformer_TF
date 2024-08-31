# Vision Transformer (ViT) Project

This project implements the Vision Transformer (ViT) model as described in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). The Vision Transformer model applies a Transformer architecture to image classification tasks, achieving state-of-the-art results on various benchmarks.

![Vision Transformer](https://raw.githubusercontent.com/emla2805/vision-transformer/master/vit.png)

## Usage

To use the Vision Transformer model, you need to configure the model variant and input shape. Below is an example of how to set up and use the model:

1. **Configure the model variant**:
    ```python
    import vit.config as model_config
    from vit.vit import VisionTransformer

    # Choose "ViT-Base", "ViT-Large" or "ViT-Huge" based on the paper, or define your custom variant.
    model_variant = model_config.get_model_config()['ViT-Base']
    ```

2. **Set the input shape**:
    ```python
    # The resolution of the image in the original paper is 224x224 with 3 channels (RGB).
    vit = VisionTransformer(input_shape=(224, 224, 3), model_variant=model_variant).build_model()
    ```

3. **Build and summarize the model**:
    ```python
    vit.summary()
    ```

## Configuration

The Vision Transformer model can be configured using different variants as described in the paper. The available variants are:

- `ViT-Base`
- `ViT-Large`
- `ViT-Huge`

You can select a variant by modifying the `model_variant` variable in the code:

```python
model_variant = model_config.get_model_config()['ViT-Base']
```

You can also define your custom variant by modifying the configuration file located at [`vit/config.py`](https://github.com/Baha2rM98/Vision_Transformer_TF/blob/master/vit/config.py).

## Model Summary

To get a summary of the model, you can use the `summary` method provided by the Keras model:

```python
vit.summary()
```

This will print a detailed summary of the model architecture, including the number of parameters and the shape of each layer.
