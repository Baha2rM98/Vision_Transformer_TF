import vit.config as model_config
from vit.vit import VisionTransformer

# You can choose "ViT-Base", "ViT-Large" or "ViT-Huge" base on paper, or you define your custom variant.
model_variant = model_config.get_model_config()['ViT-Base']

# the resolution of image in original paper is 224x224 with 3 channels (RGB).
vit = VisionTransformer(input_shape=(224, 224, 3), model_variant=model_variant).build_model()

vit.summary()
