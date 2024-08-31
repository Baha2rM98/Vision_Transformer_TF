def get_model_config(pre_defined_config=True, subpatch_dim=None, layers=None, hidden_size=None, heads=None,
                     mlp_head=None):
    if pre_defined_config:
        # 'mlp_head' size is not included in the original paper and is optional, so you can change it base on your
        # specific problem.
        return {'ViT-Base': {'subpatch_dim': 16, 'layers': 12, 'hidden_size': 768, 'heads': 12, 'mlp_head': 512},
                'ViT-Large': {'subpatch_dim': 16, 'layers': 24, 'hidden_size': 1024, 'heads': 16, 'mlp_head': 1024},
                'ViT-Huge': {'subpatch_dim': 14, 'layers': 32, 'hidden_size': 1280, 'heads': 16, 'mlp_head': 2048}}

    return {'subpatch_dim': subpatch_dim, 'layers': layers, 'hidden_size': hidden_size, 'heads': heads,
            'mlp_head': mlp_head}
