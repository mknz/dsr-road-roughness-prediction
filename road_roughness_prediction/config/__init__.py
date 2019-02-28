'''Configs'''


class Config:
    '''Config class'''

    OUTPUT_SIZE = 256
    TRANSFORMATION = 'BASIC_TRANSFORM'
    GAUSSIAN_NOISE_SCALE = (0.01 * 255, 0.15 * 255.)
    BLUR_LIMIT = 4
    IMAGENET_PARAMS = dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    NORMALIZE_PARAMS = IMAGENET_PARAMS


class EvalConfig(Config):
    '''Config for evaluation'''
    TRANSFORMATION = 'BASIC_EVAL_TRANSFORM'
