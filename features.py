import numpy as np

def image_to_vector(image, normalize=True):
    """
    28x28 이미지를 784차원 피처 벡터(784 pixels)로 변환합니다.
    """
    # 1. Raster scan order로 1차원 변환 (784,)
    x = np.array(image).flatten().astype(float)
    
    # 2. Normalization (0~1 범위)
    if normalize:
        x = x / 255.0
    
    return x