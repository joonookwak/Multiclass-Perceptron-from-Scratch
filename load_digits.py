import struct
import numpy as np
from array import array

def load_images(filename):
    """
    MNIST 바이너리 파일을 읽어 이미지 리스트를 반환합니다.
    """
    with open(filename, 'rb') as file:
        # 헤더 정보 읽기: magic number, 데이터 개수, 행, 열
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
        
        # 실제 이미지 데이터 읽기
        image_data = array("B", file.read())        
        
    images = []
    for i in range(size):
        # 1차원 데이터를 28x28 2차원 배열로 변형
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        images.append(img.reshape(28, 28))
        
    return images

def load_labels(filename):
    """
    MNIST 바이너리 파일을 읽어 라벨 리스트를 반환합니다.
    """
    with open(filename, 'rb') as file:
        # 헤더 정보 읽기: magic number, 데이터 개수
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
        
        # 실제 라벨 데이터 읽기
        labels = array("B", file.read())
        
    return list(labels)