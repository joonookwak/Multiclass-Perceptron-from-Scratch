import matplotlib.pyplot as plt

def show_digit(image):
    """
    28x28 numpy 배열 이미지를 시각화합니다.
    """
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()