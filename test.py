import numpy as np
from features import image_to_vector


def test_perceptron(test_images, test_labels, weights):
    """
    Evaluate a trained multiclass perceptron.

    Args:
        test_images: list of 28x28 images
        test_labels: list of integer labels in {0, ..., 9}
        weights: numpy array of shape (10, 784)

    Returns:
        error_rate: percentage of misclassified samples
    """
    num_errors = 0
    num_samples = len(test_images)

    for image, true_label in zip(test_images, test_labels):
        # TODO 1: Convert the image into a normalized 784-D vector.
        x = image_to_vector(image, normalize=True) 

        # TODO 2: Compute the score for each class and predict the label
        # with the highest score.
        scores = np.dot(weights, x)
        predicted_label = np.argmax(scores) # np.argmax(scores) 를 사용하여 가장 높은 점수를 가진 클래스의 인덱스를 예측된 레이블로 설정

        # TODO 3: Count misclassified samples.
        if predicted_label != true_label:
            num_errors += 1

    # TODO 4: Compute and return the error rate (%).
    error_rate = (num_errors / num_samples) * 100
    print(f"Test Error Rate: {error_rate:.2f}%")
    return error_rate
