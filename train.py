import numpy as np
from features import image_to_vector


def train_perceptron(train_images, train_labels, epochs=5, lr=1.0):
    """
    Train a multiclass perceptron using the One-vs-All strategy.

    Args:
        train_images: list of 28x28 images
        train_labels: list of integer labels in {0, ..., 9}
        epochs: number of passes over the training set
        lr: learning rate

    Returns:
        weights: numpy array of shape (10, 784)
    """
    num_classes = 10
    input_dim = 28 * 28

    # TODO 1: Initialize the weight matrix for all 10 perceptrons.
    weights = np.zeros((num_classes, input_dim))

    for epoch in range(epochs):
        num_updates = 0

        for image, true_label in zip(train_images, train_labels):
            # TODO 2: Convert the image into a normalized 784-D vector.
            x = image_to_vector(image, normalize=True)

            # TODO 3: For each class c, compute the binary target (+1 or -1),
            # compute the perceptron score, and update the weights if misclassified.
            # Use the update rules from class:
            #   False Positive: w <- w - x
            #   False Negative: w <- w + x
            for c in range(num_classes):
                # Binary target: +1 for the true class, -1 for others
                target = 1 if c == true_label else -1

                # Compute the perceptron score (weighted sum) using dot product
                score = np.dot(weights[c], x)

                # Update rule for misclassification
                if (score > 0 and target == -1):  # False Positive, 강의자료에 S>1일때 +1, S<=1일때 -1 이라고 명시되어 있음.
                    weights[c] -= x # lr=1.0 이지만 문제에 명시된 대로 weights[c] -= (lr * x) 대신 weights[c] -= x 로 작성
                    num_updates += 1
                elif (score <= 0 and target == 1):  # False Negative
                    weights[c] += x # lr=1.0 이지만 문제에 명시된 대로 weights[c] += (lr * x) 대신 weights[c] += x 로 작성
                    num_updates += 1

        print(f"Epoch {epoch + 1}/{epochs} completed - updates: {num_updates}")

    return weights
