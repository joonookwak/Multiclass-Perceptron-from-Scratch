from load_digits import load_images, load_labels
from train import train_perceptron
from test import test_perceptron


def main():
    # Load MNIST data files
    train_images = load_images("train-images-idx3-ubyte/train-images-idx3-ubyte")
    train_labels = load_labels("train-labels-idx1-ubyte/train-labels-idx1-ubyte")
    test_images = load_images("t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
    test_labels = load_labels("t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")

    # Train the multiclass perceptron
    perceptron_weights = train_perceptron(train_images, train_labels, epochs=5, lr=1.0)

    # Evaluate on the test set
    error_rate = test_perceptron(test_images, test_labels, perceptron_weights)

    print(f"Final Test Error Rate: {error_rate:.2f}%")


if __name__ == "__main__":
    main()