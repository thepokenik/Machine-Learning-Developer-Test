import numpy as np
import matplotlib.pyplot as plt
from main import plot_tsne

def test_plot_tsne():
    # Mock data for testing
    embeddings = np.random.rand(100, 10)  # Random embeddings
    labels = np.random.choice(['class_1', 'class_2', 'class_3'], size=100)  # Random labels
    classes = ['class_1', 'class_2', 'class_3']  # List of classes

    # Call the function with mock data
    fig = plot_tsne(embeddings, labels, classes)

    # Assert that the figure object is not None
    assert fig is not None

# Run the test
test_plot_tsne()
