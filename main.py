from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import numpy as np
import random

def balanced_dataset(file: str, balanced: bool = True) -> list:
    """
    Load the data from the pickle file and organize it in lists for t-SNE and bar chart

    Args: 
        file: the path to the pickle file
        balanced: if the data should be balanced
    
    Returns:
        embeddings: the embeddings of the images
        labels: the labels of the images
        classes: the classes of the images
    """

    # Load the data from the pickle file
    data = pd.read_pickle(file)

    # Initialize the dictionaries
    class_counts = {}
    class_data = {}

    # Organize the data in lists or arrays for t-SNE
    embeddings = []
    labels = []

    # Iterate over the data and organize it in lists
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                if balanced == False:
                    embeddings.append(embedding)
                    labels.append(syndrome_id)
                if syndrome_id in class_data:
                    class_data[syndrome_id].append([syndrome_id, subject_id, image_id, embedding])
                else:
                    class_data[syndrome_id] = [[syndrome_id, subject_id, image_id, embedding]]

                class_counts[syndrome_id] = class_counts.get(syndrome_id, 0) + 1

    # Convert the count data to lists for the bar chart
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    # If the data should be balanced
    if balanced:
        embeddings.clear()
        labels.clear()

        # Perform random balancing
        min_samples = min(counts)

        for syndrome_id, images_list in class_data.items():
            numbers = len(images_list) - min_samples
            excluded_numbers = random.sample(images_list, numbers)

            final_imgs = [image for image in images_list if image not in excluded_numbers]
            for image in final_imgs:
                embeddings.append(image[3])
                labels.append(syndrome_id)

            class_data[syndrome_id] = final_imgs

    return embeddings, labels, classes

def plot_tsne(embeddings, labels, classes):
    """
    Plot the t-SNE of the embeddings

    Args:
        embeddings: the embeddings of the images
        labels: the labels of the images
        classes: the classes of the images
    """

    # Perform t-SNE and get the 2D embeddings
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)
    x = embeddings_2d[:, 0]
    y = embeddings_2d[:, 1]

    # Plot the t-SNE
    plt.figure(figsize=(6, 5))
    for syndrome_id in classes:
        plt.scatter(x[labels == syndrome_id], y[labels == syndrome_id], label=syndrome_id)
    plt.legend()
    plt.show()
 
    return True

def knn_classifier(embeddings, labels, classes):
    """
    Perform KNN classification using cosine and euclidean distances

    Args:
        embeddings: the embeddings of the images
        labels: the labels of the images
        classes: the classes of the images
    """

    # Initialize the lists for the metrics
    cosine_accuracies = []
    cosine_f1_scores = []
    euclidean_accuracies = []
    euclidean_f1_scores = []
    cosine_roc_aucs = []
    euclidean_roc_aucs = []

    # Initialize the stratified k-fold object:
    kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold = 0

    # Performing 10-fold cross-validation:
    for train_index, test_index in kfolds.split(embeddings, labels):
        X_train_fold, X_test_fold = embeddings[train_index], embeddings[test_index]
        y_train_fold, y_test_fold = labels[train_index], labels[test_index]

        # Calculating cosine distance between train and test:
        cosine_dists_train = np.dot(X_train_fold, X_train_fold.T)
        cosine_dists_test = np.dot(X_test_fold, X_train_fold.T)

        # Calculating euclidean distance between train and test:
        euclidean_dists_train = np.linalg.norm(X_train_fold[:, np.newaxis] - X_train_fold, axis=2)
        euclidean_dists_test = np.linalg.norm(X_test_fold[:, np.newaxis] - X_train_fold, axis=2)

        # Fit KNN models with calculated distances:
        knn_cosine = KNeighborsClassifier()
        knn_euclidean = KNeighborsClassifier()
        knn_cosine.fit(cosine_dists_train, y_train_fold)
        knn_euclidean.fit(euclidean_dists_train, y_train_fold)

        # Predict for test using cosine distance:
        y_pred_cosine = knn_cosine.predict(cosine_dists_test)

        # Metrics for cosine distances:
        cosine_accuracy = accuracy_score(y_test_fold, y_pred_cosine)
        cosine_accuracies.append(cosine_accuracy)
        cosine_f1 = f1_score(y_test_fold, y_pred_cosine, average='macro')
        cosine_f1_scores.append(cosine_f1)

        # Predict using euclidean distance:
        y_pred_euclidean = knn_euclidean.predict(euclidean_dists_test)

        # Metrics for euclidean distances:
        euclidean_accuracy = accuracy_score(y_test_fold, y_pred_euclidean)
        euclidean_accuracies.append(euclidean_accuracy)
        euclidean_f1 = f1_score(y_test_fold, y_pred_euclidean, average='macro')
        euclidean_f1_scores.append(euclidean_f1)

        # Increment Fold:
        fold += 1

        # Get classes probabilities:
        y_score_cosine = knn_cosine.predict_proba(cosine_dists_test)

        fpr_cosine = {}  # false positives rate
        tpr_cosine = {}  # true positives rate
        roc_auc_cosine = {}

        # For each class in our data:
        for i, syndrome_id in enumerate(knn_cosine.classes_):
            fpr_cosine[i], tpr_cosine[i], _ = roc_curve(y_test_fold == syndrome_id, y_score_cosine[:, i])
            roc_auc_cosine[i] = auc(fpr_cosine[i], tpr_cosine[i])

        # Retrieve data for this fold:
        cosine_roc_aucs.append(roc_auc_cosine)

        # Get classes probabilities:
        y_score_euclidean = knn_euclidean.predict_proba(euclidean_dists_test)

        fpr_euclidean = {}  # false positives rate
        tpr_euclidean = {}  # true positives rate
        roc_auc_euclidean = {}

        # For each class in our data:
        for i, syndrome_id in enumerate(knn_euclidean.classes_):
            fpr_euclidean[i], tpr_euclidean[i], _ = roc_curve(y_test_fold == syndrome_id, y_score_euclidean[:, i])
            roc_auc_euclidean[i] = auc(fpr_euclidean[i], tpr_euclidean[i])

        # Retrieve data for this fold:
        euclidean_roc_aucs.append(roc_auc_euclidean)

    # Calculate the average ROC AUC for each class across all folds
    avg_roc_auc_cosine = {i: np.mean([roc_auc[i] for roc_auc in cosine_roc_aucs]) for i in range(len(classes))}
    avg_roc_auc_euclidean = {i: np.mean([roc_auc[i] for roc_auc in euclidean_roc_aucs]) for i in range(len(classes))}

    # Plot the ROC AUC for each class
    for i, syndrome_id in enumerate(classes):

        # Print metrics for current fold:
        print(f'\nSyndrome: {syndrome_id}')
        print(f'Accuracy - Cosine Distance: {cosine_accuracy}')
        print(f'Accuracy - Euclidean Distance: {euclidean_accuracy}')
        print(f'F1 Score - Cosine Distance: {cosine_f1}')
        print(f'F1 Score - Euclidean Distance: {euclidean_f1}')

        fig, axs = plt.subplots(2, figsize=(10, 5))  # Create a new figure for each class

        # Get the number of observations for this class
        n_cosine = len([label for label in labels if label == syndrome_id])

        # Plot ROC AUC for cosine distance
        axs[0].plot([0, 1], [0, 1], 'k--')
        axs[0].plot(fpr_cosine[i], tpr_cosine[i], label=f'{syndrome_id} (n={n_cosine}, area = {avg_roc_auc_cosine[i]:.2f})')
        axs[0].set_xlabel('False positive rate')
        axs[0].set_ylabel('True positive rate')
        axs[0].set_title('ROC curve - Cosine Distance')
        axs[0].legend(loc='best')

        # Get the number of observations for this class
        n_euclidean = len([label for label in labels if label == syndrome_id])

        # Plot ROC AUC for euclidean distance
        axs[1].plot([0, 1], [0, 1], 'k--')
        axs[1].plot(fpr_euclidean[i], tpr_euclidean[i], label=f'{syndrome_id} (n={n_euclidean}, area = {avg_roc_auc_euclidean[i]:.2f})')
        axs[1].set_xlabel('False positive rate')
        axs[1].set_ylabel('True positive rate')
        axs[1].set_title('ROC curve - Euclidean Distance')
        axs[1].legend(loc='best')

        plt.tight_layout()
        plt.show()

    # Calculate the average accuracy, F1 score and ROC AUC for each class across all folds
    avg_accuracy_cosine = np.mean(cosine_accuracies)
    avg_f1_cosine = np.mean(cosine_f1_scores)
    avg_roc_auc_cosine = np.mean([roc_auc for roc_auc in avg_roc_auc_cosine.values()])

    avg_accuracy_euclidean = np.mean(euclidean_accuracies)
    avg_f1_euclidean = np.mean(euclidean_f1_scores)
    avg_roc_auc_euclidean = np.mean([roc_auc for roc_auc in avg_roc_auc_euclidean.values()])

    # Create a table for the cosine distance
    table_cosine = [["Accuracy", avg_accuracy_cosine],
                    ["F1 Score", avg_f1_cosine],
                    ["ROC AUC", avg_roc_auc_cosine]]

    # Create a table for the euclidean distance
    table_euclidean = [["Accuracy", avg_accuracy_euclidean],
                    ["F1 Score", avg_f1_euclidean],
                    ["ROC AUC", avg_roc_auc_euclidean]]

    # Write the tables to a txt file
    with open("comparison.txt", "w") as f:
        f.write("Cosine Distance\n")
        f.write(tabulate(table_cosine, headers=["Metric", "Value"], tablefmt="plain"))
        f.write("\n\nEuclidean Distance\n")
        f.write(tabulate(table_euclidean, headers=["Metric", "Value"], tablefmt="plain"))

    return True

if __name__ == '__main__':
    embeddings, labels, classes = balanced_dataset('./mini_gm_public_v0.1.p', balanced=True)
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    classes = np.array(classes)

    plot_tsne(embeddings, labels, classes)
    knn_classifier(embeddings, labels, classes)
