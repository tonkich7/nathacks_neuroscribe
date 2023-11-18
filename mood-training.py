import importlib  
models = importlib.import_module("mood-data-models")
import numpy as np
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
from mne.decoding import (
    SlidingEstimator,
    cross_val_multiscore,
    Scaler,
    Vectorizer
)

FILE_PATH = './experiment_data/mood_epochs_'
FILE_TIME = '1700266292'

NUM_FOLDS = 5

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = torch.randperm(len(a))
    return a[p], b[p]

def one_hot_encode(y):
    labels = np.unique(y)
    y_indices = ((y / 1000) - 1).astype(int) #maps 1000,2000,3000 to 0,1,2
    one_hot_y = np.zeros((y.shape[0], labels.shape[0]))
    for row in range(y.shape[0]):
        one_hot_y[row,y_indices[row]] += 1
    return one_hot_y

def build_folds(X, y, num_folds=5):
    print(X.shape)
    X_folds = np.zeros((num_folds, X.shape[0]//num_folds, X.shape[1], X.shape[2]))
    y_folds = np.zeros((num_folds, y.shape[0]//num_folds, y.shape[1]))
    samples_per_label = X.shape[0]//(num_folds*y.shape[1])
    for label_count in range(y.shape[1]):
        start_idx = label_count*samples_per_label
        end_idx = (label_count+1)*samples_per_label
        indices = np.argwhere(y[:,label_count] == 1)
        np.random.shuffle(indices)
        for fold_count, split in enumerate(np.split(indices, num_folds)):
            split = split.flatten()
            X_folds[fold_count, start_idx:end_idx, :, :] = X[split]
            y_folds[fold_count, start_idx:end_idx] = y[split]
    return X_folds, y_folds        

def train_model(model, X, y, alpha=1e-3, num_epochs=50):
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        X, y = unison_shuffled_copies(X, y)
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == num_epochs-1:
            max_preds = torch.argmax(preds, dim=1)
            max_labels = torch.argmax(y, dim=1)
            accuracy = torch.sum(max_preds == max_labels)/y.shape[0]
            print("\tEpoch: ", epoch, ", Train Loss: ", loss.item(), "Train Accuracy:", accuracy.item())

def cross_val(X, y, epoch_info):
    print("Building folds...")
    X_folds, y_folds = build_folds(X, y, NUM_FOLDS)
    print("Built folds with input shape:", X_folds.shape, "and output shape:", y_folds.shape)
    for test_fold_idx in range(NUM_FOLDS):
        print("Test fold:", test_fold_idx)
        #build training and testing sets
        for fold_idx in range(NUM_FOLDS):
            if fold_idx == test_fold_idx:
                continue
            try:
                X_train = np.vstack((X_train, X_folds[fold_idx]))
                y_train = np.vstack((y_train, y_folds[fold_idx]))
            except:
                X_train = X_folds[fold_idx]
                y_train = y_folds[fold_idx]
        print("Built training set with inputs of shape:", X_train.shape, "and outputs of shape:", y_train.shape)
        X_test = X_folds[test_fold_idx]
        y_test = y_folds[test_fold_idx]
        print("Built testing set with inputs of shape:", X_test.shape, "and outputs of shape:", y_test.shape)
        scaler = Scaler(epoch_info)
        vectorizer = Vectorizer()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_vectorized = vectorizer.fit_transform(X_train_scaled)

        model = models.LinearModel(X_train_vectorized.shape[1], y.shape[1])
        train_model(model, torch.tensor(X_train_vectorized).float(), torch.tensor(y_train).float())

        #get test loss and accuracy from model
        X_test_scaled = scaler.transform(X_test)
        X_test_vectorized = vectorizer.transform(X_test_scaled)
        loss_fn = nn.CrossEntropyLoss()
        test_preds = model(torch.tensor(X_test_vectorized).float())
        y_tensor = torch.tensor(y_test).float()
        test_loss = loss_fn(test_preds, y_tensor)
        max_preds = torch.argmax(test_preds, dim=1)
        print(max_preds)
        max_labels = torch.argmax(y_tensor, dim=1)
        print(max_labels)
        test_accuracy = torch.sum(max_preds == max_labels)/y_test.shape[0]
        print("Test Loss:", test_loss.item(), "Test Accuracy:", test_accuracy.item())


def main():
    print("Reading data file...")
    epochs = mne.read_epochs(FILE_PATH + FILE_TIME)
    epochs = epochs.apply_baseline()
    X = epochs.get_data()
    y = epochs.events[:,2]
    print("One-hot Encoding labels...")
    one_hot_y = one_hot_encode(y)
    print("Cross-validating...")
    cross_val(X, one_hot_y, epochs.info)

main()

