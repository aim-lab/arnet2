""" A script for vanilla semi-supervised learning. Didn't have time to evaluate it"""

import numpy as np
from src.train_and_eval import train

print(f"Semi-supervised learning...")
print("Loading unlabeled data...")
X_rbaf = np.load("/MLdata/AIMLab/Tom/rbaf_features.npy", allow_pickle=True)
X_rbaf = np.concatenate((X_rbaf[:, :-1], np.zeros((X_rbaf.shape[0], 1)), X_rbaf[:, -1:]), axis=1)
# X_rbaf = np.concatenate((X_rbaf, np.load(main_path / "X_train_not_reann.npy", allow_pickle=True)), axis=0)

# SSL loop
hypercomb = {"n_blocks": 6, "n_filters_start": 32, "filter_length": 5, "activation": "relu",
             "dropout_conv": 0.2, "n_hidden_start": 512, "dropout_fcn": 0.5, "learning_rate": 0.001,
             "af_weight": 3}
iter = 0
while iter < 50:
    print()
    print()
    print("===============")
    print(f"ITERATION {iter}")
    print("===============")

    # Train
    model = train((X_train, y_train), hypercomb["ResNet"], algo="ResNet", n_epochs=5, validation_data=None,
                  path_feature_extractor=None)

    # Predict unlabeled
    probas_rbaf = model.predict_proba(X_rbaf[:, :-3])[:, 1]
    y_rbaf = (probas_rbaf > best_th)

    # Add samples with confident predictions to training set and remove them from unlabeled set
    confidence_rbaf = np.maximum(probas_rbaf, 1 - probas_rbaf)
    confidence_criterion = (0.99 <= confidence_rbaf)
    print(f"Number of confident samples = {np.sum(confidence_criterion)}")
    if np.sum(confidence_criterion) < 1000:
        print("SSL converged !")
        print("Saving augmented data...")
        np.save(main_path / "X_ssl", X_train)
        np.save(main_path / "y_ssl", y_train)
        break
    elif np.sum(confidence_criterion) > 100000:
        print("Limiting added samples to the top-100,000 confidences")
        top_confidence_idx = confidence_rbaf.argsort()[-100000:]
        top_confidence_criterion = np.zeros_like(confidence_criterion)
        top_confidence_criterion[top_confidence_idx] = 1
        confidence_criterion = np.logical_and(confidence_criterion, top_confidence_criterion)

    X_add = X_rbaf[confidence_criterion]
    y_add = y_rbaf[confidence_criterion]
    proba_add = probas_rbaf[confidence_criterion]
    print("Probas added : ", end='')
    print(proba_add.shape)
    print(proba_add)

    X_train, y_train = np.concatenate((X_train, X_add), axis=0), np.concatenate((y_train, y_add), axis=0)
    X_rbaf = np.delete(X_rbaf, confidence_criterion, axis=0)
    iter += 1