# General imports
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective
import matplotlib.pyplot as plt
import pickle

# Relative imports
import src.data.data_loading as data_loading
import src.utils.consts as cts
import src.models.model_utils as model_utils
from src.train_and_eval import train, eval

# Models hyperparameters lists
hyperparameters_space = {
    'XGB':
        {
            "n_estimators": Integer(low=50, high=500, prior="uniform", name='n_estimators'),
            "max_depth": Integer(low=2, high=10, prior="uniform", name='max_depth'),
        },
    '1D-CNN':
        {
            "n_filters_start": Integer(low=32, high=128, prior="log-uniform", base=2, name='n_filters_start'),
            "len_sub_window": Integer(low=3, high=10, prior="uniform", name='len_sub_window'),
            "dropout": Real(low=0, high=0.5, prior="uniform", name='dropout'),
            "n_hidden_start": Integer(low=64, high=512, prior="log-uniform", base=2, name='n_hidden_start')
        },
    'ResNet':
        {
            "n_blocks": Integer(low=2, high=6, prior="uniform", name='n_blocks'),
            "n_filters_start": Integer(low=32, high=64, prior="log-uniform", base=2, name='n_filters_start'),
            "filter_length": Integer(low=3, high=10, prior="uniform", name='filter_length'),
            "dropout_conv": Real(low=0, high=0.5, prior="uniform", name='dropout_conv'),
            "n_hidden_start": Integer(low=64, high=512, prior="log-uniform", base=2, name='n_hidden_start'),
            "dropout_fcn": Real(low=0, high=0.8, prior="uniform", name='dropout_fcn'),
            "learning_rate": Real(low=1e-5, high=1e-2, prior='log-uniform', name='learning_rate'),
            "af_weight": Integer(low=1, high=5, prior="uniform", name='af_weight')
        },
    'ArNet2':
        {
            "time_history": Integer(low=3, high=20, prior="uniform", name='time_history'),
            "extract_level": Categorical(categories=['dense', 'dense_1', 'dense_2'], name='extract_level'),
            "n_units": Integer(low=8, high=64, prior="log-uniform", base=2, name='n_units'),
            "dropout": Real(low=0, high=0.8, prior="uniform", name='dropout'),
            "learning_rate": Real(low=1e-5, high=1e-2, prior='log-uniform', name='learning_rate'),
            "af_weight": Integer(low=1, high=5, prior="uniform", name='af_weight')
        },
    'CRNN':
        {
            "time_history": Integer(low=3, high=20, prior="uniform", name='time_history'),
            "n_blocks": Integer(low=2, high=7, prior="uniform", name='n_blocks'),
            "n_filters_start": Integer(low=8, high=32, prior="log-uniform", base=2, name='n_filters_start'),
            "filter_length": Integer(low=3, high=10, prior="uniform", name='filter_length'),
            "dropout_conv": Real(low=0, high=0.5, prior="uniform", name='dropout_conv'),
            "n_units": Integer(low=8, high=512, prior="log-uniform", base=2, name='n_units'),
            "n_dense": Integer(low=1, high=3, prior="uniform", name='n_dense'),
            "dropout_fcn": Real(low=0, high=0.8, prior="uniform", name='dropout_fcn'),
            "learning_rate": Real(low=1e-5, high=1e-2, prior='log-uniform', name='learning_rate'),
            "af_weight": Integer(low=1, high=5, prior="uniform", name='af_weight')
        }
}
default_hyperparameters = {
    'XGB': [70, 4],
    '1D-CNN': [32, 5, 0.2, 512],
    'ResNet': [[6, 32, 5, 0.2, 512, 0.4, 1e-3, 3], [2, 32, 5, 0.2, 64, 0.4, 1e-3, 3]],
    'ArNet2': [[20, 'dense_1', 16, 0.1, 0.001, 3], [10, 'dense_1', 8, 0.4, 0.001, 3]],
    'CRNN': [[10, 2, 32, 5, 0.2, 8, 1, 0.4, 0.001, 3], [10, 6, 32, 5, 0.2, 64, 3, 0.4, 0.001, 3]]
}

def bayesian_search(data, path_models, algo="1D-CNN", n_calls=100, criterion_name="AUROC_val"):
    """ Perform Bayesian Hyperparameter Tuning for a given model.

    :param data: a tuple of datasets, each dataset being a tuple (X, y)
    :param path_models: the path of the folder in which we want to save the model and related results, and in which the possible feature extractor is previously stored in the case of ArNet.
    :param algo: a string representing the model's name
    :param n_calls: the number of iterations of the bayesian search
    :param criterion_name: the metric to optimize during the tuning. It can be e.g. "AUROC_val" or "Fb-Score_val".
    """

    @use_named_args(dimensions=hyperparameters_space[algo].values())
    def objective(**hypercomb):  # inspired from https://towardsdatascience.com/bayesian-hyper-parameter-optimization-neural-networks-tensorflow-facies-prediction-example-f9c48d21f795
        """ The objective function to minimize using a gaussian process. It needs to follow `skopt` constraints on inputs and outputs.

        :param hypercomb: the dictionary of model's hyperparameters (for saving)
        :return criterion: the objective function's output, i.e. the criterion we want to minimize
        """
        # Convert hyperparameter combination (skopt issue)
        def convert_hypercomb(hypercomb):
            for key, value in hypercomb.items():
                if isinstance(value, np.int64):
                    hypercomb[key] = int(value)
        convert_hypercomb(hypercomb)
        print("\n\nHYPERPARAMETERS : ", hypercomb, end='\n\n')

        # Train and evaluate
        print(f"Training {algo}...")
        model = train(data_train, hypercomb, algo=algo, n_epochs=5, validation_data=data_val, path_feature_extractor=path_models/"best_cv_ResNet.pkl")
        print(f"Evaluating {algo}...")
        metrics_dict, best_th_dict, mean_abs_afb_error_dict = eval(model, algo=algo, decision_th=None, **{'train': data_train, 'val': data_val})

        # Add new line in the results dataframe for the current hypercomb
        global results
        result_values = list(hypercomb.values()) + [best_th_dict['train']] + list(metrics_dict['train']) + [mean_abs_afb_error_dict['train']] + [best_th_dict['val']] + list(metrics_dict['val']) + [mean_abs_afb_error_dict['val']]
        result_dict = {key: value for key, value in zip(list(results.columns), result_values)}
        print("Adding to the results dataframe : ", result_dict)
        results = results.append(result_dict, ignore_index=True)
        model_utils.save_df(results, path_models / f'bayesian_optimization_{algo}.csv')

        # If criterion > best previous one, update it and save the corresponding model
        criterion = result_dict[criterion_name]
        print(f"{criterion_name} = {criterion}")
        global best_criterion
        print(f"best {criterion_name} = {best_criterion}")
        if criterion > best_criterion:
            print(f"NEW BEST {criterion_name} !")
            best_criterion = criterion
            print("Saving model...")
            model_dict = {"hyperparameters": hypercomb, 'best_th': best_th_dict['val']}  # note that we save the decision threshold optimized on the validation set
            with open(path_models / f"best_cv_{algo}.pkl", 'wb') as file:
                if algo == "XGB":
                    model_dict['classifier'] = model
                else:
                    model_dict['classifier'] = model.get_state_dict()
                pickle.dump(model_dict, file)
            del model  # delete the Keras model with these hyper-parameters from memory

        return -criterion  # be it AUROC or F1, we'll have to maximize it, so minimize it's opposite

    # Get data
    data_train, data_val, data_test, data_ltaf, data_jpaf, data_afdb = data

    # Bayesian optimization
    search_result = gp_minimize(func=objective, dimensions=hyperparameters_space[algo].values(), n_calls=n_calls, x0=x0_gp, y0=y0_gp)

    # Plot and save Bayesian Optimization results
    print(search_result)
    plot_convergence(search_result)
    plt.savefig(path_models / f"Converge_{algo}.png", dpi=400)
    plot_objective(result=search_result)
    plt.savefig(path_models / f"Lr_numnods_{algo}.png", dpi=400)


    # FINAL EVALUATION
    print()
    print("===========")
    print("FINAL EVALUATION")
    print("===========")

    # Load saved best model
    print("Loading best model...")
    model_dict = model_utils.load_model(path_models / f"best_cv_{algo}.pkl", algo, path_feature_extractor=path_models/"best_cv_ResNet.pkl")
    model = model_dict['classifier']
    hypercomb = model_dict["hyperparameters"]
    best_th = model_dict['best_th']
    print("BEST HYPERPARAMS : ", hypercomb)
    print("Results on train-val-test-ltaf-jpaf : ")
    eval(model, algo=algo, decision_th=best_th, **{'train': data_train, 'val': data_val, 'test': data_test, 'ltaf': data_ltaf, 'jpaf': data_jpaf, 'afdb': data_afdb})

    print("Retraining best model on train + val...")
    data_train = (np.concatenate((data_train[0], data_val[0]), axis=0), np.concatenate((data_train[1], data_val[1]), axis=0))
    model = train(data_train, hypercomb, algo=algo, n_epochs=5, validation_data=None, path_feature_extractor=path_models/"best_cv_ResNet.pkl")
    eval(model, algo=algo, decision_th=best_th, save=True, path_models=path_models, hypercomb=hypercomb, **{'train': data_train, 'test': data_test, 'ltaf': data_ltaf, 'jpaf': data_jpaf, 'afdb': data_afdb})


if __name__ == '__main__':
    folder_date = "2021-08-26T15:49:46"  # np.datetime64('now') / "2021-04-21T10:00:00"
    path_models = cts.BASE_DIR / "Tom" / "hyperparameter_tuning" / str(folder_date)
    algo = "CRNN"
    n_calls = 150
    criterion_name = "AUROC_val"
    recup_mode = True

    # Load data
    print("Loading data...")
    data = data_loading.load_data(algo)

    # Initialize bayesian search results (global variables)
    if not recup_mode:
        results = pd.DataFrame(columns=list(hyperparameters_space[algo].keys()) + ['best_th_train'] + [metric + '_train' for metric in cts.METRICS] + ['mean_abs_afb_error_train'] + ['best_th_val'] + [metric + '_val' for metric in cts.METRICS] + ['mean_abs_afb_error_val'])
        x0_gp = default_hyperparameters[algo]
        y0_gp = None
        best_criterion = 0
    else:
        results = pd.read_csv(path_models / f'bayesian_optimization_{algo}.csv', index_col=0)
        x0_gp = [[results.loc[i, hyper_name] for hyper_name in list(hyperparameters_space[algo].keys())] for i in range(len(results))]
        y0_gp = - np.array(results[criterion_name])
        best_criterion = - np.min(y0_gp)
        n_calls = n_calls - len(y0_gp)  # subtract the runs already done
        print("RESULTS OF PREVIOUS RUNS : ")
        print(results)
        print("Input parameters : ")
        print(x0_gp)
        print(y0_gp)
        print(f"Best {criterion_name}")
        print(best_criterion)

    # Bayesian Search
    bayesian_search(data, path_models, algo=algo, n_calls=n_calls, criterion_name=criterion_name)
