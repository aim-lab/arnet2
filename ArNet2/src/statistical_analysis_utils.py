from utils.base_packages import *


def paired_T_test(errors_af_burden_dict, model_1, model_2, set='test_all'):
    """
    Calculates the t-test on afb error dicts from two models, model_1 and model_2.
    :param errors_af_burden_dict: dictionary of test sets names as keys, models as sub-keys and errors between true afb
    and predicted afb as values.
    :param model_1: str. name of the first model to test.
    :param model_2: str. name of the second model to test.
    :param set: The test set for which to fetch errors_af_burden values for.
    :return: None.
    """
    # Fetch error lists per model per test set.
    errors_af_burden_all_1 = np.hstack(errors_af_burden_dict[set][model_1].values())
    errors_af_burden_all_2 = np.hstack(errors_af_burden_dict[set][model_2].values())

    # Calculate mean, std and T value for the t-test.
    mean = np.mean(np.abs(errors_af_burden_all_1) - np.abs(errors_af_burden_all_2))
    std = np.std(np.abs(errors_af_burden_all_1) - np.abs(errors_af_burden_all_2), ddof=1)
    T = mean / (std / np.sqrt(len(errors_af_burden_all_2)))

    # Calculate t-test
    (stat, pval) = ttest_rel(errors_af_burden_all_1, errors_af_burden_all_2)
    print(f"Statistical testing (T-test) {model_1} vs {model_2}:")
    print(f"The t-values is: {T}\n")
    print('|EAF (%)| was {} with p-value of {}<0.0001'.format(
        'statistically significant' if pval < 0.0001 else 'not statistically significant', pval))


def propotional_T_test(model_1, model_2, set='test_all'):
    df_model_1 = pd.read_csv(cts.REPO_DIR / 'output' / f'{model_1}_{set}_pred.csv')
    df_model_2 = pd.read_csv(cts.REPO_DIR / 'output' / f'{model_2}_{set}_pred.csv')
    sample_success_model_1, sample_size_model_1 = (np.count_nonzero(df_model_1.pred), len(df_model_1))
    sample_success_model_2, sample_size_model_2 = (np.count_nonzero(df_model_2.pred), len(df_model_2))
    successes = np.array([sample_success_model_1, sample_success_model_2])
    samples = np.array([sample_size_model_1, sample_size_model_2])
    (stat, pval) = proportions_ztest(count=successes, nobs=samples, alternative='two-sided')
    print(f"Statistical testing (propotional_T_test) {model_1} vs {model_2}:")
    print('F1-score was {} with p-value of {}<0.0001'.format(
        'statistically significant' if pval < 0.0001 else 'not statistically significant', pval))


def statistical_post_hoc_test(df, val_col_kruskal, val_col, group_col):
    for group in df[group_col].unique():
        stat, p = kruskal(df.loc[df[group_col].eq(val_col_kruskal), val_col],
                          df.loc[df[group_col].eq(group), val_col])
        print(f'{val_col_kruskal} vs {group}:')
        print('Statistics=%.3f, p=%f' % (stat, p))
    psthoc_df = sp.posthoc_dunn(df, val_col=val_col, group_col=group_col, p_adjust='holm')
    return psthoc_df
