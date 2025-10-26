import pandas as pd
import matplotlib.pyplot as plt
from rbaf_parser import RBAFDB_Parser
import numpy as np
import dat_reader as dr
import scipy.stats as stats
# from lmfit import Model
from scipy.optimize import curve_fit
from scipy.special import i0
import matplotlib.gridspec as gridspec
import time
import consts as cts
import pathlib
import itertools

# Define the von mises kernel density estimator
def circular_von_mises_kde(x, mu, sigma):
    # Adjust data to take it to range of 2pi
    x = [(hr) * 2 * np.pi / 24 for hr in x]
    mu *= 2 * np.pi / 24
    sigma *= 2 * np.pi / 24

    # Compute kappa for vm kde
    kappa = 1 / sigma ** 2
    return np.exp((kappa) * np.cos((x - mu))) / (2 * np.pi * i0(kappa))


def sinefunction(x, a, b, c):
    return a + b * np.sin(x * np.pi / 180.0 + c)


def residual(variables, x, data, eps_data):
    """Model a decaying sine wave and subtract data."""
    amp = variables[0]
    phaseshift = variables[1]
    freq = variables[2]
    decay = variables[3]

    model = amp * sin(x * freq + phaseshift) * exp(-x * x * decay)
    return (data - model) / eps_data


def get_shape_hist(ids, circadian_df):
    flag_hist = pd.DataFrame(columns={'timestamp', 'N', 'AB', 'I', 'P'})
    flag_dict = {}
    times = pd.date_range("00:00", "23:59", freq="60min").time
    for i in range(len(times)):
        flag_dict[i] = {1: 0, 4: 0, 5: 0, 6: 0}
    population_ABERRANT_df = pd.DataFrame([])
    population_INHIBIT_df = pd.DataFrame([])
    population_PACED_df = pd.DataFrame([])

    for i in ids:
        start = circadian_df['start_recording'][i]
        beat_flags = dr.combflag_file_reader(db.raw_ecg_path / (i + '.FUL') / (db.beat_flag_file_name + '.dat'))
        beat_flags[beat_flags == 3] = 1
        beat_df = pd.DataFrame(columns={'rrt', 'flag'})
        beat_df['rrt'] = start + db.rrt_dict[i][:len(beat_flags)]
        beat_df['flag'] = beat_flags[:len(beat_df['rrt'])]
        beat_df['timestamp'] = pd.to_datetime(beat_df['rrt'], unit='s', utc=True)
        # beat_df.timestamp = beat_df.timestamp.dt.strftime('%H:%M:%S')
        population_ABERRANT_df = pd.concat([population_ABERRANT_df, beat_df[beat_df.flag == 4].timestamp])
        population_INHIBIT_df = pd.concat([population_INHIBIT_df, beat_df[beat_df.flag == 5].timestamp])
        population_PACED_df = pd.concat([population_PACED_df, beat_df[beat_df.flag == 6].timestamp])

        # beat_df = beat_df.set_index('timestamp')
        dfs = [x for y, x in beat_df.groupby(beat_df.timestamp.dt.hour)]

        for t in range(len(dfs)):
            idx = dfs[t].timestamp.iloc[0].hour
            dfs_values = dfs[t].flag.value_counts().to_dict()
            for key, value in dfs_values.items():
                flag_dict[idx][key] += value

    flag_hist = flag_hist.from_dict(flag_dict, orient='index')
    flag_hist.rename(columns={1: 'Normal', 4: 'ABERRANT', 5: 'INHIBIT', 6: 'PACED'}, inplace=True)
    return flag_hist, population_ABERRANT_df, population_INHIBIT_df, population_PACED_df


def get_shape_hist_AF(med_windows):
    flag_hist = pd.DataFrame(columns={'timestamp', 'N', 'AF'})
    flag_dict = {}
    times = pd.date_range("00:00", "23:59", freq="60min").time
    for i in range(len(times)):
        flag_dict[i] = {0: 0, 1: 0}
    population_AF_df = pd.DataFrame([])

    for i, ID in enumerate(med_windows.patient_id.values):
        beat_df = pd.DataFrame(columns={'rrt', 'flag'})
        beat_df['rrt'] = [med_windows.iloc[i]['start']]
        beat_df['flag'] = [med_windows.iloc[i]['window'].astype(int)]
        beat_df['timestamp'] = pd.to_datetime(beat_df['rrt'], unit='s', utc=True)
        # beat_df.timestamp = beat_df.timestamp.dt.strftime('%H:%M:%S')
        population_AF_df = pd.concat([population_AF_df, beat_df[beat_df.flag == 1].timestamp])
        dfs = [x for y, x in beat_df.groupby(beat_df.timestamp.dt.hour)]

        for t in range(len(dfs)):
            idx = dfs[t].timestamp.iloc[0].hour
            dfs_values = dfs[t].flag.value_counts().to_dict()
            for key, value in dfs_values.items():
                flag_dict[idx][key] += value
    flag_hist = flag_hist.from_dict(flag_dict, orient='index')
    flag_hist.rename(columns={0: 'N', 1: 'AF'}, inplace=True)
    return flag_hist, population_AF_df


def fit_cosinor(flag_hist, flags, c, save_fig=False, fig_name=None, set_dpi=400):
    Xdata = np.linspace(0, 360, 24)
    # hr_data = np.linspace(0,24,24)
    r = range(24)
    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), dpi=set_dpi)
    for f, c, ax in zip(flags, c, axes):
        ticks = [0, 50, 100, 150, 200, 250, 300, 350]
        times = [time.strftime('%H:%M', time.gmtime(round((x / 360) * 24) * 60 ** 2)) for x in ticks]
        smodel = Model(sinefunction)
        result = smodel.fit(flag_hist[f].values, x=Xdata, a=0, b=30000, c=0,
                            weights=np.sqrt(1.0 / flag_hist[f].values))
        chisqr = result.redchi
        rsq = 1 - result.residual.var() / np.var(flag_hist[f].values)
        p_value = stats.distributions.chi2.sf(chisqr, 3)
        print(result.fit_report())
        print("r_squred = ", rsq)
        print("p_value = ", p_value)

        ax.plot(Xdata, result.best_fit, '*', color='r', label=r'best fit, p_value< $10^{-30}$')
        ax.bar(Xdata, flag_hist[f].values, color=c, label=f)
        ax.legend(loc="upper left", ncol=1, fontsize=10)
        ax.set_xticks(ticks)
        ax.set_xticklabels(times)
        ax.set_xlabel("Time [hour]", fontsize=10)
        ax.tick_params(axis='y', which='major', labelsize=10, rotation=0)
        ax.tick_params(axis='x', which='major', labelsize=10, rotation=45)
    plt.tight_layout()

    Y = axes[0].get_tightbbox(figure.canvas.get_renderer())

    labels = ["(a)", "(b)", "(c)"]
    for a, label in zip(axes, labels):
        bbox = a.get_tightbbox(figure.canvas.get_renderer())
        figure.text(bbox.x0, Y.y1 + 100, label, fontsize=16, va="top", ha="left",
                    transform=None)

    # plt.tight_layout()

    # plt.subplots_adjust(top=0.2)
    if save_fig:
        figure.savefig("/home/shanybiton/AIMLabProjects/advanced_lab/figures/" + fig_name + ".png", dpi=set_dpi,
                       bbox_inches='tight')
    plt.close()


def fit_von_mises(population_PACED_df, population_INHIBIT_df, population_ABERRANT_df, c, save_fig=False, fig_name=None,
                  set_dpi=400):
    time_ABERRANT = population_ABERRANT_df[0].dt.hour + (population_ABERRANT_df[0].dt.minute / 60) + (
            (population_ABERRANT_df[0].dt.second + 10 ** -3 * population_ABERRANT_df[0].dt.microsecond) / 60 ** 2)
    time_INHIBIT = population_INHIBIT_df[0].dt.hour + (population_INHIBIT_df[0].dt.minute / 60) + (
            (population_INHIBIT_df[0].dt.second + 10 ** -3 *
             population_INHIBIT_df[0].dt.microsecond) / 60 ** 2)
    time_PACED = population_PACED_df[0].dt.hour + (population_PACED_df[0].dt.minute / 60) + (
            (population_PACED_df[0].dt.second + 10 ** -3 *
             population_PACED_df[0].dt.microsecond) / 60 ** 2)

    fit_params_ABERRANT, cov_ABERRANT = curve_fit(circular_von_mises_kde, time_ABERRANT, np.ones(len(time_ABERRANT)),
                                                  bounds=(0, 24))
    fit_params_INHIBIT, cov_INHIBIT = curve_fit(circular_von_mises_kde, time_INHIBIT, np.ones(len(time_INHIBIT)),
                                                bounds=(0, 24))
    fit_params_PACED, cov_PACED = curve_fit(circular_von_mises_kde, time_PACED.sort_values(), np.ones(len(time_PACED)),
                                            bounds=(0, 24))

    x_deg = [(hr) * 2 * np.pi / 24 for hr in np.linspace(1, 25, 1000)]
    ax = plt.subplot(111, polar=True, frameon=True)
    ax.plot(x_deg, circular_von_mises_kde(np.linspace(1, 25, 1000), *fit_params_ABERRANT), color=c[0],
            label='Von Mises Fit ABERRANT')
    ax.plot(x_deg, circular_von_mises_kde(np.linspace(1, 25, 1000), *fit_params_INHIBIT), color=c[1],
            label='Von Mises Fit INHIBIT')
    ax.plot(x_deg, circular_von_mises_kde(np.linspace(1, 25, 1000), *fit_params_PACED), color=c[2],
            label='Von Mises Fit PACED')
    # ax.grid(False)
    # ax.spines['polar'].set_visible(False)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=8)
    plt.tight_layout()
    if save_fig:
        plt.savefig("/home/shanybiton/AIMLabProjects/advanced_lab/figures/" + fig_name + ".png", dpi=set_dpi,
                    bbox_inches='tight')
    # plt.show()
    plt.close()

    return fit_params_ABERRANT, cov_ABERRANT, fit_params_INHIBIT, cov_INHIBIT, fit_params_PACED, cov_PACED


def plot_date_hist(circadian_df, save_fig=False, fig_name=None, set_dpi=400):
    # plot recording date histogram
    plt.figure(figsize=(17, 7), dpi=set_dpi)
    ax = circadian_df.groupby(circadian_df.recording_date.dt.year).recording_date.count().plot(kind="bar",
                                                                                               color='#16697a',
                                                                                               edgecolor='white',
                                                                                               zorder=3, alpha=0.8)
    ax.set_xlabel("Year", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=22, rotation=0)
    ax.set_ylabel("count", fontsize=24)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.grid(axis='y', zorder=0)
    if save_fig:
        plt.savefig("/home/shanybiton/AIMLabProjects/advanced_lab/figures/" + fig_name + ".png", dpi=set_dpi)
    # plt.box(False)
    plt.show()
    plt.close()


def plot_flag_hist(flag_hist, flags, c, N, save_fig=False, fig_name=None, set_dpi=400):
    # consts
    fig = plt.figure(dpi=set_dpi, figsize=(20, 31))

    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    gs = gridspec.GridSpec(2, 3, hspace=0.5, wspace=0.4)
    # regular histogram
    theta, width = np.linspace(0.0, 2 * np.pi, N, endpoint=False, retstep=True)
    barWidth = 0.80
    r = range(24)
    times = pd.date_range("00:00", "23:59", freq="60min").time
    ax1 = fig.add_subplot(gs[0, :])  # row 0 (top) spans all(3) columns

    flag_hist[['ABERRANT', 'INHIBIT', 'PACED']].plot(ax=ax1, kind='bar', color=['#16697a', '#db6400', '#ffa62b'],
                                                     edgecolor='white', width=barWidth, zorder=3, alpha=0.8,
                                                     figsize=(20, 14))

    ax1.set_xticks(r)
    ax1.set_xticklabels([times[i].strftime('%H:%M') for i in range(24)])
    ax1.set_xlabel("Time [hour]", fontsize=20)
    ax1.tick_params(axis='y', which='major', labelsize=18, rotation=0)
    ax1.tick_params(axis='x', which='major', labelsize=18, rotation=45)
    ax1.legend(loc='upper left', ncol=1, fontsize=20)
    ax1.set_ylabel("count", fontsize=20)
    ax1.grid(axis='y', zorder=0)

    # circular histogram

    m = 0
    axes = []
    for i, b in zip(c, flags):
        ax = plt.subplot(gs[1, m], polar=True, frameon=True)
        axes.append(ax)
        # circular histogram
        x = 200 / (flag_hist[b].max() - flag_hist[b].min())
        bottom = flag_hist[b].max() - x * (flag_hist[b].max() - flag_hist[b].min())

        bars = ax.bar(
            theta, flag_hist[b],
            width=width - 0.03,
            bottom=bottom,
            color=i, edgecolor="black", alpha=0.8
        )

        bars = ax.bar(
            theta, [flag_hist[b].max()] * 24,
            width=width - 0.03,
            bottom=bottom,
            color="#f39c12", alpha=0.1
        )
        # ax = plt.subplot(111, polar=True)
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.tick_params(axis='x', which='major', labelsize=16, rotation=0)
        tick = [bottom * .90, 0.97 * bottom]
        for t in np.arange(0.5 * width, 24 * (width), width):
            ax.plot([t, t], tick, lw=1, color="k")
        # Use custom colors and opacity

        ax.set_yticklabels([])
        fig.text(0, bottom * .80, "00", transform=ax.transData._b, ha='center', va='center', color='black', fontsize=16,
                 fontweight='bold')
        fig.text(bottom * .80, 0, "06", transform=ax.transData._b, ha='center', va='center', color='black', fontsize=16,
                 fontweight='bold')
        fig.text(0, -bottom * .80, "12", transform=ax.transData._b, ha='center', va='center', color='black',
                 fontsize=16,
                 fontweight='bold')
        fig.text(-bottom * .80, 0, "18", transform=ax.transData._b, ha='center', va='center', color='black',
                 fontsize=16,
                 fontweight='bold')
        m += 1
        fig.add_subplot(ax, figsize=(6, 14))

    Y1 = ax1.get_tightbbox(fig.canvas.get_renderer())

    lab = "(a)"
    bbox = ax1.get_tightbbox(fig.canvas.get_renderer())
    fig.text(bbox.x0, Y1.y1 + 50, lab, fontsize=22, va="top", ha="left",
             transform=None)

    Y = axes[0].get_tightbbox(fig.canvas.get_renderer())

    labels = ["(b)", "(c)", "(d)"]
    for a, label in zip(axes, labels):
        bbox = a.get_tightbbox(fig.canvas.get_renderer())
        fig.text(bbox.x0 - 60, Y.y1, label, fontsize=22, va="top", ha="left",
                 transform=None)
    plt.tight_layout(pad=5)
    if save_fig:
        fig.savefig("/home/shanybiton/AIMLabProjects/advanced_lab/figures/" + fig_name + ".png",
                    dpi=set_dpi)  # , bbox_inches='tight')

    # fig.show()
    plt.close()


def plot_flag_AF_hist(flag_hist, flags, c, N, save_fig=False, fig_name=None, set_dpi=400):
    # consts
    fig = plt.figure(dpi=set_dpi, figsize=(12, 8))

    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # regular histogram
    theta, width = np.linspace(0.0, 2 * np.pi, N, endpoint=False, retstep=True)
    barWidth = 0.80
    r = range(24)
    times = pd.date_range("00:00", "23:59", freq="60min").time
    ax1 = fig.add_subplot()  # row 0 (top) spans all(3) columns

    flag_hist['AF'].plot(ax=ax1, kind='bar', edgecolor='white', width=barWidth, zorder=3, alpha=0.8)

    ax1.set_xticks(r)

    ax1.set_xticklabels([times[i].strftime('%H:%M') for i in range(24)])
    ax1.set_xlabel("Time [hour]", fontsize=16)
    ax1.tick_params(axis='y', which='major', labelsize=16, rotation=0)
    ax1.tick_params(axis='x', which='major', labelsize=16, rotation=45)
    ax1.legend(loc='upper left', ncol=1, fontsize=16)
    ax1.set_ylabel("count", fontsize=18)
    ax1.grid(axis='y', zorder=0)
    ax1.set_title(fig_name, fontsize=16)
    plt.tight_layout(pad=5)
    if save_fig:
        fig.savefig("/home/shanybiton/AIMLabProjects/advanced_lab/figures/" + fig_name + ".png",
                    dpi=set_dpi, transparent=True)  # , bbox_inches='tight')

    # fig.show()
    plt.close()


def med_enrichment_test(med_list, med_clusters):
    data = med_clusters[med_clusters["Medication"].isin(med_list)]
    print(
        "Medication" + "\t" + "Genes in input list w/ annotation" + "\t" + "Genes in input list" + "\t" + "Genes in background list w/ annotation" + "\t" + "Genes in background list" + "\t" + "p-value")
    prob_dict = {}
    for category in data.Medication.unique():
        prob_dict["category"] = {}
        k = len(data)  # input list (list of differentially abundant proteins -ANOVA<0.05-)
        x = len(data[data.Medication == category])  # num of genes in the input list with the annotation
        N = len(med_clusters)  # background number of genes.
        m = len(med_clusters[med_clusters.Medication == category])
        p = stats.hypergeom.sf(x, N, k, m)
        print(category + "\t" + str(x) + "\t" + str(k) + "\t" + str(m) + "\t" + str(N) + "\t" + str(p))


def age_per_cluster(data, out_dir, dpi=400):
    plt.style.use('seaborn-white')
    fig, axes = plt.subplots(figsize=(8, 8), dpi=dpi)
    pos = list(range(1, 7))
    parts = axes.violinplot([data[i].astype(float) for i in range(len(pos))], pos, widths=0.7,
                            showmeans=False,
                            showmedians=True, showextrema=True)
    axes.set_xticks(pos)
    axes.tick_params(axis="y", labelsize=26)
    axes.set_xticklabels(['C' + str(i) for i in range(1, 7)], fontsize=30)
    axes.set_ylabel('Age', fontsize=30)
    leg = axes.legend(['C' + str(i) + ": " + str(len(data[i-1])) for i in range(1, 7)], loc=3, ncol=6,
                      mode="expand",
                      bbox_to_anchor=(0., 1.02, 1., .102),
                      borderaxespad=0., facecolor='white', fontsize=16,
                      handlelength=0, handletextpad=0, fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "Age.png", transparent=True, dpi=dpi, )
    fig.clear()
    plt.close()


def stat_test(data1, data2, category, stat_test="ks"):
    stat_true = pd.DataFrame(columns={'category', 'statistic', 'pvalue'})
    if stat_test == "wilcoxon":
        print("performing Wilcoxon signed-rank test for " + str(category))
        true_res = stats.wilcoxon(data1, data2)[:]
        stat_true = stat_true.append({'category': category, 'statistic': true_res[0], 'pvalue': true_res[1]},
                                     ignore_index=True)
    if stat_test == "ks": # Kolmogorov-Smirnov
        print("performing Kolmogorov–Smirnov test for " + str(category))
        true_res = stats.kstest(data1, data2)[:]
        stat_true = stat_true.append({'category': category, 'statistic': true_res[0], 'pvalue': true_res[1]},
                                     ignore_index=True)
    x = -stat_true.groupby('category')['pvalue'].mean().sort_values(ascending=False)
    return stat_true, x


if __name__ == '__main__':
    windows = [60]
    db = RBAFDB_Parser(load_on_start=False)
    ids = db.parse_available_ids()
    META_df = pd.DataFrame(columns={'id_patient', 'Age'})
    META_df['id_patient'] = ids
    cluster_ids = pd.read_csv("/home/shanybiton/repos/CircadianAF/AF_patient_clusters.csv")
    for id in ids:
        META_df.loc[META_df['id_patient'].eq(id), 'Age'] = db.circadian_dict[id]['Age']

    cluster_ids['Age'] = cluster_ids['patient'].map(META_df.set_index(['id_patient'])['Age'])
    cluster_ids['Sex'] = cluster_ids['patient'].map(META_df.set_index(['id_patient'])['Gender'])

    C = [x.Age.values for _, x in cluster_ids.groupby(cluster_ids['cluster'].astype(float))]
    age_per_cluster(C, pathlib.PurePath("/MLdata/AIMLab/Shany/ErrorAnalysis/rbafdb/"), dpi=400)
    df_windows = pd.read_csv("/home/shanybiton/repos/CircadianAF/output/ArNet2_circAF_pred.csv", index_col=None)
    df = pd.DataFrame([])
    for i, j in list(itertools.combinations(range(6), 2)):
        category = str(j+1) + ' vs ' + str(i+1)
        stat_true, x = stat_test(C[j], C[i], category, stat_test="ks")
        df = df.append(stat_true)
    med_list = np.array(['PROPAFENONE HCL', 'FLECAINIDE ACET', 'AMIODARONE HCL',
                         'Flecainide-Tab'])
    EMR_list = pd.read_excel(cts.DATA_DIR / "documentation" / "holter_list_research.xlsx", engine='openpyxl')
    EMR_use = EMR_list[EMR_list['Patient ID'].isin(db.excel_sheet['ID_NO'])]
    EMR_use = EMR_use.drop_duplicates(['Patient ID'])
    EMR_use['Path ID'] = EMR_use['Patient ID'].map(
        db.excel_sheet.drop_duplicates(['ID_NO']).set_index(['ID_NO'])['Path ID'])
    EMR_use = EMR_use[EMR_use['Path ID'].isin(cluster_ids.patient)]
    med_clusters = EMR_use[EMR_use['Anti-arrhythmic drugs-Medication'].notnull()]
    med_windows = df_windows.loc[df_windows.patient_id.isin(med_clusters['Path ID'])]
    med_list = np.array(['PROPAFENONE HCL', 'FLECAINIDE ACET', 'AMIODARONE HCL',
                         'Flecainide-Tab'])
    med_windows['medication'] = med_windows['patient_id'].map(
        EMR_use.set_index(['Path ID'])['Anti-arrhythmic drugs-Medication'])
    c = ['#16697a', '#db6400', '#ffa62b']
    N = 24
    for med in med_list:
        temp_df = med_windows.loc[med_windows.medication.eq(med)]
        flag_hist, _ = get_shape_hist_AF(temp_df)
        plot_flag_AF_hist(flag_hist, flags=['AF'], c=c, N=N, set_dpi=400, fig_name=str(med), save_fig=True)

    #
    # flags = ['ABERRANT', 'INHIBIT', 'PACED']
    # c = ['#16697a', '#db6400', '#ffa62b']
    # N = 24
    # circadian_df= pd.DataFrame.from_dict(db.circadian_dict, orient='index')
    # AF_pat = db.excel_sheet[~db.excel_sheet['Diagnosis'].isna()]["Path ID"].values
    # AF_ids = circadian_df[circadian_df.index.isin(AF_pat)].index.values
    # non_Af_ids = circadian_df[~circadian_df.index.isin(AF_pat)].index.values
    # flag_hist, population_ABERRANT_df, population_INHIBIT_df, population_PACED_df = get_shape_hist(AF_ids, circadian_df)
    # flag_hist_non_AF, population_ABERRANT_df_non_AF, population_INHIBIT_df_non_AF, population_PACED_df_non_AF = get_shape_hist(
    #     non_Af_ids, circadian_df)
    # fit_cosinor(flag_hist, flags=flags, c=c, fig_name="cosinor_fit_AF", set_dpi=400)
    # fit_cosinor(flag_hist_non_AF, flags=flags, c=c, fig_name="cosinor_fit_non_AF", set_dpi=400)
    #
    # fit_param = fit_von_mises(population_PACED_df, population_INHIBIT_df, population_ABERRANT_df, c=c, save_fig=True, fig_name="von_mises_fit_AF")
    # fit_param_2 = fit_von_mises(population_PACED_df_non_AF, population_INHIBIT_df_non_AF, population_ABERRANT_df_non_AF, c=c, save_fig=True, fig_name="von_mises_fit_non_AF")
    #
    # plot_date_hist(circadian_df)
    # plot_flag_hist(flag_hist, flags=flags, c=c, N=N, set_dpi=400, fig_name="circular_beat_diurnal_3_hist_AF", save_fig=True)
    # plot_flag_hist(flag_hist_non_AF, flags=flags, c=c, N=N, set_dpi=400, fig_name="circular_beat_diurnal_3_hist_non_AF", save_fig=True)
    #
    #
    # #circadian analysis of beat shape
    # #
    # #pd.to_timedelta(circadian_df.start_recording, unit="s")
