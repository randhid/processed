import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats.stats import pearsonr
from statannot import add_stat_annotation

# %% LOAD STATS
seaborndf = pickle.load(open("./seaborndf.pkl", 'rb'))
cycles_df = pickle.load(open("./allData100.pkl", 'rb'))
foot_df = pickle.load(open("./foot.pkl", 'rb'))


# %% FUNCTIONS FOR PLOTTING AND FINDING CORRELATIONS
def squat_plots(df_in, var_in, pvalues_in=[0], new_ylabel='Same'):
    boxpairs = [
        (('Avg', 'b'), ('Avg', 'p')),
        (('Max', 'b'), ('Max', 'p')),
        (('Min', 'b'), ('Min', 'p')),
        (('25%', 'b'), ('25%', 'p')),
        (('75%', 'b'), ('75%', 'p')),
        (('Std', 'b'), ('Std', 'p')),
    ]
    plot_out = sns.pointplot(x='Stat', y=var_in, hue='Mode', join=False, dodge=0.25,
                             data=df_in, split=True, palette="dark", ci='sd')

    add_stat_annotation(plot_out, data=df_in, x='Stat', y=var_in, hue='Mode',
                        box_pairs=boxpairs,
                        test='Mann-Whitney', comparisons_correction=None,
                        # perform_stat_test=False, pvalues=pvalues_in,
                        text_format='star', loc='inside', verbose=1,
                        pvalue_thresholds=[[1e-4, "**"], [1e-3, "**"], [1e-2, "**"], [0.05, "*"], [1, ""]]
                        )
    plot_out.set_ylabel(new_ylabel)
    plot_paper_params()
    return plot_out


def squats_lineplts(df_in, var_in, new_ylabel='Same') -> object:
    plot_out = sns.lineplot(x='Cycle %', y=var_in, hue='Mode', ci='sd',
                            data=df_in, palette="dark")
    plot_out.set_xlabel('Cycle %')
    plot_out.set_ylabel(new_ylabel)
    return plot_out


def foot_plots(df_in, var_in):
    boxpairs = [
        (('Early', 'b'), ('Early', 'p')),
        (('Mid', 'b'), ('Mid', 'p')),
        (('Late', 'b'), ('Late', 'p')),
    ]
    plot_out = sns.pointplot(x='Timing', y=var_in, hue='mode', join=False, dodge=0.25,
                             data=df_in, split=True, palette="dark", ci='sd')
    add_stat_annotation(plot_out, data=df_in, x='Timing', y=var_in, hue='mode',
                        box_pairs=boxpairs,
                        test='Mann-Whitney', comparisons_correction=None,
                        # perform_stat_test=False, pvalues=pvalues_in,
                        text_format='star', loc='inside', verbose=1,
                        pvalue_thresholds=[[1e-4, "**"], [1e-3, "**"], [1e-2, "**"], [0.05, "*"], [1, ""]]
                        )

    #    plot_out.set_ylabel(new_ylabel)
    plot_paper_params()


def plot_paper_params():
    sns.set_context("paper")
    sns.despine()
    params = {'legend.fontsize': 10,
              'legend.handlelength': 1,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'axes.labelsize': 10,
              'axes.titlesize': 10,
              'font.family': 'Times New Roman',
              'font.size': 10,
              "savefig.format": 'pdf'
              }
    plt.rcParams.update(params)


def remove_legend():
    ax = plt.gca()
    ax.get_legend().remove()


def show_and_save(fig_name, legend_show='No'):
    fig = plt.gcf()
    fig.set_size_inches(4.0, 3.0)
    if legend_show == 'No':
        remove_legend()
    fig.savefig(fig_name, bbox_inches='tight')
    plt.show()


def find_var_corr(df_in, var_in):
    var_p = df_in[df_in['Mode'] == 'p'].groupby('Cycle %').mean()[var_in]
    var_b = df_in[df_in['Mode'] == 'b'].groupby('Cycle %').mean()[var_in]
    corr = pearsonr(var_b, var_p)
    return corr


# %% Categorical PLOTS
squat_plots(seaborndf, 'Time', [.008, .008, 1, .011, .008, .008], 'Time (s)')
show_and_save('time_err.pdf', 'Yes')
squat_plots(seaborndf, 'RRF', [.001, 1, .003, .002, .001, .025], 'nRRF % (V/V)')
show_and_save('rrf_err')
squat_plots(seaborndf, 'LRF', [1, .031, .003, .002, .001, .025], 'nLRF % (V/V)')
show_and_save('lrf_err')
squat_plots(seaborndf, 'RBF', [1, .001, 1, 1, 1, .016], 'nRBF % (V/V)')
show_and_save('rbf_err')
squat_plots(seaborndf, 'LBF', [.000, .000, .005, .000, .002, 1], 'nLBF % (V/V)')
show_and_save('lbf_err')
squat_plots(seaborndf, 'Angles', [.047, .000, 1, 1, .002, .000], 'KneeF$^\circ$')
show_and_save('angles_err')

# %% LINEPLOTS
squats_lineplts(cycles_df, 'Time', 'Time (s)')
show_and_save('time_line', 'Yes')
squats_lineplts(cycles_df, 'RRF', 'nRRF % (V/V)')
show_and_save('rrf_line')
squats_lineplts(cycles_df, 'LRF', 'nLRF % (V/V)')
show_and_save('lrf_line')
squats_lineplts(cycles_df, 'RBF', 'nRBF % (V/V)')
show_and_save('rbf_line')
squats_lineplts(cycles_df, 'LBF', 'nLBF % (V/V)')
show_and_save('lbf_line')
squats_lineplts(cycles_df, 'Angles', 'KneeF$^\circ$')
show_and_save('angles_line')


# %% Correlation Coeffications
corr_time = find_var_corr(cycles_df, 'Time')
corr_angles = find_var_corr(cycles_df, 'Angles')
corr_RRF = find_var_corr(cycles_df, 'RRF')
corr_LRF = find_var_corr(cycles_df, 'LRF')
corr_RBF = find_var_corr(cycles_df, 'RBF')
corr_LBF = find_var_corr(cycles_df, 'LBF')


# %% FOOT PLOTS
foot_plots(foot_df, 'Ball-Ball Distance (cm)')
show_and_save(".\B-BDist", 'Yes')
foot_plots(foot_df, 'Heel-Heel Distance (cm)')
show_and_save(".\H-HDist")
foot_plots(foot_df, 'Loading Line L (cm)')
show_and_save(".\LoadLineLeft")
foot_plots(foot_df, 'Loading Line L (cm)')
show_and_save(".\LoadlineRight")


# %% Foot maps
def foot_plots_loading(df_in, x_in, y_in):
    boxpairs = {
        (('Early', 'b'), ('Early', 'p')),
        (('Mid', 'b'), ('Mid', 'p')),
        (('Late', 'b'), ('Late', 'p')),
        (('Early', 'p'), ('Mid', 'p')),
        (('Early', 'p'), ('Late', 'p')),
        (('Mid', 'p'), ('Late', 'p')),
        (('Early', 'b'), ('Mid', 'b')),
        (('Early', 'b'), ('Late', 'b')),
        (('Mid', 'b'), ('Late', 'b')),
    }
    plot_out = sns.pointplot(x=x_in, y=y_in, hue='mode', join=False, dodge=0.25,
                             data=df_in, split=True, palette="dark", ci='sd')
    add_stat_annotation(plot_out, data=df_in, x=x_in, y=y_in, hue='mode',
                        box_pairs=boxpairs,
                        test='Man-Whitney', comparisons_correction=None,
                        # perform_stat_test=False, pvalues=pvalues_in,
                        text_format='star', loc='inside', verbose=1,
                        pvalue_thresholds=[[1e-4, "**"], [1e-3, "**"], [1e-2, "**"], [0.05, "*"], [1, ""]]
                        )

    #    plot_out.set_ylabel(new_ylabel)
    plot_paper_params()


def foot_err_plots(df_in, x_var, y_var, colour_in, mult= 1):
    x = df_in[x_var].mean()-15
    y = mult*df_in[y_var].mean()
    xe = df_in[x_var].std()
    ye = df_in[y_var].std()
    ax = plt.errorbar(x, y, xerr=xe, yerr=ye, color=colour_in, marker='o')



#%% FOOT ERR PLOTS
foot_err_plots(foot_df[foot_df['mode']=='b'], 'xLBallMax (cm)', 'yLBallMax (cm)', (0.0, 0.10980392156862745, 0.4980392156862745))
foot_err_plots(foot_df[foot_df['mode']=='b'], 'xRBallMax (cm)', 'yRBallMax (cm)', (0.0, 0.10980392156862745, 0.4980392156862745))
foot_err_plots(foot_df[foot_df['mode']=='b'], 'xLHeelMax (cm)', 'yLHeelMax (cm)', (0.0, 0.10980392156862745, 0.4980392156862745), -1)
foot_err_plots(foot_df[foot_df['mode']=='b'], 'xRHeelMax (cm)', 'yRHeelMax (cm)', (0.0, 0.10980392156862745, 0.4980392156862745), -1)

foot_err_plots(foot_df[foot_df['mode']=='p'], 'xLBallMax (cm)', 'yLBallMax (cm)', (0.6941176470588235, 0.25098039215686274, 0.050980392156862744))
foot_err_plots(foot_df[foot_df['mode']=='p'], 'xRBallMax (cm)', 'yRBallMax (cm)', (0.6941176470588235, 0.25098039215686274, 0.050980392156862744))
foot_err_plots(foot_df[foot_df['mode']=='p'], 'xLHeelMax (cm)', 'yLHeelMax (cm)', (0.6941176470588235, 0.25098039215686274, 0.050980392156862744), -1)
foot_err_plots(foot_df[foot_df['mode']=='p'], 'xRHeelMax (cm)', 'yRHeelMax (cm)', (0.6941176470588235, 0.25098039215686274, 0.050980392156862744), -1)

ax= plt.gca()
ax.set_ylabel('y Position (cm)')
ax.set_xlabel('x Position (cm)')
plot_paper_params()
show_and_save(".\Footshift", 'Yes')