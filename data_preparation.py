import math
import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import resample
from skimage.transform import resize
pd.set_option('precision', 10)


# %% import files
def files_modes_subjects_oh_my(search_path):
    files = os.listdir(search_path)
    ja_files = list(filter(lambda x: 'jar' in x, files))
    emg_files = list(filter(lambda x: 'emg' in x, files))
    sbj = []
    mode = []
    for idx in range(0, len(ja_files)):
        sbj.append(ja_files[idx][0:3])
        mode.append(ja_files[idx][10])
    return ja_files, emg_files, sbj, mode


# %% finding peaks
# ja_files, _, _, _ = files_modes_subjects_oh_my(r'C:\Users\randh\Desktop\SpringExo\processed')
# idx=12
# peak_w =50
# peak_d=45
# j_a_file = pd.read_csv(ja_files[idx], header=3, skiprows=[4])
# rthi = j_a_file['RTHIZ']
# indices = find_peaks(j_a_file['RTHIZ'], width=peak_w, distance=peak_d)[0]
# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     y=rthi,
#     mode = 'lines+markers',
#     name='Thigh Marker for Segmentation',
# ))
# fig.add_trace(go.Scatter(
#     x=indices,
#     y= [rthi[j] for j in indices],
#     mode='markers',
#     marker= dict(
#         size=8,
#         color='red',
#         symbol='cross'
#     ),
#     name='DetectedPeaks'
# ))
# fig.update_layout(title=ja_files[idx] + ' peak_w:' + str(peak_w) + ' peak_d:' + str(peak_d))
# fig.show()

# %% Angles and EMG
def angle_between(df_in):
    thigh_p = df_in[['RTHIZ', 'RTHIY']].to_numpy()
    knee_p = df_in[['RKNEZ', 'RKNEY']].to_numpy()
    ankle_p = df_in[['RANKZ', 'RANKY']].to_numpy()
    vector_1 = thigh_p - knee_p
    vector_2 = knee_p - ankle_p
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1, axis=1)[:, None]
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2, axis=1)[:, None]
    dot_product = np.einsum('ij,ij->i', unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    df_in['Angles'] = angle * 180 / math.pi
    return df_in

def segment_emg(emg_df, indices):
    emg_idx = []
    for idx in indices:
        match_idx = emg_df['Frame'].isin([idx])
        last_emg_idx = list(match_idx[match_idx == True].index)[-1]
        emg_idx.append(last_emg_idx)
    a = 0
    emg_idx.insert(0, a)
    emg_idx.append(len(emg_df))
    return emg_idx


def rename_columns(emg_df_in):
    emg_df_out = emg_df_in.rename(columns={'EMG1': 'RRF', 'EMG2': 'LRF', 'EMG3': 'RBF', 'EMG4': 'LBF'})
    return emg_df_out


def append_df(emg_df, ja_df, peak_widths, peak_distances, sbjstr, modestr):
    indices = find_peaks(ja_df['RTHIZ'], width=peak_widths, distance=peak_distances)[0]
    emg_df = rename_columns(emg_df)
    emg_df = emg_df.abs()
    angle_between(ja_df)
    emg_idx = segment_emg(emg_df, ja_df.Frame[indices])
    cycle = np.zeros(emg_idx[-1])
    a = 0
    time_out = np.array([])
    for idx in range(len(emg_idx) - 1):
        a = a + 1
        cycle[emg_idx[idx]: emg_idx[idx + 1]] = a
        time_cyc = np.linspace(0, (emg_idx[idx + 1] - emg_idx[idx]) * 0.00025, num=emg_idx[idx + 1] - emg_idx[idx])
        time_out = np.append(time_out, time_cyc)
    df_out = pd.merge(emg_df, ja_df, on=['Frame', 'Frame'])
    df_out['Subject'] = sbjstr
    df_out['Mode'] = modestr
    df_out['Cycle'] = cycle
    _, counts = np.unique(cycle, return_counts=True)
    tot_cyc_time = np.array([])
    for idxc in range(len(counts)):
        cyc_time = np.linspace(0, 100, num=counts[idxc])
        tot_cyc_time = np.append(tot_cyc_time, cyc_time)
    df_out['Cycle %'] = tot_cyc_time
    df_out['Time'] = time_out
    return df_out


# %% read files
def make_giant_df(ja_files_in, emg_files_in, sbj_in, modes_in):
    appended_data = []
    peak_ws = [20, 20, 20, 20, 10, 50, 50, 50, 50, 50, 50, 50, 50]
    peak_ds = [10, 10, 10, 10, 30, 50, 45, 45, 45, 45, 45, 45, 45]
    for idx in range(len(ja_files_in)):
        j_a_file = pd.read_csv(ja_files_in[idx], header=3, skiprows=[4])
        emg_file = pd.read_csv(emg_files_in[idx], header=3, skiprows=[4])
        data = append_df(emg_file, j_a_file, peak_ws[idx], peak_ds[idx], sbj_in[idx], modes_in[idx])
        appended_data.append(data)
    all_df = pd.concat(appended_data)
    return all_df

def normalize_emg(df_in, sbjstr):
    sbj_df = df_in[(df_in['Subject'] == sbjstr)].copy()
    b_df = sbj_df[(sbj_df['Mode'] == 'b')].copy()
    p_df = sbj_df[(sbj_df['Mode'] == 'p')].copy()
    max_rrf = b_df.loc[:, 'RRF'].dropna().max()
    max_lrf = b_df.loc[:, 'LRF'].dropna().max()
    max_rbf = b_df.loc[:, 'RBF'].dropna().max()
    max_lbf = b_df.loc[:, 'LBF'].dropna().max()

    b_df.loc[:, 'RRF'] = b_df.RRF.abs().to_numpy() / max_rrf
    b_df.loc[:, 'LRF'] = b_df.LRF.abs().to_numpy() / max_lrf
    b_df.loc[:, 'RBF'] = b_df.RBF.abs().to_numpy() / max_rbf
    b_df.loc[:, 'LBF'] = b_df.LBF.abs().to_numpy() / max_lbf
    p_df.loc[:, 'RRF'] = p_df.RRF.abs().to_numpy() / max_rrf
    p_df.loc[:, 'LRF'] = p_df.LRF.abs().to_numpy() / max_lrf
    p_df.loc[:, 'RBF'] = p_df.RBF.abs().to_numpy() / max_rbf
    p_df.loc[:, 'LBF'] = p_df.LBF.abs().to_numpy() / max_lbf
    df_out = pd.concat([b_df, p_df])
    print(sbjstr)
    return df_out

def relevant_var_df(filepath):
    ja_files, emg_files, sbj, mode = files_modes_subjects_oh_my(filepath)
    appended_data = []
    df_all = make_giant_df(ja_files, emg_files, sbj, mode)
    sbj_no_dup = list(dict.fromkeys(sbj))
    for idx in range(len(sbj_no_dup)):
        data = normalize_emg(df_all, sbj_no_dup[idx])
        appended_data.append(data)
    df_out = pd.concat(appended_data)
    df_out = df_out[['Cycle %', 'Time', 'RRF', 'LRF', 'RBF', 'LBF', 'Angles', 'Cycle', 'Subject', 'Mode']]
    return df_out


# %% Main
alldf = relevant_var_df(r'C:\Users\randh\Desktop\SpringExoSquat\processed')

# %%Save as pickle
alldf.to_pickle("./allData.pkl")

#%% FUNCTIONS
def add_stuff_to_some(df_in, string):
    keep_same = {'Subject', 'Mode', 'Cycle'}
    df_in.columns = ['{}{}'.format(c, '' if c in keep_same else string) for c in df_in.columns]
    return df_in


def makestatsdf(grouped_df):
    group_max = grouped_df.max()
    group_min = grouped_df.min()
    group_avg = grouped_df.mean()
    group_q25 = grouped_df.quantile(0.25)
    group_q75 = grouped_df.quantile(0.75)
    group_std = grouped_df.std()
    group_max = add_stuff_to_some(group_max, '_max')
    group_min = add_stuff_to_some(group_min, '_min')
    group_avg = add_stuff_to_some(group_avg, '_avg')
    group_q25 = add_stuff_to_some(group_q25, '_q25')
    group_q75 = add_stuff_to_some(group_q75, '_q75')
    group_std = add_stuff_to_some(group_std, '_std')
    frames = [group_avg, group_max, group_min, group_q25, group_q75, group_std]
    result = pd.concat(frames, axis=1, sort=False)
    result = result.loc[:, ~result.columns.duplicated()]
    return result


def makeseaborndf(grouped_df):
    group_max = grouped_df.max()
    group_max = group_max[group_max.RRF !=0]
    group_max = group_max[group_max.LRF !=0]
    group_max = group_max[group_max.RBF !=0]
    group_max = group_max[group_max.LBF !=0]
    group_max["Stat"] = "Max"
    group_min = grouped_df.min()
    group_min = group_min[group_min.RRF != 0]
    group_min = group_min[group_min.LRF != 0]
    group_min = group_min[group_min.RBF != 0]
    group_min = group_min[group_min.LBF != 0]
    group_min["Stat"] = "Min"
    group_avg = grouped_df.mean()
    group_avg = group_avg[group_avg.RRF !=0]
    group_avg = group_avg[group_avg.LRF !=0]
    group_avg = group_avg[group_avg.RBF !=0]
    group_avg = group_avg[group_avg.LBF !=0]
    group_avg["Stat"] = "Avg"
    group_q25 = grouped_df.quantile(0.25)
    group_q25 = group_q25[group_q25.RRF !=0]
    group_q25 = group_q25[group_q25.LRF !=0]
    group_q25 = group_q25[group_q25.RBF !=0]
    group_q25 = group_q25[group_q25.LBF !=0]
    group_q25["Stat"] = "25%"
    group_q75 = grouped_df.quantile(0.75)
    group_q75= group_q75[group_q75.RRF !=0]
    group_q75= group_q75[group_q75.LRF !=0]
    group_q75= group_q75[group_q75.RBF !=0]
    group_q75= group_q75[group_q75.LBF !=0]
    group_q75["Stat"] = "75%"
    group_std = grouped_df.std()
    group_std= group_std[group_std.RRF !=0]
    group_std= group_std[group_std.LRF !=0]
    group_std= group_std[group_std.RBF !=0]
    group_std= group_std[group_std.LBF !=0]
    group_std["Stat"] = "Std"
    result = pd.concat([group_avg, group_max, group_min, group_q25, group_q75, group_std],
                       axis=0, sort=False)
    result = result.loc[:, ~result.columns.duplicated()]
    return result

# %% STANDARDIZING CYCLES TO 500
def makeSbjModeDF(df_in, sbjstr, modestr):
    sbj_mode_df = df_in[(df_in['Subject'] == sbjstr) & (df_in['Mode'] == modestr)]
    return sbj_mode_df

def cut_cycles(df_in, cycle):
    num =500
    cycle_df = df_in[df_in['Cycle'] == cycle]
    numeric_cols = ['Time', 'RRF', 'LRF', 'RBF', 'LBF', 'Angles']
    df_out  = df_resample(cycle_df[numeric_cols], num)
    df_out['Cycle %'] = np.linspace(0, 100, num)
    df_out['Cycle'] = cycle
    return df_out

def makesmallerdf(df_in, sbjstr, modestr):
    sbj_mode_df = makeSbjModeDF(df_in, sbjstr, modestr)
    cycles = sbj_mode_df.Cycle.drop_duplicates().to_numpy()
    appended_df = []
    for c, value in enumerate(cycles):
            cycle_df = sbj_mode_df[sbj_mode_df['Cycle']==value]
            resized_df = cut_cycles(cycle_df, value)
            resized_df['Subject'] = sbjstr
            resized_df['Mode'] = modestr
            appended_df.append(resized_df)
    appended_df = pd.concat(appended_df)
    return appended_df


#%% RESAMPLING
def df_resample(df1, num=1):
    df2 = pd.DataFrame()
    for key, value in df1.iteritems():
        temp = value.to_numpy()/value.abs().max() # normalize
        resampled = resize(temp, (num,1), mode='edge')*value.abs().max() # de-normalize
        df2[key] = resampled.flatten().round(2)
    return df2

# %%
def make_regular_length_df(df_in):
    mode = ['b', 'p']
    sbj = ['s01', 's02', 's03', 's04', 's05', 's06', 's07']
    appended_df = []
    for c1, mode_val in enumerate(mode):
        for c2, sbj_val in enumerate(sbj):
            if ((sbj_val == 's02') and (mode_val == 'p')):
                continue
            appended_df.append(makesmallerdf(df_in, sbj_val, mode_val))
    appended_df = pd.concat(appended_df).fillna(0.0)
    return appended_df


#%% CYCLES
procdf = pickle.load(open('allData.pkl', 'rb'))
procdf = procdf[procdf['Cycle'] <26]
procdf = procdf[procdf['Cycle'] > 4]
procdf.to_pickle("./allData20.pkl")

procdf500 = make_regular_length_df(procdf)
procdf500.to_pickle("./allData100.pkl")

# %% STATISTICS
stats_df = makestatsdf(procdf500.groupby(['Subject', 'Mode', 'Cycle'], as_index=False))
stats_summarized_df = makestatsdf(procdf500.groupby(['Mode', 'Subject'], as_index=False))
seaborndf = makeseaborndf(procdf500.groupby(['Subject', 'Mode', 'Cycle'], as_index=False))
seaborndf=seaborndf[seaborndf['Cycle']<26]
seaborndf=seaborndf[seaborndf['Cycle']>4]

# %% SAVE STATS
seaborndf.to_pickle("./seaborndf.pkl")
stats_df.to_csv('./statsSPSS.csv', sep=',')
stats_summarized_df.to_csv('./stats_summarySPSS.csv', sep=',')

# %% FOOT PRESSUSRE
foot_df = pd.read_excel('FootPressure.xlsx').dropna()
foot_df.to_pickle("./foot.pkl")