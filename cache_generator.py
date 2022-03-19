import pandas as pd
import glob
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from os import path
import json
from shutil import copyfile
import random
import time

from scipy.optimize import curve_fit
import tensorflow as tf
import pickle
import multiprocessing
from multiprocessing import Pool
from pandarallel import pandarallel
        
    
import sys
sys.path.append("imports/")
import Util
from NRTables import MCS, PC_TBS, UP_MCS_1, UP_MCS_2, BIDIR_MCS





def quant(q):
    def n_quantile(x):
        return x.quantile(q)
    n_quantile.__name__ = f"{q*100:.0f}%"
    return n_quantile

def no_sort_quant(q, interpolate=True):
    def no_sort_quantile_inner(x):
        sumed = x.cumsum()
        idx = q*len(x)
        c = np.ceil(idx).astype(int)
        f = np.floor(idx).astype(int)
        if f == c or not interpolate:
            return sumed.iloc[c]
        else:
            return sumed.iloc[c]*(c-idx) + sumed.iloc[f]*(idx-f)
    no_sort_quantile_inner.__name__ =  f"{q*100:.0f}%"
    return no_sort_quantile_inner

def get_usable_UE_usages(df):
    usage_per_time = df.groupby("Time").agg({"UE usage": "sum"})
    return (usage_per_time > 0).sum()["UE usage"]

def rolling_function(slice_df, future_df):
    if slice_df.empty:
        return pd.DataFrame()
    else:
        #window_size = slice_df.Time.nunique()
        window_size = get_usable_UE_usages(slice_df)
        return_df = slice_df.copy()
        return_df["Time"] =  return_df.Time.min()
        return_df = return_df.sort_values("SNR")
        return_df = return_df.groupby("Time")
        return_df = return_df.agg({
            'Bytes': ['std',  no_sort_quant(0.05), no_sort_quant(0.25), no_sort_quant(0.5), no_sort_quant(0.75), no_sort_quant(0.95)], 
            'SNR': ['std', quant(0.05), quant(0.25), 'median', quant(0.75), quant(0.95)], 
            'Equally distributed': 'mean', 
            'UE usage': "sum", # the sum divided by the window size gives the average usage over time,
            'UE_SE': 'sum'
        })
        #bef = len(return_df[return_df["UE usage"]["sum"].isna()])
        return_df["UE usage"] = (return_df["UE usage"] / window_size)#.fillna(max_se)
        #print(bef, len(return_df[return_df["UE usage"]["sum"].isna()]))
        return_df["Used"] = future_df["UE_SE"].sum() / get_usable_UE_usages(future_df)
        return return_df.reset_index()

def get_extended_metric_df(in_df, tm, tm_source=None, min_mcs=None, max_static=None):
    df = in_df.copy()
    if min_mcs is not None:
        df = df[df.MCS >= min_mcs]
    if max_static is not None:
        # consider nodes that are either not static, or are within the max_static parameter
        considered_nodes_mask = (~df.Node.str.contains("sta")) | (pd.to_numeric(df.Node.str[3:-9],  errors='coerce') < max_static)
        df = df[considered_nodes_mask]
    df["UE usage"] = (df.UERatio * df.Bytes  / df.RBs ).replace(np.inf,np.nan).fillna(0)
    df["UE_SE"] = df.UERatio * df.SE
    df["Equally distributed"] = df.RatioHypothetical
    if tm_source is None:
        times=df.Time.unique()
        times.sort()
        orig_tm = np.diff(times).min()
    else:
        orig_tm = tm_source
    window_size = int(tm // orig_tm)
    assert orig_tm <= tm and tm % orig_tm == 0, f"Specified monitoring period not compatible with sample period. Origin: {orig_tm}, specified: {tm}"
    time_values = np.arange(df.Time.min(), df.Time.max()-tm, orig_tm)
    time_values = list(map((lambda x: [x, x+tm]), time_values))
    masks = []
    for t in time_values:
        masks.append((df.Time >= t[0]) & (df.Time < t[1]))
    extended_df = pd.DataFrame()
    for i in range(len(masks) - window_size):
        present_mask = masks[i]
        future_mask = masks[i+window_size]
        extended_df = extended_df.append(rolling_function(df.loc[present_mask], df.loc[future_mask]), ignore_index=True)
    extended_df.columns = ["Time", "Bytes_std", "Bytes_5", "Bytes_25","Bytes_50", "Bytes_75", "Bytes_95", 
                           "SNR_std", "SNR_5", "SNR_25","SNR_50", "SNR_75", "SNR_95", 
                           "Last_equally_distributed", "Last_RB_ratio", "Last_SE", "RB_ratio"]
    return extended_df

def get_extended_metric_df_for_all_cells(
    tm,
    tm_source=10,
    direction='D',
    toi=[0,23200],
    min_mcs=None,
    max_static=None,
    cells=range(0,21),
    cache_folder="processed",
    traffic_scenario="standard",
    cache=True,
    mega_cache_folder="megas",
    sliced=False
):
    if sliced:
        fname = f"{mega_cache_folder}/mega_block_{direction}_{traffic_scenario}_{tm}_{min_mcs}_{max_static}_sliced.parquet"
    else:
        fname = f"{mega_cache_folder}/mega_block_{direction}_{traffic_scenario}_{tm}_{min_mcs}_{max_static}.parquet"
    if cache and path.exists(fname):
        df = pd.read_parquet(fname)
        return df
    elif cache:
        print(fname, "not cached. Processing...")
    mega_block_df = None
    for cell in range(21):
        cell_df = process_monitoring_intervals(cell=cell, direction=direction, tm=tm_source, toi=toi, cache_folder=cache_folder, traffic_scenario=traffic_scenario, silent_cache=True)
        if sliced:
            #Inefficient code, but the same function is applied to sliced and non-sliced dataframes
            cell_df = pd.Dataframe()
            for nwslice in cell_df.Slice.unique():
                nwslice_df = cell_df[cell_df.Slice == nwslice]
                nwslice_df = get_extended_metric_df(cell_df,tm, tm_source=tm_source, min_mcs=min_mcs, max_static=max_static)
                nwslice_df["Slice"] = nwslice
                cell_df.append(nwslice_df , ignore_index=True)
            cell_df = get_extended_metric_df(cell_df, tm, tm_source=tm_source, min_mcs=min_mcs, max_static=max_static)
        else:
            cell_df = get_extended_metric_df(cell_df, tm, tm_source=tm_source, min_mcs=min_mcs, max_static=max_static)
        cell_df["Cell"] = cell
        if mega_block_df is None:
            mega_block_df = cell_df
        else:
            mega_block_df = mega_block_df.append(cell_df, ignore_index=True)
    
    if cache:
        mega_block_df.to_parquet(fname)
    
    
    return mega_block_df


def from_snr_to_traffic(filename, traffic_scenario="standard"):
    f_obj = Path(filename)
    stem = f_obj.stem
    traffic_name = "-".join(stem.split("-")[:-1]) + "-traffic.parquet.gz"
    if traffic_scenario is not None:
        traffic_folder = f"traffic-traces-{traffic_scenario}" 
    else:
        traffic_folder = "traffic_traces"
    traffic_full = str(Path("..","traffic_generator",traffic_folder,traffic_name))
    if path.exists(traffic_full):
        return traffic_full
    else:
        return None

def node_from_name(filename):
    stem = Path(filename).stem
    name_array = stem.split("-")
    if len(name_array) != 8:
        return None
    else:
        return "".join([name_array[2], name_array[4], name_array[5], name_array[6]])
    
    
def process_monitoring_intervals(
    direction = 'D',
    cell = 18,
    tm = 60,
    toi = [0, 3600],
    traffic_scenario="standard",
    min_snr = None,
    min_mcs = None,
    max_iter = None,
    cache_df = True,
    cache_cells = True,
    cache_folder = "processed",
    trace_folder = "../sinr-map/small-berlin-long-out/trace-*-snr.parquet.gz",
    remove_filter = None,
    custom_name = "",
    static_limit = 5000,
    write_cell_cache=False,
    silent_cache=False
):
    assert cache_folder is not None, "POOLERROR"
    filename = f"{cache_folder}/aggregated_{cell}_{direction}_{toi[0]}_{toi[1]}_{tm}_{traffic_scenario}_minmcs_{min_mcs}.parquet"
    if custom_name:
        filename = f"{cache_folder}/aggregated_{cell}_{direction}_{toi[0]}_{toi[1]}_{tm}_{traffic_scenario}_minmcs_{min_mcs}-{custom_name}.parquet"

    if cache_df:
        try:
            df = pd.read_parquet(filename)
            if not silent_cache:
                print(filename, "cached!")
            return df
        except:
            print(filename, "not cached. Processing")
            pass
        
    #cell_json_file = "files_per_cell.json"
    cell_json_file = "files_per_cell.pickle"
    if cache_cells:
        try:
            with open(cell_json_file,'rb') as json_file:
                cells_per_file = pickle.load(json_file)
                copyfile(cell_json_file, cell_json_file + ".bak")
        except FileNotFoundError:
            cells_per_file = {}

    progress_update_size = 0.25
    last_progress_update = 0
    files = glob.glob(trace_folder)
    if remove_filter:
        files = list(filter((lambda x: remove_filter not in x), files))
    total = max_iter if max_iter else len(files)
    last_time = time.time()
    i=-1
    static_count = 0
    aggregate_columns = ["Time", "Node", "Bytes", "SNR", "Slice"]
    aggregate_df = pd.DataFrame(columns=aggregate_columns)
    for snr_file in  files:
        i += 1
        if static_limit is not None and "sta" in snr_file:
            static_no = int(Path(snr_file).name.split("-")[2].split("sta")[1])
            if static_no > static_limit:
                continue
        if cache_cells and snr_file in cells_per_file and cell not in cells_per_file[snr_file]:
            continue
        node = node_from_name(snr_file)
        progress = i / total
        this_progress_update = progress // progress_update_size
        if this_progress_update != last_progress_update:
            this_time = time.time()
            print(f"Progress: {progress*100:.0f}%, elapsed time: {this_time - last_time}")
            last_time = this_time
            last_progress_update = this_progress_update
        if not node:
            continue
        traffic_file = from_snr_to_traffic(snr_file, traffic_scenario)
        try:
            snr_trace = pd.read_parquet(snr_file)
        except EOFError:
            continue
        if cache_cells and snr_file not in cells_per_file:
            cells_per_file[snr_file] = list(snr_trace.Cell.astype(int).unique())
        snr_trace = snr_trace[(toi[0] <= snr_trace.Time) & (snr_trace.Time < toi[1])]
        snr_trace = snr_trace[snr_trace.Cell == cell]
        if direction == 'U':
            snr_trace.SNR = snr_trace.SNR - 21
        snr_trace["Time"] = (snr_trace["Time"] // tm) * tm
        snr_grouped = snr_trace.groupby("Time").agg({'SNR': 'mean'})
        snr_grouped["Node"] = node
        traffic_exists = False
        traffic_slice = None
        if traffic_file:
            try:
                traffic_trace = pd.read_parquet(traffic_file)
            except EOFError:
                continue
            if not traffic_trace.empty: 
                traffic_slice = traffic_trace.Slice.unique()[0]
            traffic_trace = traffic_trace[(toi[0] <= traffic_trace.Time) & (traffic_trace.Time < toi[1])]
            traffic_trace = traffic_trace[traffic_trace.Cell == cell]
            traffic_trace = traffic_trace[traffic_trace.Direction == direction]
            traffic_exists = not traffic_trace.empty
            if traffic_exists:
                traffic_trace["Time"] =  (traffic_trace["Time"] // tm) * tm
                traffic_trace["Bytes"] = np.ceil(traffic_trace["Bytes"])
                traffic_grouped = traffic_trace.groupby(["Time","Slice"]).agg({'Bytes': 'sum'})
                traffic_grouped = traffic_grouped.reset_index().set_index("Time")
                traffic_grouped.index = traffic_grouped.index.astype("float64")
                #print("DBG",snr_grouped.index, traffic_grouped.index,traffic_file,sep="\n")
                last_file = traffic_file
                last_df = traffic_grouped
                grouped = pd.merge(snr_grouped, traffic_grouped, how="left", on="Time")
                grouped = grouped.fillna(0)
                aggregate_df = aggregate_df.append(grouped.reset_index(), ignore_index=True)
        if not traffic_exists:
            grouped = snr_grouped
            grouped['Bytes'] = np.zeros(len(grouped))
            if traffic_slice is None:
                rnd_number = random.uniform(0,1)
                if rnd_number < 0.25:
                    traffic_slice = 4
                else:
                    traffic_slice = 2
            grouped['Slice'] = np.zeros(len(grouped)) + traffic_slice
            aggregate_df = aggregate_df.append(grouped.reset_index(), ignore_index=True)
        if max_iter is not None and i >= max_iter:
            break
    if cache_cells and write_cell_cache:
        with open(cell_json_file, 'wb') as outfile:
            pickle.dump(cells_per_file, outfile)
    # inputs
    reference_size = 1500
    processed_columns = aggregate_columns + ["SE", "MCS", "RBs", "RatioGlobal", "RatioExperienced", "RatioHypothetical","UERatio"]

    def line_processor(line):
        snr = line.SNR
        mcs = Util.getShannonMCS(snr, mcs_table=BIDIR_MCS[direction])
        line["MCS"] = mcs
        se = BIDIR_MCS[direction][mcs]['se']
        line["SE"] = se
        if line.Bytes != 0:
            #rbs =  Util.get_required_rbs(mcs, line.Bytes, 0.001)
            rbs = int(np.ceil(line.Bytes * 8 / (se * 15 * 12)))
        else:
            rbs = 0
        line["RBs"] = rbs
        rbs_hypothetical = Util.get_required_rbs(mcs, reference_size, 0.001)
        line["RatioExperienced"] = line.Bytes / rbs if rbs != 0 else 0
        line["RatioHypothetical"] = reference_size / rbs_hypothetical
        if rbs:
            line["RatioGlobal"] = line.RatioExperienced
        else:
            line["RatioGlobal"] = line.RatioHypothetical
        return line

    extended_df = pd.DataFrame(columns=processed_columns)
    extended_df = extended_df.append(aggregate_df)
    extended_df = extended_df.parallel_apply(line_processor, axis=1)
    if min_mcs is not None:
        extended_df = extended_df[extended_df.MCS>=min_mcs]
    

    rb_sum_per_time = extended_df.groupby("Time").agg({
        'RBs': 'sum'
    }).rename(columns={'RBs': 'RBsTotal'})
    #extended_df.UERatio = extended_df.RBs / rb_sum_per_time.loc[extended_df.Time].reset_index().RBsTotal
    extended_df.UERatio = extended_df.RBs / rb_sum_per_time.loc[extended_df.Time].values.squeeze()
    extended_df.UERatio = extended_df.UERatio.replace(np.inf, 0)
    

    extended_df.to_parquet(f"{cache_folder}/aggregated_{cell}_{direction}_{toi[0]}_{toi[1]}_{tm}_{traffic_scenario}_minmcs_{min_mcs}.parquet")
    info_obj = {
        'cell': int(cell),
        'direction': direction,
        'start': toi[0],
        'end': toi[1],
        'monitorization_period': tm
    }
    with open(f"aggregated_{cell}_{direction}_{toi[0]}_{toi[1]}_{tm}.json", 'w') as outfile:
        json.dump(info_obj, outfile)

    return extended_df


if __name__ == "__main__":
    
    
    pandarallel.initialize()
    
    np.warnings.filterwarnings('ignore')
    
    directions = ['D', 'U']
    #directions = ['D']
    tm_and_toi_confs = [(10, (0,23200)), (0.1, (0, 3600))]
    #tm_and_toi_confs = [(10, (0,23200))]
    
    #tm_and_toi_confs = [(10, (23200,)), (0.1, (3600,))]
    
    cells = range(21)
    with Pool(processes=1) as pool:
        for cell in []:
            for direction in directions:
                for tm, toi in tm_and_toi_confs:
                    for scenario in ["standard", "sporadic", "undistributed"]:
                        pool.apply_async(
                            process_monitoring_intervals, 
                            kwds={
                                'cell': cell,
                                'direction': direction,
                                'tm': tm,
                                'toi': toi,
                                'traffic_scenario': scenario,
                                'cache_folder': 'processed'
                            }, 
                            error_callback=(lambda x: print(x ))
                        )
                        #print(cell, direction, tm, toi)
        tms = [0.1, 60, 1200]
        tm_and_toi_confs = [(60, (0,23200)), (1200, (0,23200)), (0.1, (0, 3600))]
        for tm, toi in tm_and_toi_confs:
            source_tm = 0.1 if tm == 0.1 else 10
            for min_mcs in [None, 2, 5]:
                for max_static in [None, 2000, 1000]:
                    for scenario in ["standard", "sporadic", "undistributed"]:
                        for direction in directions:
                            pool.apply_async(
                                get_extended_metric_df_for_all_cells, 
                                kwds={
                                    'tm': tm,
                                    'tm_source': source_tm,
                                    'direction': direction,
                                    'toi': toi,
                                    'min_mcs': min_mcs,
                                    'max_static': max_static,
                                    'traffic_scenario': scenario,
                                    'cache_folder': 'processed',
                                    'mega_cache_folder': "megas"
                                },
                                error_callback=(lambda x: print("MEGAS ERROR", x ))
                            )
                            #print(tm, min_mcs, max_static, scenario, direction)
        time.sleep(2)
        pool.close()
        pool.join()
