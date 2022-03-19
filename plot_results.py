import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import pickle
import re

from learning_parameterization import split_samples, run_model_timeseries, run_model


ci95 = (lambda m: 1.96*np.sqrt(np.nanvar(m).mean()/(m.shape[0]*m.shape[1])))
mean = (lambda m: np.nanmean(m).mean())

def slice_class_name(slice_class):
    if slice_class[0] == 1:
        slice_name = "URLLC"
    elif slice_class[0] == 2:
        slice_name = "eMBB"
    elif slice_class[0] == 2:
        slice_name = "MTC"
    return f"{slice_name}, {'DL' if slice_class[1] == 'D' else 'UL'}"

def calculate_sum_with_ci_per_cell(detailed_results, metric=0, cells=range(0,21)):
    metric_sum = 0
    metric_ci = 0
    for c in cells:
        #values = detailed_results[c][metric]
        metric_sum += np.nanmean(detailed_results[c][metric])
        metric_ci += 1.96*np.sqrt(np.nanvar(detailed_results[c][metric])/len(detailed_results[c][metric]))
    return metric_sum, metric_ci
        


def plot_scenario(fix, variables, x_metric, models_conf, models_results, subplot=None, width=0.1, variable_filter=None, x_filter=None, split_plot = False, y_max=None, colors=None, cells=range(0,21), verbose=0, figsize=(5,3)):
    #new_df = pd.DataFrame(columns=["name", "label","x","cost","violations","overprovisioning"])
    new_df = pd.DataFrame()

    for key, conf in models_conf.items():
        discarded = False
        for metric in fix:
            #if metric not in conf:
            #    print(key,conf.keys())
            fixed_value = fix[metric] if fix[metric] != "None" else None
            if x_metric != 'alpha_target' and  'alpha' in fix and conf['alpha_target'][0] != fix['alpha']:
                #print("Discard", metric, conf[metric])
                discarded = True
                if discarded and verbose > 1:
                    print("Discarded alphatarget", conf['alpha_target'][0], fix['alpha'])
                break
            if fix[metric] is not None and metric in conf and conf[metric] != fixed_value:
                #print("Discard", metric, conf[metric])
                discarded = True
                if discarded and verbose > 1:
                    print("Discarded", metric, conf[metric], fixed_value)
                break
        if discarded:
            continue
        #if not split_plot and 'cell' in conf and conf['cell'] == 'all':
        #    continue
        #if 'split' not in conf:
        #    conf['split'] = 'local'
        if len(variables) == 1:
            label = conf[variables[0]]
        else:
            label = ",".join([f"{val}:{conf[val]}" for val in variables])
        label = label.replace("m1", "same-1")
        plot_conf = ",".join([f"{val}:{fix[val]}" for val in fix])
        
        if type(models_results[key]) is np.ndarray:
            new_entry = {
                "name": plot_conf,
                "label": label,
                "x": str(conf[x_metric]),
                "cost": mean(models_results[key][:,cells,0]),
                "cost_ci": ci95(models_results[key][:,cells,0]),
                "violations": mean(models_results[key][:,cells,1]),
                "violations_ci": ci95(models_results[key][:,cells,1]),
                "overprovision": mean(models_results[key][:,cells,2]),
                "overprovisioning_ci": ci95(models_results[key][:,cells,2]),
                "split": "cell" if conf['cell'] == 'all' else "global"
            }
        else:
            results_per_metric = {}
            for m in [0, 1, 2]:
                results_per_metric[m] = calculate_sum_with_ci_per_cell(models_results[key], metric=m, cells=cells)
            new_entry = {
                "name": plot_conf,
                "label": label,
                "x": str(conf[x_metric]),
                "cost": results_per_metric[0][0]/models_conf[key]['alpha'],
                "cost_ci": results_per_metric[0][1]/models_conf[key]['alpha'],
                "violations": results_per_metric[1][0]/len(cells),
                "violations_ci": results_per_metric[1][1]/len(cells),
                "overprovision": results_per_metric[2][0]/len(cells),
                "overprovisioning_ci": results_per_metric[2][1],
                "split": "cell" if conf['cell'] == 'all' else "global"
            }
        new_entry.update(conf)
        if type(conf[x_metric]) is tuple:
            new_entry["x_cmp"] = int(conf[x_metric][0])
            if type(conf[x_metric][1]) is str:
                new_entry['x'] = slice_class_name(conf[x_metric])
        elif not (type(conf[x_metric]) is str):
            new_entry["x_cmp"] = conf[x_metric]
            

        new_df = new_df.append(new_entry, ignore_index=True)

    if "x_cmp" in new_df:
        new_df =  new_df.sort_values(by=['x_cmp'])
    else:    
        new_df = new_df.sort_values(by=['x'])
    if variable_filter is not None:
        new_df = new_df[new_df[variables[0]].isin(variable_filter)]
    if x_filter is not None:
        new_df = new_df[new_df.x.isin(x_filter)]
    if new_df.name.nunique() > 1:
        print(new_df.name.values)
        assert new_df.name.nunique() == 1
        
    if verbose == 1:
        print(new_df[["label","x","cost","cost_ci"]])

    labels = list(set(new_df.label))
    labels.sort(reverse=True)
    
    if x_metric == "target":
        new_df.x = new_df.x/1000000
        
    #x_values = np.array(list(set(new_df.x)))
    x_values = new_df.x.unique()
    #x_values.sort()
    xs = {}
    for i, x_val in enumerate(x_values):
        xs[x_val]=i
    get_x_idx = np.vectorize(lambda x: xs[x])


    if subplot is None:
        c_fig, c_ax = plt.subplots(figsize=figsize)
    else:
        c_fig, c_ax = subplot
    
    #c_ax.grid(True,'both','y')
    c_ax.set_axisbelow(True)
    
    
    lines = new_df.groupby('label')
    gps = lines.groups
    if variable_filter is not None:
        gps = []
        for v in variable_filter:
            for l in lines.groups:
                if v in l:
                    gps.append(l)
    for label in gps:
        variables = lines.get_group(label)
        x = get_x_idx(variables.x)
        if variable_filter is not None:
            bar_pos = x-width*len(variable_filter)/2+variable_filter.index(variables.layer.iloc[0])*width
        else:
            bar_pos = x-width*len(labels)/2+labels.index(label)*width
        #bar_pos = x+width*len(bars)/2-bars.index(bars)*width
        if colors is None:
            l = c_ax.bar(bar_pos, variables.cost, width, yerr=variables.cost_ci, label=label, lw=2)
        else:
            l = c_ax.bar(bar_pos, variables.cost, width, yerr=variables.cost_ci, label=label, color = colors[variables.layer.iloc[0]], lw=2)
        c_ax.bar(bar_pos, variables.violations * variables.cost, width, color="ghostwhite", edgecolor=l.patches[0].get_facecolor(), hatch='//', lw=1)
        c_ax.set_xticks(range(len(x_values)))
        c_ax.set_xticklabels(x_values)
    
    if y_max is None:
        _, y_max = c_ax.get_ylim()
    c_ax.set_ylim((0,y_max))
    c_ax.set_yticks(np.arange(0,y_max,1),minor=True)
    c_ax.set_yticks(np.arange(0,y_max,4),minor=False)
    c_ax.grid(True,'major','y', c='dimgrey')
    c_ax.grid(True,'minor','y',linestyle='--')
    return c_fig, c_ax

def plot_scenario_by_cell(fix, variables, x_metric, models_conf, models_results, subplot=None, width=0.1, variable_filter=None, x_filter=None, split_plot = False, y_max=None, colors=None, cells=range(0,21), figsize=(5,3)):
    #new_df = pd.DataFrame(columns=["name", "label","x","cost","violations","overprovisioning"])
    new_df = pd.DataFrame()

    for key, conf in models_conf.items():
        discarded = False
        for metric in fix:
            #if metric not in conf:
            #    print(key,conf.keys())
            fixed_value = fix[metric] if fix[metric] != "None" else None
            if x_metric != 'alpha_target' and conf['alpha_target'][0] != fix['alpha']:
                discarded = True
                break
            if fix[metric] is not None and metric in conf and  conf[metric] != fixed_value:
                discarded = True
                break
        if discarded:
            continue
        #if not split_plot and 'cell' in conf and conf['cell'] == 'all':
        #    continue
        #if 'split' not in conf:
        #    conf['split'] = 'local'
        if len(variables) == 1:
            label = conf[variables[0]]
        else:
            label = ",".join([f"{val}:{conf[val]}" for val in variables])
        label = label.replace("m1", "same-1")
        plot_conf = ",".join([f"{val}:{fix[val]}" for val in fix])
        for cell in cells:
            c = cell
            if type(models_results[key]) is np.ndarray:
                new_entry = {
                    "name": plot_conf,
                    "label": label,
                    "": label,
                    "x": conf[x_metric] if x_metric != 'Cell' else int(cell),
                    "cost": mean(models_results[key][:,[cell],0]),
                    "cost_ci": ci95(models_results[key][:,[cell],0]),
                    "violations": mean(models_results[key][:,[cell],1]),
                    "violations_ci": ci95(models_results[key][:,[cell],1]),
                    "overprovisioning": mean(models_results[key][:,[cell],2]),
                    "overprovisioning_ci": ci95(models_results[key][:,[cell],2]),
                    "split": "cell" if conf['cell'] == 'all' else "global",
                    "Cell": int(c),
                    "\u03B1": int(conf["alpha_target"][0])
                }
            else:
                results_per_metric = {}
                for m in [0, 1, 2]:
                    results_per_metric[m] = calculate_sum_with_ci_per_cell(models_results[key], metric=m, cells=[c])
                new_entry = {
                    "name": plot_conf,
                    "label": label,
                    "x": conf[x_metric] if x_metric != 'Cell' else int(cell),
                    "cost": results_per_metric[0][0]/models_conf[key]['alpha'],
                    "cost_ci": results_per_metric[0][1]/models_conf[key]['alpha'],
                    "violations": results_per_metric[1][0],
                    "violations_ci": results_per_metric[1][1],
                    "overprovision": results_per_metric[2][0],
                    "overprovisioning_ci": results_per_metric[2][1],
                    "split": "cell" if conf['cell'] == 'all' else "global",
                    "Cell": int(c),
                    "\u03B1": int(conf["alpha_target"][0])
                }
            new_entry.update(conf)
            new_df = new_df.append(new_entry, ignore_index=True) 

        new_df = new_df.append(new_entry, ignore_index=True)  


    assert len(new_df) > 0
    if variable_filter is not None:
        new_df = new_df[new_df[variables[0]].isin(variable_filter)]
    if x_filter is not None:
        new_df = new_df[new_df.x.isin(x_filter)]
    if x_metric in ["Cell"]:
        new_df.x = new_df.x.astype(int).astype(str)
    if new_df.name.nunique() > 1:
        print(new_df.name.values)
        assert new_df.name.nunique() == 1

    labels = list(set(new_df.label))
    labels.sort(reverse=True)
    
    if x_metric == "target":
        new_df.x = new_df.x//1000000

    #x_values = np.array(list(set(new_df.x)))
    x_values = new_df.x.unique()
    #x_values.sort()
    xs = {}
    for i, x_val in enumerate(x_values):
        xs[x_val]=i
    get_x_idx = np.vectorize(lambda x: xs[x])


    if subplot is None:
        c_fig, c_ax = plt.subplots(figsize=figsize)
    else:
        c_fig, c_ax = subplot
    
    #c_ax.grid(True,'both','y')
    c_ax.set_axisbelow(True)
    
    
    lines = new_df.groupby('label')
    gps = lines.groups
    if variable_filter is not None:
        gps = []
        for v in variable_filter:
            for l in lines.groups:
                if v in l:
                    gps.append(l)
    for label in gps:
        variables = lines.get_group(label)
        assert len(lines) > 0
        x = get_x_idx(variables.x)
        if variable_filter is not None:
            bar_pos = x-width*len(variable_filter)/2+variable_filter.index(variables.layer.iloc[0])*width
        else:
            bar_pos = x-width*len(labels)/2+labels.index(label)*width
        #bar_pos = x+width*len(bars)/2-bars.index(bars)*width
        if colors is None:
            l = c_ax.bar(bar_pos, variables.cost, width, yerr=variables.cost_ci, label=label, lw=2)
        else:
            l = c_ax.bar(bar_pos, variables.cost, width, yerr=variables.cost_ci, label=label, color = colors[variables.layer.iloc[0]], lw=2)
        c_ax.bar(bar_pos, variables.violations * variables.cost, width, color="ghostwhite", edgecolor=l.patches[0].get_facecolor(), hatch='//', lw=1)
        c_ax.set_xticks(range(len(x_values)))
        c_ax.set_xticklabels(x_values)

    
    if y_max is None:
        _, y_max = c_ax.get_ylim()
    #y_max = 2.15
    c_ax.set_ylim((0,y_max))
    c_ax.set_yticks(np.arange(0,y_max,0.05),minor=True)
    c_ax.set_yticks(np.arange(0,y_max,0.2),minor=False)
    c_ax.grid(True,'major','y', c='dimgrey')
    c_ax.grid(True,'minor','y',linestyle='--')
    return c_fig, c_ax

def plot_num_steps(fix, variables, x_metric, models_conf, models_results, subplot=None, width=0.1, variable_filter=None, x_filter=None, split_plot = False, y_max=None, colors=None, cells=range(0,21), verbose=0, figsize=(5,3)):
    new_df = pd.DataFrame()

    for key, conf in models_conf.items():
        discarded = False
        for metric in fix:
            #if metric not in conf:
            #    print(key,conf.keys())
            fixed_value = fix[metric] if fix[metric] != "None" else None
            if x_metric != 'alpha_target' and  'alpha' in fix and conf['alpha_target'][0] != fix['alpha']:
                #print("Discard", metric, conf[metric])
                discarded = True
                if discarded and verbose > 1:
                    print("Discarded alphatarget", conf['alpha_target'][0], fix['alpha'])
                break
            if fix[metric] is not None and metric in conf and conf[metric] != fixed_value:
                #print("Discard", metric, conf[metric])
                discarded = True
                if discarded and verbose > 1:
                    print("Discarded", metric, conf[metric], fixed_value)
                break
        if discarded:
            continue
        #if not split_plot and 'cell' in conf and conf['cell'] == 'all':
        #    continue
        #if 'split' not in conf:
        #    conf['split'] = 'local'
        if len(variables) == 1:
            label = conf[variables[0]]
        else:
            label = ",".join([f"{val}:{conf[val]}" for val in variables])
        label = label.replace("m1", "same-1")
        plot_conf = ",".join([f"{val}:{fix[val]}" for val in fix])
        
        if type(models_results[key]) is np.ndarray:
            new_entry = {
                "name": plot_conf,
                "label": label,
                "x": str(conf[x_metric]),
                "cost": mean(models_results[key][:,cells,0]),
                "cost_ci": ci95(models_results[key][:,cells,0]),
                "violations": mean(models_results[key][:,cells,1]),
                "violations_ci": ci95(models_results[key][:,cells,1]),
                "overprovision": mean(models_results[key][:,cells,2]),
                "overprovisioning_ci": ci95(models_results[key][:,cells,2]),
                "split": "cell" if conf['cell'] == 'all' else "global"
            }
        else:
            results_per_metric = {}
            for m in [0, 1, 2]:
                results_per_metric[m] = calculate_sum_with_ci_per_cell(models_results[key], metric=m, cells=cells)
            new_entry = {
                "name": plot_conf,
                "label": label,
                "x": str(conf[x_metric]),
                "cost": results_per_metric[0][0]/models_conf[key]['alpha'],
                "cost_ci": results_per_metric[0][1]/models_conf[key]['alpha'],
                "violations": results_per_metric[1][0]/len(cells),
                "violations_ci": results_per_metric[1][1]/len(cells),
                "overprovision": results_per_metric[2][0]/len(cells),
                "overprovisioning_ci": results_per_metric[2][1],
                "split": "cell" if conf['cell'] == 'all' else "global"
            }
        new_entry.update(conf)
        if type(conf[x_metric]) is tuple:
            new_entry["x_cmp"] = int(conf[x_metric][0])
            if type(conf[x_metric][1]) is str:
                new_entry['x'] = slice_class_name(conf[x_metric])
        elif not (type(conf[x_metric]) is str):
            new_entry["x_cmp"] = conf[x_metric]
            

        new_df = new_df.append(new_entry, ignore_index=True)  

    if "x_cmp" in new_df:
        new_df =  new_df.sort_values(by=['x_cmp'])
    else:    
        new_df = new_df.sort_values(by=['x'])
    if variable_filter is not None:
        new_df = new_df[new_df[variables[0]].isin(variable_filter)]
    if x_filter is not None:
        new_df = new_df[new_df.x.isin(x_filter)]
    if new_df.name.nunique() > 1:
        print(new_df.name.values)
        assert new_df.name.nunique() == 1
        
    if verbose == 1:
        print(new_df)

    labels = list(set(new_df.label))
    labels.sort(reverse=True)
    
    if x_metric == "target":
        new_df.x = new_df.x/1000000
        
    #x_values = np.array(list(set(new_df.x)))
    x_values = new_df.x.unique()
    #x_values.sort()
    xs = {}
    for i, x_val in enumerate(x_values):
        xs[x_val]=i
    get_x_idx = np.vectorize(lambda x: xs[x])


    if subplot is None:
        c_fig, c_ax = plt.subplots(figsize=figsize)
    else:
        c_fig, c_ax = subplot
    
    #c_ax.grid(True,'both','y')
    c_ax.set_axisbelow(True)
    
    
    lines = new_df.groupby('label')
    gps = lines.groups
    if variable_filter is not None:
        gps = []
        for v in variable_filter:
            for l in lines.groups:
                if v in l:
                    gps.append(l)
    for label in gps:
        variables = lines.get_group(label)
        x = get_x_idx(variables.x)
    
    c_ax.errorbar(x,[0.409043*21]*len(x),yerr=[0.046110*21]*len(x),label="Past")
    for label in gps:
        variables = lines.get_group(label)
        x = get_x_idx(variables.x)
        if colors is None:
            c_ax.errorbar(x,variables.cost,yerr=variables.cost_ci,label=label)
        else:
            c_ax.errorbar(x,variables.cost,yerr=variables.cost_ci,label=label,color=colors[variables.layer.iloc[0]])
        c_ax.set_xticks(range(len(x_values)))
        c_ax.set_xticklabels(x_values)
    c_ax.errorbar(x,[0.266213*21]*len(x),yerr=[0.024174*21]*len(x),label="KPIC-FixedOP")
    
    if y_max is None:
        _, y_max = c_ax.get_ylim()
    #y_max = 2.15
    c_ax.set_ylim((0,y_max))
    c_ax.set_yticks(np.arange(0,y_max,1),minor=True)
    c_ax.set_yticks(np.arange(0,y_max,4),minor=False)
    c_ax.grid(True,'major','y', c='dimgrey')
    c_ax.grid(True,'minor','y',linestyle='--')
    return c_fig, c_ax
       
    
def plot_scatter(fix, variables, models_conf, models_results, annotation_filter=None, subplot=None, variable_filter=None, x_filter=None, split_plot = False, y_max=None, colors=None, cells=range(0,21)):
    new_df = pd.DataFrame()
    for key, conf in models_conf.items():
        discarded = False
        if conf["alpha"] == 250:
            continue
        for metric in fix:
            if metric not in conf:
                print(key,conf.keys())
            fixed_value = fix[metric] if fix[metric] != "None" else None
            if fix[metric] is not None and  conf[metric] != fixed_value:
                discarded = True
                break
        if discarded:
            continue
        #if not split_plot and 'cell' in conf and conf['cell'] == 'all':
        #    continue
        #if 'split' not in conf:
        #    conf['split'] = 'local'
        if len(variables) == 1:
            label = conf[variables[0]]
        else:
            label = ",".join([f"{val}:{conf[val]}" for val in variables])
        plot_conf = ",".join([f"{val}:{conf[val]}" for val in fix])
        for cell in cells:
            c = cell
            if type(models_results[key]) is np.ndarray:
                new_entry = {
                    "name": plot_conf,
                    "label": label,
                    "": label,
                    "": label,
                    "x": conf[x_metric] if x_metric != 'Cell' else int(cell),
                    "cost": mean(models_results[key][:,[cell],0]),
                    "cost_ci": ci95(models_results[key][:,[cell],0]),
                    "Violations (%)": mean(models_results[key][:,[cell],1]),
                    "violations_ci": ci95(models_results[key][:,[cell],1]),
                    "Over-provisioning (%)": mean(models_results[key][:,[cell],2]),
                    "overprovisioning_ci": ci95(models_results[key][:,[cell],2]),
                    "split": "cell" if conf['cell'] == 'all' else "global",
                    "Cell": int(c),
                    "\u03B1": int(conf["alpha_target"][0])
                }
            else:
                results_per_metric = {}
                for m in [0, 1, 2]:
                    results_per_metric[m] = calculate_sum_with_ci_per_cell(models_results[key], metric=m, cells=[c])
                new_entry = {
                    "name": plot_conf,
                    "label": label,
                    "": label,
                    "x": conf[x_metric] if x_metric != 'Cell' else int(cell),
                    "cost": results_per_metric[0][0]/models_conf[key]['alpha'],
                    "cost_ci": results_per_metric[0][1]/models_conf[key]['alpha'],
                    "Violations (%)": results_per_metric[1][0],
                    "violations_ci": results_per_metric[1][1],
                    "Over-provisioning (%)": results_per_metric[2][0],
                    "overprovisioning_ci": results_per_metric[2][1],
                    "split": "cell" if conf['cell'] == 'all' else "global",
                    "Cell": int(c),
                    "\u03B1": int(conf["alpha_target"][0])
                }
            new_entry.update(conf)
            new_df = new_df.append(new_entry, ignore_index=True)  
    assert not new_df.empty
    new_df = new_df.sort_values(by=['label'])
    new_df["\u03B1"] = new_df["\u03B1"].astype(int)
    
    new_df["Cell"] = new_df["Cell"].astype(int).astype(str)
    if variable_filter is not None:
        new_df = new_df[new_df[variables[0]].isin(variable_filter)]
    if new_df.name.nunique() > 1:
        print(new_df.name.values)
        assert new_df.name.nunique() == 1
    # labels = list(set(new_df.label))
    
    ax = sns.relplot(data=new_df, x="Over-provisioning (%)", y="Violations (%)", hue="", style="", col="\u03B1", height=2, col_wrap=2, aspect=3/3)
    fig = ax.fig
    return fig, ax


def compile_results(folders, sim_reference=None, run_dummy=True):

    simulations = {}
    models_conf = {}
    models_results = {}
    models_results_by_cell = {}
    
    for sim_type in folders:
        print(sim_type)
        for sim_file in glob.glob(folders[sim_type], recursive=True):

            #Preping
            simulations[sim_file] = pickle.load(open(sim_file,"rb"))
            underscore_separated = re.sub(r"source_attenuation_([0-9]*)",r"source_attenuation$\1",".".join(sim_file.split("/")[-1].split(".")[:-1]))
            fname_confs = dict(map(lambda x: list(map(lambda y: y.replace("$","_"),x.split("_"))), underscore_separated.split("-")[1:]))
            alpha = int(simulations[sim_file]["setup"]['alpha'])
            target = simulations[sim_file]["setup"]['target_bps']/1000000
            # nwslice = int(fname_confs['slice'])
            # tm = float(fname_confs['tm'])
            # max_static = fname_confs['maxstatic']
            # min_mcs = fname_confs['minmcs']
            scenario = fname_confs['scenario']
            scenario = scenario[0:1].upper() + scenario[1:]
            # direction = fname_confs['direction']
            # source_folder = "/".join(simulations[sim_file]["setup"]['source_fname'].split("/")[:-1])
            cells = simulations[sim_file]['setup']['cells']
            if len(cells) == 1: #old files bug
                cells = range(0,21)

            mega_df = pd.read_parquet(simulations[sim_file]["setup"]['source_fname'].replace("megas-","megas"))
            mega_df = mega_df[~mega_df.RB_ratio.isna()]
            mega_df = mega_df[~mega_df.Last_RB_ratio.isna()]
            max_se = simulations[sim_file]["setup"]["max_se"]
            training, test, validation = split_samples(mega_df, ratio=simulations[sim_file]["setup"]['ratio'])
            set_origin = test
            set_name = "test"

            #Confs and results: learning
            models_conf[sim_file] = {}
            models_conf[sim_file].update(simulations[sim_file]["setup"])
            models_conf[sim_file]['alpha_target'] = (int(simulations[sim_file]["setup"]['alpha']), simulations[sim_file]["setup"]['target_bps']/1000000)
            models_conf[sim_file]['alpha'] = int(simulations[sim_file]["setup"]['alpha'])
            models_conf[sim_file]['max_static'] = int(fname_confs['maxstatic']) if fname_confs['maxstatic'] != 'None' else None
            models_conf[sim_file]['min_mcs'] = int(fname_confs['minmcs']) if fname_confs['minmcs'] != 'None' else None
            models_conf[sim_file]['scenario'] = scenario
            models_conf[sim_file]['source'] = fname_confs['source']
            models_conf[sim_file]['numsteps'] = int(fname_confs['numsteps'])
            if 'megas' in fname_confs['source']:
                try:
                    models_conf[sim_file]['attenuation'] = int(fname_confs['source'].replace("megas",""))
                except:
                    models_conf[sim_file]['attenuation'] = 0
            else:
                models_conf[sim_file]['attenuation'] = int(models_conf[sim_file]['source'].split("_")[1])
            models_conf[sim_file]['direction'] = fname_confs['direction']
            models_conf[sim_file]['tm'] = float(fname_confs['tm'])
            models_conf[sim_file]['layer'] = sim_type
            models_conf[sim_file]['nwslice'] = int(fname_confs['slice'])
            models_conf[sim_file]['slice_class'] = (models_conf[sim_file]['nwslice'], models_conf[sim_file]['direction'])
            models_conf[sim_file]['cell'] = 'split'
            simulations[sim_file]["test_results"] = simulations[sim_file]["results"]
            models_results[sim_file] = simulations[sim_file]['detailed_results'][set_name]
            models_results_by_cell[sim_file] = {}
            for c in simulations[sim_file]['detailed_results'][set_name]:
                models_results_by_cell[sim_file][c] = np.array(simulations[sim_file]['detailed_results'][set_name][c][0:3])
                models_results_by_cell[sim_file][c][0] /= simulations[sim_file]["setup"]['alpha']

            trained_resources=simulations[sim_file]['setup']['train_resources'] if 'train_resources' in simulations[sim_file]['setup'] else False
            if sim_reference is not None and sim_type == sim_reference:
                settings_str = sim_file  + "_fozero"
                models_conf[settings_str] = models_conf[sim_file].copy()
                models_conf[settings_str]['layer'] = "Same"
                cell_results = {}
                for cell in cells:
                    if "dataset" in simulations[sim_file]:
                        cell_results[cell] = run_model_timeseries(
                            "Last", 
                            alpha, 
                            target*1000000, 
                            simulations[sim_file]["dataset"][cell][set_name]["features"],
                            simulations[sim_file]["dataset"][cell][set_name]["predictions"],
                            max_se,
                            fixed_overprovisioning=0,
                            trained_resources=trained_resources
                        )
                    else:
                        assert 1 != 1
                models_results[settings_str] = cell_results
        
    return simulations, models_conf, models_results

if __name__ == "__main__":
    simulations, models_conf, models_results = compile_results({
        'KPIC-NN': 'results-kpic-9/**/*.pickle',
        'KPIC-FixedOP': 'results-kpic-1-simple/**/*.pickle',
        'Past': 'results-last/**/*.pickle'
    })

    simulations_ns, models_conf_ns, models_results_ns = compile_results({
        'KPIC-NN': 'results-kpic-bynumsteps/**/*.pickle'
    })
    
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    default_fix = {'scenario': 'Standard', 'max_static': 'None', 'min_mcs': 'None', 'alpha': 2000, 'tm': 60, 'slice_class': (2, 'D'), 'attenuation':0}

    cells=range(0,21)

    global_ymax = 0.8*21

    fix = default_fix.copy()
    variables = ['layer']
    x_metric = 'alpha'
    #x_filter = [500, 1000, 2000, 5000]
    x_filter = None

    del fix['alpha']

    fig, ax = plot_scenario(fix, variables, x_metric, models_conf, models_results, x_filter=x_filter, subplot=None, width=0.12, y_max=global_ymax,colors=None, cells=cells)
    ax.legend()


    ax.set_xlabel("α")
    ax.set_ylabel("Cost")

    fig.savefig("figures/cost_by_alpha_standard_2D.pdf", bbox_inches='tight')


    fix = default_fix.copy()
    variables = ['layer']
    x_metric = 'numsteps'



    fig, ax = plot_num_steps(fix, variables, x_metric, models_conf_ns, models_results_ns, subplot=None, width=0.12, y_max=global_ymax,colors=None, cells=cells)
    ax.legend()


    ax.set_xlabel("# Time steps")
    ax.set_ylabel("Cost")

    fig.savefig("figures/cost_by_numsteps_standard_2000.pdf", bbox_inches='tight')

    fix = default_fix.copy()
    variables = ['layer']
    x_metric = 'Cell'
    fix['alpha']=1000
    #x_filter = [0, 3, 4, 6, 11, 13]
    x_filter = [0, 1, 2, 3, 4, 6, 11, 13]
    #x_filter=range(1,10)

    global_ymax = 2

    fig, ax = plot_scenario_by_cell(fix, variables, x_metric, models_conf, models_results, subplot=None, width=0.12, y_max=global_ymax, colors=None, x_filter=x_filter, cells=cells)
    ax.legend()
    ax.set_xlabel("Cell")
    ax.set_ylabel("Cost")


    fig.savefig("figures/cost_by_cell_filtered_2000_2D.pdf", bbox_inches='tight')



    global_ymax = 0.8*21


    fix = default_fix.copy()
    fix['alpha']
    variables = ['layer']
    x_metric = 'scenario'
    del fix[x_metric]


    fig, ax = plot_scenario(fix, variables, x_metric, models_conf, models_results, subplot=None, width=0.12, y_max=global_ymax,colors=None, cells=cells)
    ax.legend()


    ax.set_xlabel("Scenario")
    ax.set_ylabel("Cost")


    fig.savefig("figures/cost_by_scenario_2000_2D.pdf", bbox_inches='tight')


    fix = default_fix.copy()
    del fix['alpha']
    variable_filter = ['Past', 'KPIC-FixedOP','KPIC-NN']
    fig, ax= plot_scatter(fix, variables, models_conf, models_results, variable_filter=variable_filter, cells=cells)

    fig.savefig("figures/scatter_standard_2000_2D.pdf", bbox_inches='tight')

    fix = default_fix.copy()
    variables = ['layer']
    x_metric = 'attenuation'
    del fix[x_metric]


    fig, ax = plot_scenario(fix, variables, x_metric, models_conf, models_results, subplot=None, width=0.12, y_max=global_ymax,colors=None, cells=cells)
    ax.legend()


    ax.set_xlabel("Attenuation (dB)")
    ax.set_ylabel("Cost")


    fig.savefig("figures/cost_by_dataset_standard_2000_2D.pdf", bbox_inches='tight')

