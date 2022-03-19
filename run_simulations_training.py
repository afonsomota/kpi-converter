from learning_parameterization import *
import pickle
import time
import os
import sys
from pathlib import Path
from multiprocessing import Pool
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, LSTM, GRU, SimpleRNN
import traceback

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer


if __name__ == "__main__":
    hl = 2
    hu = 5
    epochs= 1000
    overwrite = False
        
    if len(sys.argv) > 1:
        conf=sys.argv[1]
    else:
        conf = "Test"  # Test, FeatSel, NonFeatSel, TrainAll
        
    with_max_penalty=True
    train_subset = True
    do_train_resources = False
    do_feature_selection = False
    do_transform_input = True
    time_flat_dense = False
    epochs_between_reports = 10
    if train_subset:
        cells = [0, 1, 2]
    else:
        cells = range(0,21)
    
    features=['SNR_std', 'SNR_5', 'SNR_25', 'SNR_50', 'SNR_75', 'SNR_95',
            'Bytes_std', 'Bytes_5', 'Bytes_25', 'Bytes_50', 'Bytes_75', 'Bytes_95',
            'Last_equally_distributed', 'Reverse_SE', 'Last_SE'
           ]
    
    learning_family = "NN"
        
    
    target_mbps = {
        'D': {
            2: 500
        },
        'U': {
            2: 2,
            1: 10,
            3: 0.01
        }
    }
    neural_cell = SimpleRNN
           
    dropout = 0
    gradient = 0.0005
    eps = 0.1
    batch_size = 10
    nono = False
    num_steps = 5
    shuffle = None
    loss_fun=None
    
    break_first = False
    
    learning_params = None
    num_steps_array = None
    
            
    if conf=="FeatureSelection":
        at_pairs = [
            (250, 1000),
            (500, 1000),
            (1000, 1000),
            (2000, 1000),
            #(5000, None),
            #(10000, 1000)
        ]
        tms = [60]
        scenarios = ['standard', 'sporadic', 'undistributed']
        directions = ["D"]
        max_static_confs = [None]
        min_mcs_confs = [None]
        source_folders = ["phase.2/attenuation_0", "phase.2/attenuation_30", "phase.2/attenuation_10"]
        #source_folders = ["megas", "megas10", "megas30"]
        nwslices = [ 2]
        do_feature_selection=True
        choose_last=True
        script_function=run_configurations_per_cell 
        #script_function=run_configuration
        folder_name = "learning-simulation-timeseries-nonselect2"
        custom=f"-style_bycell"
        verbose=1
        ratio=(0.6, 0.2, 0)
        cell=None
    elif conf=="KPIC-NN-numsteps":
        at_pairs = [
            (2000, 1000),
            #(250, 1000),
            #(500, 1000),
            #(1000, 1000),
            #(5000, 1000),
            #(10000, 1000)
        ]
        #num_steps = int(input("Number of past steps: "))
        neural_cell = Dense
        tms = [60]
        #scenarios = ['standard', 'sporadic', 'undistributed']
        scenarios = ['standard']
        #directions = ["D", "U"]
        directions = ["D"]
        max_static_confs = [None]
        min_mcs_confs = [None]
        num_steps_array = [11,12,13,14,15,16,17,18,19,20]
        #source_folders = ["phase.2/attenuation_0", "phase.2/attenuation_30", "phase.2/attenuation_10"]
        source_folders = ["phase.2/attenuation_0"]
        #source_folders = ["megas", "megas10", "megas30"]
        nwslices = [ 1, 2]
        do_feature_selection=False
        do_transform_input = False
        time_flat_dense = False
        choose_last=True
        script_function=run_configurations_per_cell 
        #script_function=run_configuration
        folder_name = f"results-kpic-bynumsteps"
        #num_steps = 2
        custom=f"-style_bycell"
        cell = None
        verbose = 1
        cells= range(0,21)
        do_train_resources = False
        
        #verbose=2
        #cell=0
        #epochs=1
        #script_function=run_configuration
        with_max_penalty=False
        
        ratio=(0.8, 0.2, 0)
        features=['SNR_std', 'SNR_5', 'SNR_25', 'SNR_50', 'SNR_75', 'SNR_95',
            'Bytes_std', 'Bytes_5', 'Bytes_25', 'Bytes_50', 'Bytes_75', 'Bytes_95',
          #  'Last_equally_distributed', 'Reverse_SE', 'Last_SE'
             'Last_equally_distributed', 'Last_SE'
           ]
    elif conf=="KPIC-NN":
        at_pairs = [
            (2000, 1000),
            (250, 1000),
            (500, 1000),
            (1000, 1000),
            (5000, 1000),
            #(10000, 1000)
        ]
        num_steps = int(input("Number of past steps: "))
        neural_cell = Dense
        tms = [60]
        scenarios = ['standard', 'sporadic', 'undistributed']
        #scenarios = ['standard']
        #directions = ["D", "U"]
        directions = ["D"]
        max_static_confs = [None]
        min_mcs_confs = [None]
        source_folders = ["phase.2/attenuation_0"]
        nwslices = [ 1, 2]
        do_feature_selection=False
        do_transform_input = False
        time_flat_dense = False
        choose_last=True
        script_function=run_configurations_per_cell 
        #script_function=run_configuration
        folder_name = f"results-kpic-{num_steps}"
        #num_steps = 2
        custom=f"-style_bycell"
        cell = None
        verbose = 1
        cells= range(0,21)
        do_train_resources = False
        
        #verbose=2
        #cell=0
        #epochs=1
        #script_function=run_configuration
        with_max_penalty=False
        
        ratio=(0.8, 0.2, 0)
        features=['SNR_std', 'SNR_5', 'SNR_25', 'SNR_50', 'SNR_75', 'SNR_95',
            'Bytes_std', 'Bytes_5', 'Bytes_25', 'Bytes_50', 'Bytes_75', 'Bytes_95',
          #  'Last_equally_distributed', 'Reverse_SE', 'Last_SE'
             'Last_equally_distributed', 'Last_SE'
           ]
    elif conf=="Last":
        overwrite = False
        at_pairs = [
            (2000, 1000),
            (250, 1000),
            (500, 1000),
            (1000, 1000),
            (5000, 1000),
            #(10000, 1000)
        ]
        neural_cell = Dense
        tms = [60]
        #scenarios = ['standard', 'sporadic', 'undistributed']
        #directions = ["D", "U"]
        scenarios = ['standard']
        directions = ["D"]
        max_static_confs = [None]
        min_mcs_confs = [None]
        source_folders = ["phase.2/attenuation_0"]
        nwslices = [ 1, 2]
        do_feature_selection=False
        do_transform_input = False
        time_flat_dense = True
        choose_last=True
        script_function=run_configurations_per_cell
        folder_name = f"results-last"
        num_steps = 1
        custom=f"-style_bycell"
        cell = None
        verbose = 1
        cells= range(0,21)
        do_train_resources = True
        learning_family="Last"
        
        ratio=(0.8, 0.2, 0)
        features=[ 'Last_SE']
    # Using Gradient boosting
    elif conf=="KPIC-GB":
        overwrite = True
        at_pairs = [
            (2000, 1000),
            #(250, 1000),
            (500, 1000),
            (1000, 1000),
            (5000, None),
            (10000, 1000)
        ]
        tms = [60]
        scenarios = ['standard', 'sporadic', 'undistributed']
        #scenarios = ['standard']
        directions = ["D", "U"]
        #directions = ["D"]
        max_static_confs = [None]
        min_mcs_confs = [None]
        source_folders = ["phase.2/attenuation_0", "phase.2/attenuation_30", "phase.2/attenuation_10"]
        #source_folders = ["phase.2/attenuation_0"]
        #source_folders = ["megas", "megas10", "megas30"]
        nwslices = [ 1, 2]
        do_feature_selection=False
        do_transform_input = True
        time_flat_dense = True
        script_function=run_configurations_per_cell 
        #script_function=run_configuration
        folder_name = "results-kpic-1-gb-std-unbound"
        num_steps = 1
        custom=f"-style_bycell"
        cell = None
        verbose = 1
        cells= range(0,21)
        do_train_resources = True
        epochs=50
        learning_family = "GB"
        
        #verbose=2
        #cell=0
        #epochs=1
        #script_function=run_configuration
        
        ratio=(0.8, 0.2, 0)
        features=['SNR_std', 'SNR_5', 'SNR_25', 'SNR_50', 'SNR_75', 'SNR_95',
            'Bytes_std', 'Bytes_5', 'Bytes_25', 'Bytes_50', 'Bytes_75', 'Bytes_95',
            'Last_equally_distributed', 'Reverse_SE', 'Last_SE'
           ]
    elif conf=="KPIC-FixedOP":
        overwrite = False
        at_pairs = [
            (2000, 1000),
            (250, 1000),
            (500, 1000),
            (1000, 1000),
            (5000, None),
            #(10000, 1000)
        ]
        tms = [60]
        #scenarios = ['standard', 'sporadic', 'undistributed']
        scenarios = ['standard']
        #directions = ["D", "U"]
        directions = ["D"]
        max_static_confs = [None]
        min_mcs_confs = [None]
        #source_folders = ["phase.2/attenuation_0", "phase.2/attenuation_30", "phase.2/attenuation_10"]
        source_folders = ["phase.2/attenuation_0"]
        #source_folders = ["megas", "megas10", "megas30"]
        nwslices = [ 1, 2]
        time_flat_dense = True
        script_function=run_configurations_per_cell 
        #script_function=run_configuration
        folder_name = "results-kpic-1-simple"
        num_steps = 1
        custom=f"-style_bycell"
        cell = None
        verbose = 1
        cells= range(0,21)
        do_train_resources = True
        epochs=50
        learning_family = "Simple"
        
        ratio=(0.8, 0.2, 0)
        features=['Last_SE']
    # Using different asymmetrical function
    elif conf=="KPIC-asym":
        at_pairs = [
            (2000, 1000),
            #(250, 1000),
            (500, 1000),
            (1000, 1000),
            (5000, 1000),
            (10000, 1000)
        ]
        neural_cell = Dense
        tms = [60]
        scenarios = ['standard', 'sporadic', 'undistributed']
        #scenarios = ['standard']
        directions = ["D", "U"]
        #directions = ["D"]
        max_static_confs = [None]
        min_mcs_confs = [None]
        source_folders = ["phase.2/attenuation_0", "phase.2/attenuation_30", "phase.2/attenuation_10"]
        #source_folders = ["phase.2/attenuation_0"]
        #source_folders = ["megas", "megas10", "megas30"]
        nwslices = [ 1, 2]
        do_feature_selection=False
        do_transform_input = True
        time_flat_dense = True
        choose_last=True
        script_function=run_configurations_per_cell 
        #script_function=run_configuration
        folder_name = "results-kpic-1-asymlog"
        num_steps = 1
        custom=f"-style_bycell-featsel_false"
        cell = None
        verbose = 1
        cells= range(0,21)
        do_train_resources = True
        eps=0.001
        loss_fun = standard_asymmetric_rbs_with_log
        epochs=300
        
        #verbose=2
        #cell=0
        #epochs=1
        #script_function=run_configuration
        
        ratio=(0.8, 0.2, 0)
        features=['SNR_std', 'SNR_5', 'SNR_25', 'SNR_50', 'SNR_75', 'SNR_95',
            'Bytes_std', 'Bytes_5', 'Bytes_25', 'Bytes_50', 'Bytes_75', 'Bytes_95',
            'Last_equally_distributed', 'Reverse_SE', 'Last_SE'
           ]
    # Using LSTM and different asymmetrical function
    elif conf=="TSstyle-asym":
        at_pairs = [
            (2000, 1000),
            #(250, 1000),
            (500, 1000),
            (1000, 1000),
            (5000, 1000),
            (10000, 1000)
        ]
        
        num_steps = 5
        neural_cell = Dense
        tms = [60]
        scenarios = ['standard', 'sporadic', 'undistributed']
        #scenarios = ['standard']
        directions = ["D", "U"]
        #directions = ["D"]
        max_static_confs = [None]
        min_mcs_confs = [None]
        source_folders = ["phase.2/attenuation_0", "phase.2/attenuation_30", "phase.2/attenuation_10"]
        #source_folders = ["phase.2/attenuation_0"]#source_folders = ["megas", "megas10", "megas30"]
        nwslices = [ 1, 2]
        do_feature_selection=False
        do_transform_input = False
        choose_last=True
        script_function=run_configurations_per_cell 
        #script_function=run_configuration
        folder_name = "results-ts-gru-asym"
        custom=f"-style_bycell"
        verbose=1
        ratio=(0.8, 0.2, 0)
        cell=None
        cells= range(0,21)
        do_train_resources = True
        eps=0.001
        loss_fun = standard_asymmetric_rbs_with_log
        epochs=300
        
        features=[ 'Last_SE' ]
    # Using LSTM and get values for different number of steps 
    elif conf=="TSstyle":
        at_pairs = [
            (2000, 1000),
            #(250, 1000),
            #(500, 1000),
            #(1000, 1000),
            #(5000, None),
            #(10000, 1000)
        ]
        
        num_steps_array = [2,3,4,5,6,7,8,9,10]
        neural_cell = LSTM
        tms = [60]
        #scenarios = ['standard', 'sporadic', 'undistributed']
        scenarios = ['standard']
        #directions = ["D", "U"]
        directions = ["D"]
        max_static_confs = [None]
        min_mcs_confs = [None]
        #source_folders = ["phase.2/attenuation_0", "phase.2/attenuation_30", "phase.2/attenuation_10"]
        source_folders = ["phase.2/attenuation_0"]#source_folders = ["megas", "megas10", "megas30"]
        nwslices = [ 1, 2]
        do_feature_selection=False
        do_transform_input = False
        choose_last=True
        do_train_resources = False
        script_function=run_configurations_per_cell 
        #script_function=run_configuration
        folder_name = "results-ts-bynumsteps"
        custom=f"-style_bycell"
        verbose=1
        ratio=(0.8, 0.2, 0)
        cell=None
        cells=range(0,21)
        
        features=[ 'Last_SE' ]
    else:
        assert 1==0, "Incorrect configuration"
        
    if num_steps_array is None:
        num_steps_array = [num_steps]
        
  
 
    jobs = 0
    with Pool(processes=8) as pool:
        for num_steps in num_steps_array:
            if learning_params is None and learning_family == "NN":
                learning_params = {
                    'hidden_layers': hl,
                    'hidden_units': hu,
                    'epochs': epochs,
                    'neural_cell': neural_cell,
                    'batch_size': batch_size,
                    'choose_last': choose_last,
                    'shuffle': shuffle,
                    # 'optimizer': optimizer,
                    # 'optimizer_rate': optimizer_rate,
                    # 'hidden_activation': hidden_activation,
                    # 'output_activation': output_activation,
                    # 'dropout': dropout,
                    # 'kernel_initializer': initializers.Constant(0.1),
                    # 'bias_initializer': initializers.Zeros(),
                    #'time_flat_dense': time_flat_dense,
                    'loss_fun': loss_fun,
                    'loss_function_params': {
                        'eps': eps,
                        'with_max_penalty': with_max_penalty
                    }
                }
            elif learning_params is None and learning_family == "GB":
                learning_params = {
                    "epochs": epochs,
                    "min_epochs": epochs//4
                }

            for source_folder in source_folders:
                #eps = alpha * eps_static
                for scenario in scenarios:
                    for tm in tms:
                        for max_static in max_static_confs:
                            for min_mcs in min_mcs_confs:
                                for direction in directions:
                                    for alpha, target_m in at_pairs:

                                        source_fname = f"{source_folder}/mega_block_{direction}_{scenario}_{tm}_{min_mcs}_{max_static}_sliced.parquet"
                                        #print(source_fname)

                                        # sum of two Trues is 2 and so on, so this code gives the number of non_default configurations where the first position is the default
                                        non_defaults =  (min_mcs != min_mcs_confs[0]) + \
                                            (max_static != max_static_confs[0]) + \
                                            (direction != directions[0]) + \
                                            (tm != tms[0]) + \
                                            (scenario != scenarios[0]) + \
                                            (source_folder != source_folders[0])+ \
                                            ( (alpha, target_m) != at_pairs[0]) 
                                        if non_defaults > 1:
                                            continue

                                        if not os.path.exists(source_fname):
                                            print("ERROR",source_fname,"does not exist!")
                                            continue

                                        mega_block_df = pd.read_parquet(source_fname)

                                        mega_grouped = mega_block_df.groupby("Slice")
                                        for nwslice in mega_grouped.groups:
                                            if target_m is None:
                                                target = target_mbps[direction][int(nwslice)] * 1000000
                                            else:
                                                target = target_m * 1000000
                                            #if non_defaults == 1 and nwslice != 2 and direction != 'U':
                                            #    continue
                                            if nwslice not in nwslices:
                                                continue
                                            nwslice_df = mega_grouped.get_group(nwslice)

                                            name = f"simulation-source_{source_folder.split('/')[-1].replace('-','.')}-tm_{tm}-maxstatic_{max_static}-minmcs_{min_mcs}-scenario_{scenario}-direction_{direction}-alpha_{alpha}-target_{target/1000000}-slice_{nwslice}-numsteps_{num_steps}{custom}"
                                            #name = f"params-correct-hl_{hl}-hu_{hu}-epochs_{epochs}"
                                            #fname = f"learning-simulation-sliced-timeseries/{name}.pickle"
                                            fname = f"{folder_name}/{name}.pickle"
                                            parent_folder = str(Path(fname).parent)
                                            if not os.path.exists(parent_folder):
                                                os.makedirs(parent_folder)
                                            if os.path.exists(fname) and not overwrite:
                                                print(f"Skipping {name}...")
                                                continue
                                            def error_callback(sim_name):
                                                def inside(ex):
                                                    print("Error in", sim_name,ex)
                                                    #raise e
                                                    traceback.print_exception(type(ex), ex, ex.__traceback__)
                                                    return
                                                return inside
                                            jobs += 1
                                            if not nono:
                                                pool.apply_async(
                                                    script_function,
                                                    #run_configurations_per_cell,
                                                    #run_configuration,
                                                    kwds = {
                                                        'aggregation': tm,
                                                        'iterations': 1,
                                                        'alpha': alpha,
                                                        'target_bps': target,
                                                        'source_folder': source_folder,
                                                        'fname': fname,
                                                        'scenario': scenario,
                                                        'max_static': max_static,
                                                        'direction': direction,
                                                        'min_mcs': min_mcs,
                                                        'cell': cell,
                                                        'cells': cells,
                                                        'features': features,
                                                        'source_df': (source_fname, nwslice_df),
                                                        'verbose': verbose,
                                                        'learning_family': learning_family,
                                                        'learning_params': learning_params,
                                                        #'scaler':  PowerTransformer(method='yeo-johnson'),
                                                        'ratio': ratio,
                                                        'reversed_split': False,
                                                        'time_min': 1600,
                                                        'num_steps': num_steps,
                                                        'time_flat_dense': time_flat_dense,
                                                        #'neural_cell': LSTM,
                                                        #'update': 300,
                                                        #'retries': 0,
                                                        'train_resources': do_train_resources,
                                                        #'kernel_initializer':"glorot_uniform",
                                                        #'bias_initializer':"zeros",
                                                        'do_transform_input': do_transform_input,
                                                        'do_feature_selection': do_feature_selection,
                                                        'epochs_between_reports': epochs_between_reports
                                                    },
                                                    error_callback= error_callback(fname)
                                                )
                                            else:
                                                print(fname)
        print("Done starting",jobs,"jobs")
        pool.close()
        pool.join()