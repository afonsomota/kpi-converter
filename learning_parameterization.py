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
import sys
import pickle
import copy
import resource


from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, LSTM, GRU
from keras import initializers
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from scipy.optimize import curve_fit, minimize
import scipy.stats as stats
import tensorflow as tf
import uuid
import lightgbm as lgb

        
sys.path.append("imports/")
import Util
from NRTables import MCS, PC_TBS, BIDIR_MCS
from cache_generator import *

def new_cost_function(beta, gamma=1):
    def get_cost(contracted, required):
        diff = contracted - required
        cost_rbs = np.empty(diff.shape)
        cost_rbs.fill(np.NaN)
        violation =  (contracted < required)
        cost_rbs[violation] = beta
        overprovisioning = (contracted >= required)
        cost_rbs[overprovisioning] = gamma * diff[overprovisioning]
        return cost_rbs
    return get_cost


def get_no_rbs_se(se, target):
    return np.ceil(target/(se*15000*12))


def sla_violations(reserved, required):
    notnan = ~np.isnan(reserved)
    violations = reserved.copy()
    violations[notnan] = (reserved < required)[notnan]
    return violations


def overprovisioning(reserved, required):
    assert reserved.shape == required.shape, f"Shape mismatch {reserved.shape} != {required.shape}"
    notnan = ~np.isnan(reserved)
    overprovisioning = reserved.copy()
    overprovisioning[notnan & (reserved < required)] = 0 
    overprovisioning[notnan & (reserved >= required)] = (reserved / required - 1)[notnan & (reserved >= required)]
    return overprovisioning


def transform_df_to_temporal_input(
    df,
    column_lists,
    aggr,
    num_steps,
    skip_steps = 10
):

    max_time = df.Time.max()
    one_time_set = np.arange(0,aggr * num_steps, aggr)
    time_sets_base = np.arange(0, max_time, skip_steps)
    time_sets = np.array(list(map((lambda x: x+one_time_set),time_sets_base)))

    cells_df = df.groupby("Cell")
    blocks = {}
    for i, columns in enumerate(column_lists):
        blocks[i] = []

    for cell in cells_df.groups:
        cell_df = cells_df.get_group(cell)
        for ts in time_sets:
            block = cell_df[cell_df.Time.isin(ts)]
            if len(ts) == len(block):
                for i, columns in enumerate(column_lists):
                    blocks[i].append(block[columns].values)

    return map(np.array, blocks.values())
    


def run_model_timeseries(
    model,
    alpha,
    target,
    features_df,
    predictions_df,
    max_se,
    aggregate=60,
    update=None,
    scaler=None,
    skip_steps=1,
    fixed_overprovisioning=None,
    is_flat_dense=False,
    trained_resources=False
):
    assert is_flat_dense or (features_df.shape[0] == predictions_df.shape[0] and predictions_df.shape[2] == 1), f"features: {features_df.shape}, predictions: {predictions_df.shape}"
    if update is None:
        update = aggregate
    
    se_scaler = max_se
    
    required_rbs = np.empty(features_df.shape[0])
    required_rbs[:] = np.NaN
    reserved_rbs = np.empty(features_df.shape[0])
    reserved_rbs[:] = np.NaN
    cost_rbs = np.empty(features_df.shape[0])
    cost_rbs[:] = np.NaN
    
    beta = alpha
    gamma = 1
    cost_fun = new_cost_function(beta, gamma)
    
    for i, entry in enumerate(features_df):
        
        if is_flat_dense:
            required_prime = predictions_df[i,0]*se_scaler
        else:
            if type(predictions_df[i,-1,0]) is list:
                predictions = np.array(predictions_df[i,-1,0])
            else:
                predictions = predictions_df[i,-1,0]
            required_prime = predictions*se_scaler
        
        if trained_resources:
            required = np.ceil(required_prime)
        else:
            required = get_no_rbs_se(required_prime, target)
            
        if type(required) is np.ndarray:
            required_rbs[i] = required.min()
        else:
            required_rbs[i] = required
        
        if type(model) != str:
            if is_flat_dense:
                predicted = model.predict(entry.reshape(1,entry.shape[0])).squeeze()*se_scaler
            else:
                predicted_raw = model.predict(entry.reshape(1,entry.shape[0],entry.shape[1])).squeeze(axis=(0,2))
                predicted = predicted_raw[-1]*se_scaler
                
        elif model == "Last":
            predicted_raw = predictions_df[i,-2,0]
            if type(predicted_raw) is list:
                predicted_raw = predicted_raw[0]
            predicted = predicted_raw*se_scaler
            
        if trained_resources:
            reserved = np.ceil(predicted)
        else:
            reserved = get_no_rbs_se(predicted, target)
        if fixed_overprovisioning is not None:
            reserved = np.ceil(reserved * (1 + fixed_overprovisioning))
        reserved_rbs[i] = reserved
        if type(required) is np.ndarray:
            #cost_rbs[i] = np.vectorize(lambda req: cost_fun(reserved_rbs,req))(required).mean()
            cost_rbs[i] = cost_fun(reserved,required).mean()
        else:
            cost_rbs[i] = cost_fun(reserved, required)
    violations_number = sla_violations(reserved_rbs, required_rbs)
    overprovisioning_value = overprovisioning(reserved_rbs, required_rbs)
    
    return cost_rbs, violations_number, overprovisioning_value, required_rbs, reserved_rbs
    

def run_model(
    model,
    alpha,
    target,
    dataset,
    max_se=1,
    skip_steps=15,
    cells=None,
    features=["Bytes_std", "Bytes_5", "Bytes_25","Bytes_50", "Bytes_75", "Bytes_95", 
              "SNR_std", "SNR_5", "SNR_25","SNR_50", "SNR_75", "SNR_95", 
              "Last_equally_distributed", "Last_SE"],
    scaler=None,
    fixed_overprovisioning=None,
    trained_resources=False
):
    np.warnings.filterwarnings('ignore')

    if scaler is not None:
        unscaled_df = dataset.copy()
        unscaled_values = scaler.inverse_transform(dataset[features])
        unscaled_df[features] = unscaled_values
        se_scaler = max_se
    else:
        unscaled_df = dataset.copy()
        se_scaler = 1
        
    
    if cells is not None:
        per_cell_dfs = dataset[dataset.Cell.isin(cells)].groupby("Cell")
        per_cell_unscaled = unscaled_df[unscaled_df.Cell.isin(cells)].groupby("Cell")
    else:
        per_cell_dfs = dataset.groupby("Cell")
        per_cell_unscaled = unscaled_df.groupby("Cell")
        cells = list(per_cell_dfs.groups.keys())
    

    beta = alpha
    gamma = 1
    target_bps = target
    if type(model) is not str:
        if len(dataset.shape) == 3:
            num_steps = dataset.shape[1]
        else:
            num_steps = 1
    

    if (~per_cell_dfs.count().RB_ratio.isna()).sum() == 0:
        max_trace = 0
        return np.empty((len(cells),max_trace)), np.empty((len(cells),max_trace)), np.empty((len(cells),max_trace)), np.empty((len(cells),max_trace)), np.empty((len(cells),max_trace))
    max_trace = np.ceil(per_cell_dfs.count().RB_ratio.max()/skip_steps).astype(int)

    required_rbs = np.empty((len(cells),max_trace))
    required_rbs[:] = np.NaN
    reserved_rbs = np.empty((len(cells),max_trace))
    reserved_rbs[:] = np.NaN
    cost_rbs = np.empty((len(cells),max_trace))
    cost_rbs[:] = np.NaN
    
    i = 0

    cost_fun = new_cost_function(beta, gamma)
    for cell in per_cell_dfs.groups:
        trace_df = per_cell_dfs.get_group(cell)
        unscaled_trace = per_cell_unscaled.get_group(cell)
        trace_np = trace_df[features].to_numpy()
        se_np = trace_df.RB_ratio.to_numpy().squeeze()
        idx = 0
        indexes = np.arange(0,len(trace_df), skip_steps).astype(int)
        steps_df = trace_df.iloc[indexes]
        required = get_no_rbs_se(steps_df.RB_ratio*se_scaler, target)
        required_rbs[i][:len(required)] = required
        if type(model) != str:
            prediction_raw =  model.predict(steps_df[features].values)
            predicted = prediction_raw.squeeze(axis=1)*se_scaler
            #if type(predicted) is not float and type(prediction_raw.squeeze()) is float:
            #    print("DANGER ADVERTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        elif model == "Last":
            predicted = unscaled_trace.Last_SE
        reserved = get_no_rbs_se(predicted, target)
        if fixed_overprovisioning is not None:
            reserved = np.ceil(reserved * (1 + fixed_overprovisioning))
        reserved_rbs[i][:len(reserved)] = reserved
        i+=1

    cost_rbs = cost_fun(reserved_rbs, required_rbs)
    violations_number = sla_violations(reserved_rbs, required_rbs)
    overprovisioning_value = overprovisioning(reserved_rbs, required_rbs)
    
    return cost_rbs, violations_number, overprovisioning_value, required_rbs, reserved_rbs



def split_samples(in_df, ratio=None, times=(6000,600,600), reverse=False):
    df = in_df.copy()
    df = df[~df.RB_ratio.isna()]
    df = df[~df.Last_SE.isna()]
    if ratio is not None:
        if reverse:
            ratio = (ratio[2], ratio[1], ratio[0])
        time_stamps = df.Time.unique()
        time_stamps = sorted(time_stamps)
        t_0_index = int(len(time_stamps)*ratio[0])
        training_mask = (df.Time >= 0) & (df.Time < time_stamps[t_0_index])
        t_1_index = min(t_0_index + int(len(time_stamps)*ratio[1]), len(time_stamps)-1)
        testing_mask = (df.Time >= time_stamps[t_0_index]) & (df.Time < time_stamps[t_1_index])
        t_2_index = min(t_1_index + int(len(time_stamps)*ratio[2]), len(time_stamps)-1)
        validation_mask = (df.Time >= time_stamps[t_1_index]) & (df.Time < time_stamps[t_2_index])
    elif times is not None:
        training_times = times[0]
        testing_times = times[1] if len(times) == 3 else 0
        validation_times = times[2] if len(times) == 3 else times[1]
        t = 0
        training_mask = (df.Time >= t) & (df.Time < t + training_times)
        t += training_times
        testing_mask = (df.Time >= t) & (df.Time < t + testing_times)
        t += testing_times
        validation_mask = (df.Time >= t) & (df.Time < t + validation_times)
    
    if reverse:
        return df.loc[validation_mask].copy() , df.loc[testing_mask].copy() , df.loc[training_mask].copy()
    else:
        return df.loc[training_mask].copy() , df.loc[testing_mask].copy() , df.loc[validation_mask].copy()


def alpha_omc_linear_se(alpha, scaler=5.5547, eps=0.0001, target=10000000):
    @tf.function
    def compare(y, y_hat):
        LARGE_NUM = 1797693134861627367219837217831273612
        
        aim_se = y*scaler + 0.00001
        pred_se = y_hat*scaler + 0.00001
        get_rbs = (lambda t, se: t/(15000*12*se))
        x = tf.clip_by_value(get_rbs(target, pred_se), -LARGE_NUM, LARGE_NUM) - tf.clip_by_value(get_rbs(target, aim_se), -LARGE_NUM, LARGE_NUM)
        
        bp = 1000
        reference_value = 0
        less_than_zero = tf.cast(tf.math.less_equal(x,reference_value), tf.float32)
        if eps != 0:
            in_zero_zone = tf.cast(tf.math.logical_and(tf.math.greater(x,reference_value), tf.math.less_equal(x,reference_value+eps)), tf.float32)
        greater_than_zero = tf.cast(tf.math.greater(x,reference_value+eps), tf.float32)
        greater_than_bp = tf.cast(tf.math.greater(x,bp), tf.float32)
        
        a = tf.multiply(less_than_zero, 1 - eps*x/alpha)
        if eps != 0:
            b = tf.multiply(in_zero_zone, 1 - (1/eps)*x)
        c = tf.multiply(greater_than_zero, x*(1/(eps+alpha)) - eps/(eps+alpha))
        
        if eps != 0:
            loss = a + b + c
        else:
            loss = a + c
         
        return loss
    return compare


def alpha_omc_linear_se_with_max_penalty(alpha, scaler=5.5547, eps=0.0001, target=10000000, penalty_multiplier=100):
    @tf.function
    def compare(y, y_hat):
        LARGE_NUM = 1797693134861627367219837217831273612
        
        aim_se = y*scaler + 0.00001
        pred_se = y_hat*scaler + 0.00001
        get_rbs = (lambda t, se: t/(15000*12*se))
        x = tf.clip_by_value(get_rbs(target, pred_se), -LARGE_NUM, LARGE_NUM) - tf.clip_by_value(get_rbs(target, aim_se), -LARGE_NUM, LARGE_NUM)
        
        bp = 1000
        reference_value = 0
        less_than_zero = tf.cast(tf.math.less_equal(x,reference_value), tf.float32)
        if eps != 0:
            in_zero_zone = tf.cast(tf.math.logical_and(tf.math.greater(x,reference_value), tf.math.less_equal(x,reference_value+eps)), tf.float32)
        greater_than_zero = tf.cast(tf.math.greater(x,reference_value+eps), tf.float32)
        greater_than_bp = tf.cast(tf.math.greater(x,bp), tf.float32)
        greater_than_max = tf.cast(tf.math.greater(y_hat,1), tf.float32)
        max_diff_penalty = penalty_multiplier*(y_hat - 1)**2
        
        a = tf.multiply(less_than_zero, 1 - eps*x/alpha)
        if eps != 0:
            b = tf.multiply(in_zero_zone, 1 - (1/eps)*x)
        c = tf.multiply(greater_than_zero, x*(1/(eps+alpha)) - eps/(eps+alpha))
        penalty = tf.multiply(greater_than_max, max_diff_penalty)
        
        if eps != 0:
            loss = a + b + c + penalty
        else:
            loss = a + c + penalty
         
        return loss
    return compare

def alpha_omc_linear_se_with_max_penalty_nok(alpha, scaler=5.5547, eps=0.0001, target=10000000, penalty_multiplier=100):
    @tf.function
    def compare(y, y_hat):
        LARGE_NUM = 1797693134861627367219837217831273612
        
        aim_se = y*scaler + 0.00001
        pred_se = y_hat*scaler + 0.00001
        get_rbs = (lambda t, se: t/(15*12*se))
        x = tf.clip_by_value(get_rbs(target, pred_se), -LARGE_NUM, LARGE_NUM) - tf.clip_by_value(get_rbs(target, aim_se), -LARGE_NUM, LARGE_NUM)
        
        bp = 1000
        reference_value = 0
        less_than_zero = tf.cast(tf.math.less_equal(x,reference_value), tf.float32)
        if eps != 0:
            in_zero_zone = tf.cast(tf.math.logical_and(tf.math.greater(x,reference_value), tf.math.less_equal(x,reference_value+eps)), tf.float32)
        greater_than_zero = tf.cast(tf.math.greater(x,reference_value+eps), tf.float32)
        greater_than_bp = tf.cast(tf.math.greater(x,bp), tf.float32)
        greater_than_max = tf.cast(tf.math.greater(y_hat,1), tf.float32)
        max_diff_penalty = penalty_multiplier*(y_hat - 1)**2
        
        a = tf.multiply(less_than_zero, 1 - eps*x/alpha)
        if eps != 0:
            b = tf.multiply(in_zero_zone, 1 - (1/eps)*x)
        c = tf.multiply(greater_than_zero, x*(1/(eps+alpha)) - eps/(eps+alpha))
        penalty = tf.multiply(greater_than_max, max_diff_penalty)
        
        if eps != 0:
            loss = a + b + c + penalty
        else:
            loss = a + c + penalty
         
        return loss
    return compare


def alpha_omc_linear_rbs(alpha, scaler, eps=0.0001, target=10000000):
    
    @tf.function
    def compare_rbs(y, y_hat):
        
        x = (y_hat*scaler - y*scaler)
        
        reference_value = 0
        less_than_zero = tf.cast(tf.math.less_equal(x,reference_value), tf.float32)
        if eps != 0:
            in_zero_zone = tf.cast(tf.math.logical_and(tf.math.greater(x,reference_value), tf.math.less_equal(x,reference_value+eps)), tf.float32)
        greater_than_zero = tf.cast(tf.math.greater(x,reference_value+eps), tf.float32)
        greater_than_max = tf.cast(tf.math.greater(y_hat,1), tf.float32)
        
        a = tf.multiply(less_than_zero, 1 - 100*eps*x/alpha)
        b = tf.multiply(in_zero_zone, 1 - (1/eps)*x)
        c = tf.multiply(greater_than_zero, x*(1/(eps+alpha)) - eps/(eps+alpha))
        
        loss = a + b + c 
         
        return loss
    return compare_rbs


def standard_asymmetric_rbs(alpha, scaler, eps=0.01, target=None):
    
    high_slope = (1/(eps * alpha))**2
    low_slope = (1/(alpha * (1 - eps)))**2
    
    @tf.function
    def compare_rbs(y, y_hat):
        
        x = (y_hat*scaler - y*scaler)
        
        less_than_eps = tf.cast(tf.math.less_equal(x,eps * alpha), tf.float32)
        greater_than_eps = tf.cast(tf.math.greater(x,eps * alpha), tf.float32)
        
        a = tf.multiply(less_than_eps, high_slope*(x-eps*alpha)**2)
        b = tf.multiply(greater_than_eps, low_slope*(x-eps*alpha)**2)
        loss = a + b
        
        return loss
    return compare_rbs

def standard_asymmetric_rbs_with_log(alpha, scaler, eps=0.0001, target=None):
    
    high_slope = (1/(eps * alpha))**2
    low_slope = (1/(alpha * (1 - eps)))**2
    
    @tf.function
    def compare_rbs(y, y_hat):
        
        x = (y_hat*scaler - y*scaler)
        
        less_than_zero = tf.cast(tf.math.less_equal(x,0), tf.float32)
        less_than_eps = tf.cast(tf.math.logical_and(tf.math.less_equal(x,eps * alpha), tf.math.greater(x,0))
                                , tf.float32)
        greater_than_eps = tf.cast(tf.math.greater(x,eps * alpha), tf.float32)
        
        a = tf.math.multiply_no_nan((2/alpha**2) * tf.math.log(-(alpha/eps)*x + 1) + 1, less_than_zero)
        #a = tf.multiply(less_than_zero, 1 - 100*eps*x/alpha)
        b = tf.multiply(less_than_eps, high_slope*(x-eps*alpha)**2)
        c = tf.multiply(greater_than_eps, low_slope*(x-eps*alpha)**2)
        loss = a + b + c
        
        return loss
    return compare_rbs


class StdOverProvisioningModel:

    #error_std  = np.std(y_training - model_gb.predict(x_training))

#     def __init__(self):
#         self.alpha = alpha
#         self.error_std  = np.std(y_training - model_gb.predict(x_training))

    def __init__(self, alpha, y_training, x_training, model, force_2d=False):
        self.alpha = alpha
        self.model = model
        self.error_std = np.std(y_training - self.model.predict(x_training))
        if self.error_std == 0:
            self.error_std += 0.0000001
        self.force_2d = force_2d
        
        guess = self.error_std
        #bounds = [(-1,5)]
        cost_error = (lambda h: self.alpha*(1-stats.norm.cdf(h/self.error_std)) + h*stats.norm.cdf(h/self.error_std) + self.error_std*stats.norm.pdf(h/self.error_std)/stats.norm.cdf(h/self.error_std))
        d_cost_error = (lambda h: (h-self.alpha)*stats.norm.pdf(h/self.error_std)/self.error_std - stats.norm.pdf(h/self.error_std)**2/stats.norm.cdf(h/self.error_std)**2 - (h/self.error_std)*(stats.norm.pdf(h/self.error_std)/stats.norm.cdf(h/self.error_std)) + stats.norm.cdf(h/self.error_std))
        solved = minimize(cost_error, guess, jac=d_cost_error)
        self.op = solved.x

    def predict(self, x_eval):
#         guess = self.error_std
#         bounds = [(-1,5)]
#         def get_overprovisioning(y_pred_single):
#             cost_error = (lambda h: self.alpha*(1-stats.norm.cdf(h/self.error_std)) + h*stats.norm.cdf(h/self.error_std) + self.error_std*stats.norm.pdf(h/self.error_std)/stats.norm.cdf(h/self.error_std))
#             d_cost_error = (lambda h: (h-self.alpha)*stats.norm.pdf(h/self.error_std)/self.error_std - stats.norm.pdf(h/self.error_std)**2/stats.norm.cdf(h/self.error_std)**2 - (h/self.error_std)*(stats.norm.pdf(h/self.error_std)/stats.norm.cdf(h/self.error_std)) + stats.norm.cdf(h/self.error_std))
#             solved = minimize(cost_error, guess, bounds=bounds, jac=d_cost_error)
#             return y_pred_single + solved.x
#         vectored_overprovisioning = np.vectorize(get_overprovisioning)
        if self.force_2d and len(x_eval.shape)==3:
            x = x_eval.squeeze(axis=1)
        else:
            x = x_eval
        just_predictions = self.model.predict(x)
#         overprovisioned_predictions = vectored_overprovisioning(just_predictions)
        overprovisioned_predictions = just_predictions + self.op
        if self.force_2d and len(x_eval.shape)==3:
            overprovisioned_predictions = np.expand_dims(overprovisioned_predictions, axis=(1,2))
        return overprovisioned_predictions

                 
class GlobalOverProvisioningModel:
    def __init__(self, alpha, y_training, x_training, model, force_2d=False):
        y_pred = model.predict(x_training)
        def cost_function_last(h):
            residuals = y_pred + h - y_training
            residuals[residuals<0] = alpha
            return residuals.mean()
        guess = (y_pred - y_training).std()
        solve = minimize(cost_function_last, guess)
        self.overprovisioning = solve.x
        self.model = model
        self.force_2d = force_2d

    def predict(self, x_eval):
        if self.force_2d and len(x_eval.shape)==3:
            x = x_eval.squeeze(axis=1)
        else:
            x = x_eval
        prediction = self.model.predict(x) + self.overprovisioning
        if self.force_2d and len(x_eval.shape)==3:
            prediction = np.expand_dims(prediction, axis=(1,2))
        return prediction


def transform_input(in_df, features, max_se, add_features=False):
    to_scale = features.copy()
    snr_cols = ["SNR_5", "SNR_25", "SNR_50", "SNR_75", "SNR_95"]
    snr_cols = list(filter(lambda x: x in features, snr_cols))
    in_df[snr_cols] = np.log2(1+10**(in_df[snr_cols]))
    for col in ["Last_RB_ratio", "Last_equally_distributed"]:
        if col in features:
            in_df[col] = (in_df[col]* 8 / (12*15))
    in_df["Reverse_SE"] = max_se - in_df["Last_SE"]
    if "Reverse_SE" not in features and add_features:
        features.append("Reverse_SE")
#     if "Last_equally_distributed" in features:
#         in_df["Last_equally_distributed"] = (in_df["Last_RB_ratio"] - in_df["Last_equally_distributed"])
    in_df["Last_SE_diff"] = in_df["Last_SE"].diff().fillna(0)
    for col in ["Last_RB_ratio", "Last_equally_distributed", "Reverse_SE", "Last_SE_diff", "Last_SE"]:
        in_df[col] /= max_se
        if col in to_scale:
            to_scale.remove(col)
    
    traffic_cols = ["Bytes_5", "Bytes_25", "Bytes_50", "Bytes_75", "Bytes_95"]
    traffic_cols = list(filter(lambda x: x in features, traffic_cols))
    total_bytes_col = "Bytes_sum" if "Bytes_sum" in in_df else "Bytes_95"
    for col in traffic_cols:
        in_df[col] = in_df[col] / in_df[total_bytes_col].replace(0,1)
        if col in to_scale:
            to_scale.remove(col)
    
    return in_df, features, to_scale

def feature_selection(in_df, features, use_weak=True, main_feature='Last_SE'):
    
    
    corr = in_df[features].corr()
    
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= 0.9:
                if columns[i] and in_df[features].columns[i] != 'Last_SE':
                    columns[i] = False
                elif columns[i] and columns[j]:
                    columns[j] = False
    selected_features = in_df[features].columns[columns]
#    selected_features = features # DELETE
    forest_boruta = RandomForestRegressor(
       n_jobs = -1,
       max_depth = 7
    )
    boruta = BorutaPy(
       estimator = forest_boruta, 
       n_estimators = 'auto',
       alpha=0.005,
       max_iter = 100
    )
    
    boruta.fit(in_df[selected_features].values, in_df["RB_ratio"].values)

    strong_features = in_df[selected_features].columns[boruta.support_].to_list()
    weak_features = in_df[selected_features].columns[boruta.support_weak_].to_list()
    
    if use_weak:
        selected_features = strong_features + weak_features
    else:
        selected_features = strong_features
        
    if len(selected_features) == 0:
        selected_features = ["Last_SE"]

    return selected_features


def one_hot_encode(original_dataframe, feature_to_encode, features):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode], prefix=feature_to_encode)
    features += list(dummies.columns)
    res = pd.concat([original_dataframe, dummies], axis=1)
    return res, features
            

def get_comp_resouces():
    comp_resources = resource.getrusage(resource.RUSAGE_SELF)
    return np.array([comp_resources[0], comp_resources[1], comp_resources[2]])


def learn_neural(split_dataset, learning_params, return_obj, alpha, target_bps, predictor_scaler=None, verbose=0):
    
    i = learning_params['current_iteration']
    if len(split_dataset["training"]["features"].shape) == 3:
        time_flat_dense = False
        len_features = split_dataset["training"]["features"].shape[2]
        num_steps = split_dataset["training"]["features"].shape[1]
    else:
        time_flat_dense = True
        len_features = split_dataset["training"]["features"].shape[1]
        num_steps = 1
    if learning_params['loss_fun'] is None:
        if learning_params['train_resources']:
            loss_fun = alpha_omc_linear_rbs(alpha, scaler=predictor_scaler, eps=learning_params['loss_function_params']['eps'], target=target_bps)
        elif learning_params['loss_function_params']['with_max_penalty']:
            loss_fun = alpha_omc_linear_se_with_max_penalty(alpha, eps=learning_params['loss_function_params']['eps'], target=target_bps, scaler=predictor_scaler)
        else:
            loss_fun = alpha_omc_linear_se(alpha, eps=learning_params['loss_function_params']['eps'], target=target_bps, scaler=predictor_scaler)
    elif type(learning_params['loss_fun']) is not str:
        loss_fun = learning_params['loss_fun'](alpha, scaler=predictor_scaler, eps=learning_params['loss_function_params']['eps'], target=target_bps)
        
    model = Sequential()
    if num_steps is None or learning_params['time_flat_dense']:
        input_dim =  (len_features)
        time_distributed = (lambda x: x)
    else:
        input_dim = (num_steps, len_features)
        time_distributed = TimeDistributed
    if learning_params['neural_cell'] == Dense:
        return_sequences = False
    else:
        return_sequences = True
    if return_sequences:
        if learning_params['hidden_layers'] == -1:
            model.add(learning_params['neural_cell'](
                1, 
                input_shape = input_dim, 
                activation=learning_params['output_activation'],
                return_sequences=return_sequences,
                kernel_initializer=learning_params['kernel_initializer'],
                bias_initializer=learning_params['bias_initializer']
            ))
        else:
            model.add(learning_params['neural_cell'](
                len_features, 
                input_shape = input_dim,
                return_sequences=return_sequences,
                kernel_initializer=learning_params['kernel_initializer'],
                bias_initializer=learning_params['bias_initializer']
            ))
            if learning_params['dropout'] > 0:
                model.add(Dropout(learning_params['dropout']))
            for layer in range(learning_params['hidden_layers']):
                model.add(learning_params['neural_cell'](
                    units=learning_params['hidden_units'],
                    return_sequences=return_sequences,
                    activation=learning_params['hidden_activation'],
                    kernel_initializer=learning_params['kernel_initializer'],
                    bias_initializer=learning_params['bias_initializer']
                ))
                if learning_params['dropout'] > 0:
                    model.add(Dropout(learning_params['dropout']))

            model.add(time_distributed(Dense(
                1,
                activation=learning_params['output_activation'],
                kernel_initializer=learning_params['kernel_initializer'],
                bias_initializer=learning_params['bias_initializer']
            )))
    elif not time_flat_dense and num_steps is not None:
        input_dim = (num_steps, len_features)
        input_dense_dim = len_features
        if learning_params['hidden_layers'] == -1:
            model.add(learning_params['neural_cell'](
                1, 
                input_shape = input_dim, 
                activation=learning_params['output_activation'],
                kernel_initializer=learning_params['kernel_initializer'],
                bias_initializer=learning_params['bias_initializer']
            ))
        else:
            model.add(learning_params['neural_cell'](
                input_dense_dim, 
                input_shape = input_dim,
                kernel_initializer=learning_params['kernel_initializer'],
                bias_initializer=learning_params['bias_initializer']
            ))
            if learning_params['dropout'] > 0:
                model.add(Dropout(learning_params['dropout']))
            for layer in range(learning_params['hidden_layers']):
                model.add(learning_params['neural_cell'](
                    units=learning_params['hidden_units'],
                    activation=learning_params['hidden_activation'],
                    kernel_initializer=learning_params['kernel_initializer'],
                    bias_initializer=learning_params['bias_initializer']
                ))
                if learning_params['dropout'] > 0:
                    model.add(Dropout(learning_params['dropout']))

            model.add(time_distributed(Dense(
                1,
                activation=learning_params['output_activation'],
                kernel_initializer=learning_params['kernel_initializer'],
                bias_initializer=learning_params['bias_initializer']
            )))
    else:
        if num_steps is not None:
            input_dim = num_steps * len_features
            input_dense_dim = num_steps * len_features #len_features
        else:
            input_dim = input_dense_dim = len_features
        if learning_params['hidden_layers'] == -1:
            model.add(learning_params['neural_cell'](
                1, 
                input_dim = input_dim, 
                activation=learning_params['output_activation'],
                kernel_initializer=learning_params['kernel_initializer'],
                bias_initializer=learning_params['bias_initializer']
            ))
        else:
            model.add(learning_params['neural_cell'](
                input_dense_dim, 
                input_dim = input_dim,
                kernel_initializer=learning_params['kernel_initializer'],
                bias_initializer=learning_params['bias_initializer']
            ))
            if learning_params['dropout'] > 0:
                model.add(Dropout(learning_params['dropout']))
            for layer in range(learning_params['hidden_layers']):
                model.add(learning_params['neural_cell'](
                    units=learning_params['hidden_units'],
                    activation=learning_params['hidden_activation'],
                    kernel_initializer=learning_params['kernel_initializer'],
                    bias_initializer=learning_params['bias_initializer']
                ))
                if learning_params['dropout'] > 0:
                    model.add(Dropout(learning_params['dropout']))

            model.add(time_distributed(Dense(
                1,
                activation=learning_params['output_activation'],
                kernel_initializer=learning_params['kernel_initializer'],
                bias_initializer=learning_params['bias_initializer']
            )))

    opt = learning_params['optimizer'](learning_rate=learning_params['optimizer_rate'])
    model.compile(loss=loss_fun, optimizer=opt)

    if verbose == 2:
        print(model.summary())

    #return_obj['learning_history'].append(history)
    model_fname = None
    while True:
        model_fname = "models/" + str(uuid.uuid4()) + ".hd"
        if not path.exists(model_fname):
            break

    class resourceCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % learning_params['monitoring_params']['epochs_between_reports'] == 0:
                if verbose >= 2:
                    print("Resource usage in epoch",epoch,"before model calculation:", get_comp_resouces())
                epoch_idx = epoch//learning_params['monitoring_params']['epochs_between_reports']
                return_obj['comp_resources']['by_epoch'][i, epoch_idx, 0] = epoch
                return_obj['comp_resources']['by_epoch'][i, epoch_idx, 1:4] = get_comp_resouces()

    steps_per_epoch = np.ceil(len(split_dataset["training"]["features"])/learning_params['batch_size']).astype(int)
    if learning_params['shuffle'] is not None:
        steps_per_epoch = np.ceil(learning_params['shuffle'] * steps_per_epoch).astype(int) 
    
    if learning_params['choose_last']:
        last_n = np.floor(0.2*len(split_dataset["training"]["features"])).astype(int)
        x_validation = split_dataset["training"]["features"][-last_n:]
        y_validation = split_dataset["training"]["learning-y"][-last_n:]
        monitor = 'val_loss'
    else:
        x_validation = split_dataset["test"]["features"]
        y_validation = split_dataset["test"]["learning-y"]
        monitor = 'loss'

    model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_fname,
        save_weights_only=True,
        monitor=monitor,
        mode='min',
        save_best_only=True
    )
    history = model.fit(
        split_dataset["training"]["features"],
        split_dataset["training"]["learning-y"],
        epochs=learning_params['epochs'],
        batch_size=learning_params['batch_size'],
        verbose=max(0,verbose-1),
        validation_data=(x_validation, y_validation),
        #validation_data=(training_features[-last_n:], split_dataset["training"]["learning-y"][-last_n:]),
        callbacks=[model_callback, resourceCallback()],
        shuffle=(learning_params['shuffle'] is not None)
        )
    return_obj['comp_resources']['after'][i] = get_comp_resouces()
    model.load_weights(model_fname)
    model.save(model_fname)
    return_obj['models'].append(model_fname)
    
    return model


def learn_gb(split_dataset, learning_params, return_obj, alpha, target_bps, predictor_scaler=None, verbose=0):
    i = learning_params['current_iteration']
    
    x_training = split_dataset["training"]["features"]
    y_training = split_dataset["training"]["learning-y"]
    x_test = split_dataset["test"]["features"]
    y_test = split_dataset["test"]["learning-y"]
    
    if len(split_dataset["training"]["features"].shape) == 3:
        time_flat_dense = False
        len_features = split_dataset["training"]["features"].shape[2]
        num_steps = split_dataset["training"]["features"].shape[1]
        assert num_steps == 1, "Gradient boosting only work with time-flat inputs"
        x_training = x_training.squeeze(axis=1)
        y_training = y_training.squeeze(axis=1)
        x_test = x_test.squeeze(axis=1)
        y_test = y_test.squeeze(axis=1)
    else:
        time_flat_dense = True
        len_features = split_dataset["training"]["features"].shape[1]
        num_steps = 1
        
    lgb_train = lgb.Dataset(x_training, y_training)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    
    params = {
        'boosting_type': learning_params['boosting_type'],
        'objective': learning_params['objective'],
        'metric': learning_params['metric'],
        'num_leaves': learning_params['num_leaves'],
        'learning_rate': learning_params['learning_rate'],
        'feature_fraction': learning_params['feature_fraction'],
        'bagging_fraction': learning_params['bagging_fraction'],
        'bagging_freq': learning_params['bagging_freq'],
        'verbosity': verbose-1,
        'force_col_wise':True
    }
    
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=learning_params['epochs'],
        valid_sets=lgb_eval,
        verbose_eval= (verbose>=2),
        early_stopping_rounds=learning_params['min_epochs']
    )
    
    return gbm


def organize_dataset(datasets, num_steps, aggregation, features, predictions, simulation_cols=None, time_flat_dense=False):
    training = datasets[0]
    test = datasets[1] if len(datasets) > 1 else pd.DataFrame(columns=training.columns)
    validation = datasets[2] if len(datasets) > 2 else pd.DataFrame(columns=training.columns)
    
    if simulation_cols is None:
        simulation_cols = predictions
    
    if num_steps is None:
        training_features = training[features]
        training_predictions = training[predictions]
        training_simulation = training[simulation_cols]
        test_features = test[features]
        test_predictions = test[predictions]
        test_simulation = test[simulation_cols]
        validation_features = validation[features] if not validation.empty else np.zeros((0,0))
        validation_predictions = validation[predictions] if not validation.empty else np.zeros((0,0))
        validation_simulation = validation[simulation_cols] if not validation.empty else np.zeros((0,0))
            
    else:
        training_features, training_predictions, training_simulation =  transform_df_to_temporal_input(training, [features, predictions, simulation_cols], aggregation, num_steps)
        test_features, test_predictions, test_simulation =  transform_df_to_temporal_input(test, [features, predictions, simulation_cols], aggregation, num_steps) if not test.empty else (np.zeros((0,0,0)), np.zeros((0,0,0)), np.zeros((0,0,0)))
        validation_features, validation_predictions, validation_simulation =  transform_df_to_temporal_input(validation, [features, predictions, simulation_cols], aggregation, num_steps) if not validation.empty else (np.zeros((0,0,0)), np.zeros((0,0,0)), np.zeros((0,0,0)))

        if time_flat_dense:
            training_features = training_features.reshape(training_features.shape[0], training_features.shape[1]*training_features.shape[2])
            training_predictions = training_predictions[:, -1]
            test_features = test_features.reshape(test_features.shape[0], test_features.shape[1]*test_features.shape[2])
            test_predictions = test_predictions[:, -1]
            if not validation.empty:
                validation_features = validation_features.reshape(validation_features.shape[0], validation_features.shape[1]*validation_features.shape[2])
                validation_predictions = validation_predictions[:, -1]
            else:
                validation_features, validation_predictions, validation_simulation = np.zeros((0,0)), np.zeros((0,0)), np.zeros((0,0))
                
    split_dataset = {"training": {}, "test": {}, "validation": {}}

    split_dataset["training"]["features"] =  training_features
    split_dataset["training"]["learning-y"] =  training_predictions
    split_dataset["training"]["simulation-y"] =  training_simulation
    split_dataset["training"]["predictions"] =  training_simulation
    split_dataset["test"]["features"] =  test_features
    split_dataset["test"]["learning-y"] =  test_predictions
    split_dataset["test"]["simulation-y"] =  test_simulation
    split_dataset["test"]["predictions"] =  test_simulation
    split_dataset["validation"]["features"] =  validation_features
    split_dataset["validation"]["simulation-y"] =  validation_simulation
    split_dataset["validation"]["predictions"] =  validation_simulation
    print(training_features.shape, 
          training_predictions.shape,
          test_features.shape , 
          test_predictions.shape, 
          validation_predictions.shape)

    return split_dataset
    

class DummySymmetricModel:
    def predict(self, x):
        if len(x.shape)==3:
            assert x.shape[2] == 1
        else:
            assert x.shape[1] == 1
        return x

        
def run_configuration(
    aggregation=60,
    update=60,
    scenario='standard',
    direction='D',
    cell=None,
    learning_family="NN",
    learning_params={},
    iterations=10,
    features=['SNR_std', 'SNR_5', 'SNR_25', 'SNR_50', 'SNR_75', 'SNR_95',
            'Bytes_std', 'Bytes_5', 'Bytes_25', 'Bytes_50', 'Bytes_75', 'Bytes_95',
            'Last_equally_distributed', 'Reverse_SE', 'Last_SE'
           ],
    predictions=['RB_ratio'],
    max_se=None,
    alpha=2000,
    target_bps=10000000,
    verbose=1,
    min_mcs=None,
    max_static=None,
    source_folder="megas",
    fname=None,
    source_df=None,
    num_steps=None,
    scaler=None,
    ratio=(0.8,0.1,0.1),
    reversed_split=False,
    time_min = 0,
    time_max=None,
    retries = 0,
    train_resources=False,
    register_training=False,
    do_transform_input=False,
    pre_calculate_mse=None,
    do_feature_selection=True,
    time_flat_dense=False,
    bw_rbs=True,
    cells = None,
    epochs_between_reports=10
):
    
    if learning_family == "NN":
        default_params = {
            'hidden_layers': 0,
            'hidden_units': 20,
            'epochs': 500,
            'neural_cell': Dense,
            'batch_size': 50,
            'optimizer': tf.keras.optimizers.Adam,
            'optimizer_rate': 0.0002,
            'hidden_activation': 'relu',
            'output_activation': 'linear',
            'dropout': 0.1,
            'kernel_initializer': initializers.Constant(0.1),
            'bias_initializer': initializers.Zeros(),
            'time_flat_dense': time_flat_dense,
            'train_resources': train_resources,
            'choose_last': True,
            'shuffle': None,
            'num_steps': num_steps,
            'loss_fun': None,
            'loss_function_params': {
                'eps': 0.01,
                'with_max_penalty': False
            },
            'monitoring_params': {
                'epochs_between_reports': epochs_between_reports
            }
        }
        
        default_params.update(learning_params)
        learning_params = default_params
        if verbose >= 2:
            print(learning_params)
    elif learning_family == "GB":
        default_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': 31,
            'learning_rate': 0.15,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'epochs': 30,
            'min_epochs': 5
        }
        default_params.update(learning_params)
        learning_params = default_params
    elif learning_family == "Simple":
        features = ["Last_SE"]
        train_resources = True
        do_transform_input=False
        do_feature_selection=False
        num_steps=1
        learning_params={}
    elif learning_family == "Last":
        features = ["Last_SE"]
        train_resources = True
        do_transform_input=False
        do_feature_selection=False
        num_steps=1
        learning_params={}
    
    if max_se is None:
        max_se = BIDIR_MCS[direction][27]['se']
        
    if type(pre_calculate_mse) is str and pre_calculate_mse == "default":
        pre_calculate_mse = {
            'epochs': 100,
            'hl': 1,
            'hu': 5,
            'loss': 'mse',
            'features': features
        }
    
    if fname is not None and verbose:
        print(f"Running {fname}...")
    start = time.time()
    if source_df is not None:
        source_fname = source_df[0]
        mega_block_df = source_df[1]
    else:
        source_fname = f"{source_folder}/mega_block_{direction}_{scenario}_{aggregation}_{min_mcs}_{max_static}.parquet"
        mega_block_df = pd.read_parquet(source_fname)
    
    if cell is not None:
        mega_block_df = mega_block_df[mega_block_df.Cell == cell]
        cells = [cell]
    elif cells is not None:
        mega_block_df = mega_block_df[mega_block_df.Cell.isin(cells)]
        
    
    learning_df = mega_block_df.copy()
    
    learning_df = learning_df[~learning_df.RB_ratio.isna()]
    learning_df = learning_df[~learning_df.Last_RB_ratio.isna()]
    learning_df = learning_df[~learning_df.Last_SE.isna()]
    learning_df = learning_df[learning_df.Time >= time_min]
    #correct dataset, single value standard deviation is nan, changing to zero
    learning_df.Bytes_std = learning_df.Bytes_std.fillna(0)
    learning_df.SNR_std = learning_df.SNR_std.fillna(0)
    
    
    
    if update != aggregation:
        n_future = update//aggregation
        learning_df["Timeline"] = learning_df.Time % aggregation
        grouped =learning_df.groupby(["Slice","Cell","Timeline"])
        def set_rolling(x):
            x["Rolling_RB_ratio"] = x["RB_ratio"].rolling(n_future).min().shift(-n_future+1)
            x["RB_ratio_simulation"] =  [window.to_list() for window in x["RB_ratio"].rolling(window=n_future)]
            x["RB_ratio_simulation"] = x["RB_ratio_simulation"].shift(-n_future+1)
            return x
        learning_df = grouped.apply(set_rolling)
        predictions = ["Rolling_RB_ratio"]
        simulation_cols = ["RB_ratio_simulation"]
    else:
        simulation_cols = predictions
    
    # Transform input: features and resource or SE-based
    if do_transform_input:
        learning_df, features, to_scale = transform_input(learning_df, features, max_se)
    else:
        to_scale = features.copy() 
    if train_resources:
        if do_transform_input:
            learning_df['Last_SE'] *= max_se
        else:
            to_scale.remove('Last_SE')
        if bw_rbs:
            t = 1000
        else:
            t=1
        #assert not (learning_df['Last_SE'] == 0).any(), learning_df[lerning_df['Last_SE']==0]
        learning_df[predictions] = target_bps/(15*t*12*learning_df[predictions])
        learning_df['Last_SE'] = target_bps/(15*t*12*learning_df['Last_SE'])
        alpha_calc = alpha*1000/t
        alpha = alpha_calc
        max_se = learning_df['RB_ratio'].max()
        assert not np.isinf(max_se)
        learning_df['RB_ratio'] = learning_df.RB_ratio / max_se
        learning_df['Last_SE'] = learning_df.Last_SE / max_se
    else:
        alpha_calc = alpha
        learning_df['RB_ratio'] = learning_df.RB_ratio / max_se
        
    #One hot encode cells if no cell filter
    if cell is None:
        learning_df, features = one_hot_encode(learning_df, 'Cell', features)
        
    #Scaler-fit
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
    if to_scale:
        scaler.fit(learning_df.loc[:, to_scale])
    
    #Split
    training, test, validation = split_samples(learning_df, ratio=ratio, reverse=reversed_split)
    
    #Scale transform
    if to_scale:
        training.loc[:, to_scale] = scaler.transform(training.loc[:, to_scale])
        if not test.empty:
            test.loc[:, to_scale] = scaler.transform(test.loc[:, to_scale])
        if not validation.empty:
            validation.loc[:, to_scale] = scaler.transform(validation.loc[:, to_scale])
        
    if verbose == 2:
        print(learning_df.loc[(learning_df.Time%aggregation)==0, ["Time"] + features + predictions].iloc[50:80].head(10))
        
    cells = training.Cell.unique()
    
    return_obj = {
        'setup':{
            'aggregation': aggregation,
            'update': update,
            'scenario': scenario,
            'direction': direction,
            'learning_params': learning_params,
            'iterations': iterations,
            'features': features,
            'predictions': predictions,
            'max_se': max_se,
            'alpha': alpha,
            'alpha_calc': alpha_calc,
            'target_bps': target_bps,
            'source_fname': source_fname,
            'ratio': ratio,
            'reversed_split': reversed_split,
            'time_min': time_min,
            'cells': list(cells),
            'train_resources': train_resources
        },
        'learning_history': [], # per iteration
        'models': [],
        'scaler': scaler,
        'to_scale': to_scale,
        'cells': cells,
        'results': np.zeros((iterations,len(cells),5)),
        'comp_resources': {
            'before': np.zeros((iterations,3), dtype=np.single),
            'after': np.zeros((iterations,3), dtype=np.single),
            #'by_epoch': np.zeros((iterations,learning_params['epochs']//epochs_between_reports,4), dtype=np.single) # epoch, ru_utime, ru_stime, ru_maxrss 
        },
        'training_results': np.zeros((iterations,len(cells),5)),
        'validation_results': np.zeros((iterations,len(cells),5))
    }
    
    if 'epochs' in learning_params and epochs_between_reports is not None:
        return_obj['comp_resources']['by_epoch'] = np.zeros((iterations,learning_params['epochs']//epochs_between_reports,4), dtype=np.single)
    
    
    #Feature selection and dataset organization
    if do_feature_selection:
        features = feature_selection(training, features)
        return_obj["selected_features"] = features
    split_dataset = organize_dataset([training, test, validation], num_steps, aggregation, features, predictions, simulation_cols, time_flat_dense)
    return_obj["dataset"] = split_dataset
    
    for i in range(iterations):
        lost_insist = retries
        while lost_insist >= 0:
            
            return_obj['comp_resources']['before'][i] = get_comp_resouces()
            
            learning_params['current_iteration'] = i
            if learning_family == "NN":
                model =  learn_neural(return_obj["dataset"], learning_params, return_obj, alpha=alpha_calc, target_bps=target_bps, predictor_scaler=max_se, verbose=verbose)
            elif learning_family == "GB":
                model_gb = learn_gb(return_obj["dataset"], learning_params, return_obj, alpha=alpha_calc, target_bps=target_bps, predictor_scaler=max_se, verbose=verbose)
                if len(split_dataset["training"]["features"].shape) == 3:
                    x_training = split_dataset["training"]["features"].squeeze(axis=1)
                    y_training = split_dataset["training"]["learning-y"].squeeze(axis=1)
                else:
                    x_training = split_dataset["training"]["features"]
                    y_training = split_dataset["training"]["learning-y"]
                if train_resources:
                    model = StdOverProvisioningModel(alpha/max_se, y_training, x_training, model_gb, force_2d=True)                            
            #ENDIF GB
            elif learning_family == "Simple":
                x_training = np.expand_dims(training.Last_SE.values, axis=1)
                y_training = np.expand_dims(training.RB_ratio.values, axis=1)
                
                model = GlobalOverProvisioningModel(alpha/max_se, y_training, x_training, DummySymmetricModel())
            elif learning_family == "Last":
                model = DummySymmetricModel()
            else:
                assert False, f"Learning family \"{learning_family}\" not implemented."
                    
                
            return_obj['detailed_results'] = {
                'training': {},
                'test': {},
                'validation': {}
            }

            n_cell = 0
            should_insist = False
            
            if verbose >= 2:
                print("Resource usage before model calculation:", resource.getrusage(resource.RUSAGE_SELF))
            
            for c in cells:
                print("Max_value", max_se)
                if num_steps is None:
                    results_detailed = run_model(model, alpha, target_bps, test, max_se, cells=[c], skip_steps=1, scaler=scaler, features=features)
                else:
                    results_detailed = run_model_timeseries(model, alpha, target_bps, split_dataset["test"]["features"], split_dataset["test"]["simulation-y"], max_se, scaler=scaler, is_flat_dense=time_flat_dense, trained_resources=train_resources)
                    print("Test", results_detailed[0].mean(), alpha,  target_bps/1000000, scenario, direction, c)
                return_obj['detailed_results']['test'][c] = results_detailed 
                results = list(map((lambda x: x.mean()), results_detailed))
                if results[0]/alpha > 1 and lost_insist > 0:
                    should_insist = True
                    break
                elif results[0]/alpha > 1:
                    print(f"Run out of options for cell {c}")
                return_obj['results'][i][n_cell] = np.array([results[0]/alpha, results[1], results[2], results[3], results[4]])
                
                if register_training:
                    if num_steps is None:
                        training_results_detailed = run_model(model, alpha, target_bps, training, max_se, cells=[c], skip_steps=1, scaler=scaler, features=features)
                    else:
                        training_results_detailed = run_model_timeseries(model, alpha, target_bps, split_dataset["training"]["features"], split_dataset["training"]["simulation-y"], max_se, scaler=scaler, is_flat_dense=time_flat_dense, trained_resources=train_resources)
                        print("Training", training_results_detailed[0].mean())
                    return_obj['detailed_results']['training'][c] = training_results_detailed 
                    training_results = list(map((lambda x: x.mean()),training_results_detailed))
                    return_obj['training_results'][i][n_cell] = np.array([training_results[0]/alpha, training_results[1], training_results[2], training_results[3], training_results[4]])
                
                if split_dataset["validation"]["features"]:
                    if num_steps is None:
                        validation_results_detailed = run_model(model, alpha, target_bps, validation, max_se, cells=[c], skip_steps=1, scaler=scaler, features=features)
                    else:
                        validation_results_detailed = run_model_timeseries(model, alpha, target_bps, split_dataset["validation"]["features"], split_dataset["validation"]["simulation-y"], max_se, scaler=scaler, is_flat_dense=time_flat_dense, trained_resources=train_resources)
                        print("Validation", validation_results_detailed[0].mean())
                    return_obj['detailed_results']['validation'][c] = validation_results_detailed 
                    validation_results = list(map((lambda x: x.mean()), validation_results_detailed))
                    return_obj['validation_results'][i][n_cell] = np.array([validation_results[0]/alpha, validation_results[1], validation_results[2], validation_results[3], validation_results[4]])
                
                n_cell += 1
                    
                
            if should_insist:
                lost_insist -= 1
            else:
                break
              
            
    if verbose >= 2:
        print("Resource usage after model calculation:", resource.getrusage(resource.RUSAGE_SELF))   
    
    stop = time.time()
    if verbose:
        if fname:
            print(f"{fname} done in {stop-start}s!")
        else:
            print(f"Done in {stop-start}s!")
    if fname is not None:
        pickle.dump(return_obj, open(fname, "wb"))
    
    return return_obj
    
        
def run_configurations_per_cell(
    aggregation=60,
    update=60,
    scenario='standard',
    direction='D',
    cell=None,
    cells=range(0,21),
    iterations=10,
    learning_family="NN",
    learning_params={},
    features=['SNR_std', 'SNR_5', 'SNR_25', 'SNR_50', 'SNR_75', 'SNR_95',
            'Bytes_std', 'Bytes_5', 'Bytes_25', 'Bytes_50', 'Bytes_75', 'Bytes_95',
            'Last_equally_distributed', 'Last_SE'
           ],
    predictions=['RB_ratio'],
    max_se=None,
    alpha=100,
    target_bps=10000000,
    verbose=1,
    min_mcs=None,
    max_static=None,
    source_folder="megas",
    fname=None,
    source_df=None,
    num_steps=None,
    scaler=None,
    ratio=(0.8,0.1,0.1),
    reversed_split=False,
    time_min = 0,
    time_max=None,
    retries = 2,
    train_resources=False,
    register_training=False,
    do_transform_input=False,
    time_flat_dense=False,
    pre_calculate_mse=None,
    do_feature_selection=True,
    epochs_between_reports=10
):
    ret_obj = None
    if fname is not None and verbose:
        print(f"Running {fname} per cell...")
        
    #assert cell is None, "Parameter cell only present for script flexibility"
    if cell is not None:
        cells=[cell]
    
    start = time.time()
    for ic, c in enumerate(cells):
        print(fname,"going to cell",c)
        new_ret_obj = run_configuration(
            aggregation=aggregation,
            update=update,
            scenario=scenario,
            direction=direction,
            cell=c,
            iterations=iterations,
            learning_family=learning_family,
            learning_params=learning_params,
            features=features,
            predictions=predictions,
            max_se=max_se,
            alpha=alpha,
            target_bps=target_bps,
            verbose=(verbose - 1 if verbose > 0 else 0),
            min_mcs=min_mcs,
            max_static=max_static,
            source_folder=source_folder,
            fname=None,
            source_df=source_df,
            num_steps=num_steps,
            scaler=scaler,
            ratio=ratio,
            reversed_split=reversed_split,
            time_min = time_min,
            time_max = time_max,
            retries=retries,
            train_resources=train_resources,
            register_training=register_training,
            do_transform_input=do_transform_input,
            time_flat_dense=time_flat_dense,
            pre_calculate_mse=None,
            do_feature_selection=do_feature_selection,
            epochs_between_reports=epochs_between_reports
        )
        if ret_obj is None:
            ret_obj = copy.deepcopy(new_ret_obj)
            for k in ['results', 'training_results', 'validation_results']:
                ret_obj[k] = np.zeros((iterations,len(cells),5))
            ret_obj['detailed_results'] = {
                'training': {},
                'test': {},
                'validation': {}
            }
            ret_obj['cells'] = list(cells)
            ret_obj['learning_history'] = {}
            ret_obj['models'] = {}
            ret_obj['scalers'] = {}
            ret_obj['cells']=list(cells)
            ret_obj['setup']['cells']=list(cells)
            if 'dataset' in new_ret_obj:
                ret_obj['dataset'] = {}
            
        for k in ['results', 'training_results', 'validation_results']:
            ret_obj[k][:,ic,:] = new_ret_obj[k].squeeze(axis=1)
            
        for set_name in new_ret_obj['detailed_results']:
            if c in new_ret_obj['detailed_results'][set_name]:
                ret_obj['detailed_results'][set_name][c] = new_ret_obj['detailed_results'][set_name][c]
            
        ret_obj['learning_history'][ic] = new_ret_obj['learning_history']
        ret_obj['models'][c] = new_ret_obj['models']
        ret_obj['scalers'][c] = new_ret_obj['scaler']
        if 'dataset' in new_ret_obj:
            ret_obj['dataset'][c] = new_ret_obj['dataset']
        
    stop = time.time()
        
    if verbose:
        if fname:
            print(f"{fname} done in {stop-start}s!")
        else:
            print(f"Done in {stop-start}s!")
    if fname is not None:
        pickle.dump(ret_obj, open(fname, "wb"))
    
    return ret_obj
      

if __name__ == "__main__":
    #name = "separate_cells-correct-hl_0-epochs_700"
    #ret_obj = run_configurations_per_cell(epochs=700)
    #pickle.dump(ret_obj, open(f"results/parameterization/{name}.pickle","wb"))
    name = "test"
    ret_obj = run_configuration(epochs=10, iterations=1)
    pickle.dump(ret_obj, open(f"results/parameterization/{name}.pickle","wb"))
    
    



