# cnn model visualizer
# https://dgschwend.github.io/netscope/quickstart.html
# ===================================================
# NEXT STEP: BOTTLENECK ANALYSIS FOR VARIOUS LAYERS
# We can actually do one more thing more aggresively:
# design space exploration w/o CNN structure.
# -> distribute HAC and FFT resource based on fin_p,
# -> scale down the whole design by bw.
# ===================================================

import sys
from os.path import expanduser
home = expanduser("~")

ZYTHON_PATH = "{}/Projects/".format(home)
sys.path.insert(0, ZYTHON_PATH)


import os
import argparse
from zython.logf.printf import printf
import numpy as np
import yaml
import common
import py_dse.algo_engine as algo_engine
import py_dse.arch_engine as arch_engine
import math
import copy
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*use @default decorator instead.*')


MAXD = 15


def parse_args():
    parser = argparse.ArgumentParser('FFT CNN design auto generation')
    parser.add_argument('-n','--cnn', type=str,default=None, required=True,
                    help='path to the cnn config *.yaml file')
    parser.add_argument('-p','--hardware', type=str,default=None, required=True,
                    help='path to the hardware config *.yaml file')
    parser.add_argument('-c','--dse_conf', type=str,default=None, required=True,
                    help='path to the design space exploration config *.yaml file')
    parser.add_argument('-i','--import_model', type=str,default=None,
                    help='path to the identified hardware configuration. For direct performance prediction')
    parser.add_argument('-a','--algo_dse', action='store_true',
                    help='select the mode: shall we do the algo dse only?')
    return parser.parse_args()


def parse_cnn(cnn_yaml,deselect_layers=[]):
    """
    INPUT:
        cnn_yaml                STRING of path to the CNN model file *.yaml
        deselect_layers         LIST of indices for layers to be ignored.
                                Layer index starts from 0.
    OUTPUT:
        params_cnn             {'name': ..,
                                'f_in': ..,
                                'f_out': ..,
                                'l_img': ..,
                                'l_kern': ..,
                                'stride': ..,
                                'pad': ..}
    """
    with open(cnn_yaml) as f_cnn_yaml:
        model_cnn = yaml.load(f_cnn_yaml)
    tot_layer = len(model_cnn['conv_layers'])
    filtered_layers = list(set(np.arange(tot_layer)) - set(deselect_layers))
    num_layer = len(filtered_layers)
    params_cnn = {  'name':     model_cnn['name'],
                    'f_in':     np.zeros(num_layer, dtype=np.int64),
                    'f_out':    np.zeros(num_layer, dtype=np.int64),
                    'l_img':    np.zeros(num_layer, dtype=np.int64),
                    'l_kern':   np.zeros(num_layer, dtype=np.int64),
                    'stride':   np.zeros(num_layer, dtype=np.int64),
                    'pad':      np.zeros(num_layer, dtype=np.int64)}
    for i,li in enumerate(filtered_layers):
        layer = model_cnn['conv_layers'][li]
        params_cnn['f_in'][i] = layer['f_in']
        params_cnn['f_out'][i] = layer['f_out']
        params_cnn['l_img'][i] = layer['l_img']
        params_cnn['l_kern'][i] = layer['l_kern']
        params_cnn['stride'][i] = layer['stride']
        params_cnn['pad'][i] = layer['pad']
    return params_cnn



def parse_input(cnn_yaml,hw_yaml,dse_yaml):
    """
    layer_filter:     can filter out a list of layers
    """
    with open(cnn_yaml) as f_cnn_yaml:
        model_cnn = yaml.load(f_cnn_yaml)
    with open(hw_yaml) as f_hw_yaml:
        model_hw = yaml.load(f_hw_yaml)
    with open(dse_yaml) as f_dse_yaml:
        model_tool = yaml.load(f_dse_yaml)

    params_algo = {'name':    model_tool['algo name'],
                   'N avail': np.array(model_tool['N avail']),
                   'd max':   model_tool['d max'],
                   'layer filter': None}
    params_dse = {'range P_hac': model_tool['range P_hac'],
                  'range T_img': model_tool['range T_img'],
                  'range f_in':  model_tool['range f_in'],
                  'range f_out': model_tool['range f_out']}
    tot_layers = len(model_cnn['conv_layers'])
    if model_tool['layer filter']['type'] == 'all':
        model_tool['layer filter']['index'] = np.arange(tot_layers)
    elif model_tool['layer filter']['type'] == 'deselect':
        model_tool['layer filter']['index'] = set(np.arange(tot_layers))-set(model_tool['layer filter']['index'])
        model_tool['layer filter']['index'] = np.array(list(model_tool['layer filter']['index']))
    num_layer = len(model_tool['layer filter']['index'])
    params_cnn = {  'name':     model_cnn['name'],
                    'f_in':     np.zeros(num_layer, dtype=np.int64),
                    'f_out':    np.zeros(num_layer, dtype=np.int64),
                    'l_img':    np.zeros(num_layer, dtype=np.int64),
                    'l_kern':   np.zeros(num_layer, dtype=np.int64),
                    'stride':   np.zeros(num_layer, dtype=np.int64),
                    'pad':      np.zeros(num_layer, dtype=np.int64)}
    for i,li in enumerate(model_tool['layer filter']['index']):
        layer = model_cnn['conv_layers'][li]
        params_cnn['f_in'][i] = layer['f_in']
        params_cnn['f_out'][i] = layer['f_out']
        params_cnn['l_img'][i] = layer['l_img']
        params_cnn['l_kern'][i] = layer['l_kern']
        params_cnn['stride'][i] = layer['stride']
        params_cnn['pad'][i] = layer['pad']
    params_meta = {'byte per cpx word': math.ceil(model_hw['data_precision']/8)*2,
                   'clk rate': model_hw['clk_rate'],
                   'bandwidth': model_hw['bandwidth'], # byte per sec
                   'alm': model_hw['alm'],
                   'dsp': model_hw['dsp'],
                   'memory': model_hw['memory']}
    params_arch = {'cpx mac alm': 64*3.2,
                   'cpx mac dsp': 2,            # Stratix V devices uses 2 DSP blocks to compute 18-bit complex MAC
                   'radix': 4}
    params_algo.pop('layer filter',None)
    return params_cnn,params_algo,params_meta,params_arch,params_dse




##################################################
#                                                #
##################################################
def compare_algo(cnn_dir,hw_config,tool_config):
    """
    cnn_dir: the directory for couple of cnn models
    hw_config: the hw config yaml file
    tool_config: the tool config yaml file
    """
    cnn_directory = os.fsencode(cnn_dir)
    oaa_count = None
    cap_count = None
    spa_count = None
    num_cnn = 0
    file_list = []
    for cnn_f in os.listdir(cnn_directory):
        cnn_filename = os.fsencode(cnn_f).decode('utf-8')
        if cnn_filename.endswith('.yaml'):
            num_cnn += 1
            file_list += [cnn_filename]
        else:
            continue
    file_list.sort()
    for cnn_f in file_list:
        cnn_filename = os.fsencode(cnn_f).decode('utf-8')
        params_cnn,params_algo,_1,_2,_3 = parse_input('{}/{}'.\
            format(cnn_directory.decode('utf-8'),cnn_filename),hw_config,tool_config)
        params_algo_spa = copy.deepcopy(params_algo)
        params_algo_spa['d max'] = 1
        params_algo_spa['name'] = 'Spatial'
        ae_spa = algo_engine.spatial_complexity(params_cnn,params_algo_spa)
        ae_cap = algo_engine.fft_complexity(params_cnn,params_algo)
        params_algo_oaa = copy.deepcopy(params_algo)
        params_algo_oaa['d max'] = 1
        params_algo_oaa['name'] = params_algo_oaa['name'].replace('CaP','OaA')
        ae_oaa = algo_engine.fft_complexity(params_cnn,params_algo_oaa)
        ae_spa.count()
        ae_cap.count(global_d_N=True)
        ae_oaa.count()
        printf(ae_cap.str_compare_algo(ae_oaa,ae_spa),type=None,separator='=')
        printf('{} d value: {}', cnn_filename, ae_cap.chosen_params_algo['batch fold'])
        if oaa_count is None:
            oaa_count = ae_oaa.ops_count
            cap_count = ae_cap.ops_count
            spa_count = ae_spa.ops_count
        else:
            oaa_count = np.concatenate([oaa_count,ae_oaa.ops_count])
            cap_count = np.concatenate([cap_count,ae_cap.ops_count])
            spa_count = np.concatenate([spa_count,ae_spa.ops_count])
    oaa_count = oaa_count.reshape(num_cnn,-1).T
    cap_count = cap_count.reshape(num_cnn,-1).T
    spa_count = spa_count.reshape(num_cnn,-1).T
    to_file = '{}/{}_complexity.npy'
    np.save(to_file.format(cnn_dir,'OaA'),oaa_count)
    np.save(to_file.format(cnn_dir,'CaP'),cap_count)
    np.save(to_file.format(cnn_dir,'SPA'),spa_count)


##################################################
#                                                #
##################################################

if __name__ == '__main__':
    args = parse_args()
    params_cnn,params_algo,params_meta,params_arch,params_dse\
         = parse_input(args.cnn,args.hardware,args.dse_conf)

    arch_engine_inst = arch_engine.design_space_explore(params_cnn,
                params_algo,params_meta,params_arch,params_dse)
    if args.import_model is not None:
        arch_engine_inst.import_param(args.import_model)
        arch_engine_inst.total_clk_per_layer()#output_rate()
    else:
        arch_engine_inst.design_space_exploration()

    print(arch_engine_inst.str_performance())
    arch_engine_inst.export_param()
