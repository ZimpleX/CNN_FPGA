import math
import numpy as np
from zython.logf.printf import stringf
from zython.logf.printf import printf
from zython.logf.filef import print_to_file
import py_dse.algo_engine as algo_engine
import yaml


"""
ASSUMPTION:
    CaP-OaA on image matrices.
    N-point 2D FFT on the kernel matrices.
    Split f_in and f_out to fit into kernel buffer.
    
ALGORITHM:
    Consider individual layer separately, fix for each layer i:
     *  fin_p
     *  fout_p
     *  N
     *  d
    Calculate T_img for layer i based on remaining buffer size
    Consider all layers overall. Calculate:
     *  q_1dfft, q_2dfft, q_1difft, q_2difft
     *  P_fft, P_ifft
     *  P_hac, q_hac
"""

class design_space_explore:
    """
    Assume that MAC are all implemented using DSPs and FFT/IFFT are all implemented by ALMs.
    """
    def __init__(self,cnn_layers,algo_conf,meta_conf,arch_conf,dse_conf):
        """
        <del>ASSUME THAT WE USE ONLY ONE FFT SIZE</del>
        ASSUME THAT WE USE DIFFERENT FFT SIZE ACROSS LAYERS.

        INPUT:
            cnn_layers         {'name': ..,
                                'f_in': ..,
                                'f_out': ..,
                                'l_img': ..,
                                'l_kern': ..,
                                'stride': ..,
                                'pad': ..}
            algo_conf          {'name': ..,
                                'N avail': ..,
                                'd max': ..,
                                'layer filter': ..}
            meta_conf          {'bytes per cpx word': ..,       # 4
                                'clk rate': ..,                 # 200e6
                                'bandwidth': ..,                # 
                                'alm': ..,
                                'dsp': ..,
                                'memory': ..}                   # bytes
            arch_conf          {'cpx mac alm': ..,
                                'cpx mac dsp': ..,              # 2, for 18-bit MAC on Stratix V
                                'radix': ..}
            dse_conf           {'range P_hac': ..,
                                'range T_img': ..,
                                'range f_in': ..,
                                'range f_out': ..}
        
        NOTE: FOLDING FACTORS ARE ALL RELATIVE TO THE MAX FFT SIZE. SCALE 
        ACCORDINGLY WHEN CALCULATING THE PERFORMANCE. 
        """    
        self.cnn = cnn_layers
        self.num_layers = len(self.cnn['f_in'])
        # --------
        self.algo_conf = algo_conf
        self.algo_engine = algo_engine.fft_complexity(cnn_layers,algo_conf)
        # --------
        # Layer parameter: derive these based on consideration of single layer only
        self.layer = {'Ni':     [],     # fft size for each layer i
                     'di':      [],     # folding factor for each layer i
                     'fin_pi':  [],     # partitioned fin for layer i
                     'fout_pi': [],     # partitioned fout for layer i
                     'T_img_i': []}     # 
        # --------
        # Network parameter: derive these based on consideration of complete CNN.
        self.network = {'q_1dfft':  None,
                      'q_2dfft':    None,
                      'q_1difft':   None,
                      'q_2difft':   None,
                      'P_fft':      None,
                      'P_ifft':     None,
                      'P_hac':      None,
                      'q_hac':      None}
        # ---------
        self.meta = meta_conf
        self.arch = arch_conf
        self.dse_range = dse_conf
        # ---------
        self.Nmax = None        # set this after fixing self.layer parameter
        self.ops_per_layer = algo_engine.spatial_complexity(self.cnn,'spatial conv').count()
        self.log_dir = 'output_hw_param'


    def consumption_fft(self):
        """
        VERIFIED AUG 12, 2017
        """
        num_pipeline = self.Nmax/self.network['q_2dfft']
        par_pipeline = self.Nmax/self.network['q_1dfft']
        alm = 2*(math.log(self.Nmax,self.arch['radix'])-1)*num_pipeline\
                *par_pipeline*self.network['P_fft']*self.arch['cpx mac alm']
        mem_spn_pipel = self.network['P_fft']*(2*self.Nmax/self.network['q_2dfft']*3*self.Nmax)
        mem_spn_trans = self.network['P_fft']*self.Nmax**2
        mem = (mem_spn_pipel + mem_spn_trans)*self.meta['byte per cpx word']
        return {'dsp': 0, 'alm': alm, 'mem': mem}

    def consumption_ifft(self):
        """
        VERIFIED AUG 12, 2017
        """
        num_pipeline = self.Nmax/self.network['q_2difft']
        par_pipeline = self.Nmax/self.network['q_1difft']
        alm = 2*(math.log(self.Nmax,self.arch['radix'])-1)*num_pipeline\
                *par_pipeline*self.network['P_ifft']*self.arch['cpx mac alm']
        mem_spn_pipel = self.network['P_ifft']*(2*self.Nmax/self.network['q_2difft']*3*self.Nmax)
        mem_spn_trans = self.network['P_ifft']*self.Nmax**2
        mem = (mem_spn_pipel +  mem_spn_trans)*self.meta['byte per cpx word']
        return {'dsp': 0, 'alm': alm, 'mem': mem}

    def consumption_hac(self):
        """
        VERIFIED AUG 12, 2017
        """
        dsp = self.network['P_hac']*self.Nmax**2/self.network['q_hac']*self.arch['cpx mac dsp']
        mem = 0
        return {'dsp': dsp, 'alm': 0, 'mem': mem}

    def consumption_img_buf(self):
        """
        MODIFIED AUG 13, 2017
        """
        # NOTE: THIS FIN_P IS IN TERMS OF THE MAX FFT SIZE AS WELL, YOU NEED
        # TO SCALE UP FIN_P AND FOUT_P FOR LAYERS WITH SMALLER FFT SIZES
        mem = (self.layer['T_img_i']*self.layer['fin_pi']*self.layer['Ni']**2)*self.meta['byte per cpx word']
        mem = mem.max()
        mem *= 2    # DOUBLE BUFFERING
        return {'dsp': 0, 'alm': 0, 'mem': mem}

    def consumption_kern_buf(self):
        """
        MODIFIED AUG 13, 2017
        """
        mem = (self.layer['fin_pi']*self.layer['fout_pi']*self.layer['Ni']**2)*self.meta['byte per cpx word']
        mem = mem.max()
        return {'dsp': 0, 'alm': 0, 'mem': mem}
        
    ########################
    #  IMPORTANT FUNCTION  #
    ########################
    def output_rate(self,layer_idx):
        # bw: byte/sec, clk rate: Hz. 
        thp_bw  = self.meta['bandwidth']/self.meta['clk rate']/(self.meta['byte per cpx word']/2) # cuz transfer only real num
        thp_hac = self.network['P_hac']*self.Nmax**2/self.network['q_hac']/self.layer['fin_pi']*2 # factor of 2 cuz of using imag channel as well
        # SOMETHING TO MAKE SURE THE RANGE OF Q_1DFFT, Q_2DFFT
        thp_fft = self.network['P_fft']*self.Nmax**2/self.network['q_1dfft']/self.network['q_2dfft']*2
        thp_ifft = self.network['P_ifft']*self.Nmax**2/self.network['q_1difft']/self.network['q_2difft']*2
        # this coefficient is fine, cuz fout and fin scales simultaneously.
        coef_bw = self.layer['fout_pi']/(self.layer['fin_pi']+self.layer['fout_pi'])
        coef_hac = 1.
        coef_fft = self.layer['fout_pi']/self.layer['fin_pi']
        coef_ifft = np.array([1.]*len(layer_idx))
        rate_module = np.concatenate([np.array(coef_bw*thp_bw),
                                      np.array(coef_hac*thp_hac),
                                      np.array(coef_fft*thp_fft),
                                      np.array(coef_ifft*thp_ifft)]).reshape(4,-1)
        min_rate_sys = np.min(rate_module,axis=0)
        #import pdb; pdb.set_trace()
        return min_rate_sys,np.argmin(rate_module,axis=0)

    def layer_i_param(self, li):
        """
        Design space exploration on the ith layer. Set the optimal hardware params for layer i.
        As an approximation, we assume that all memory used for kernel buffer.
        Anyway, T_img_i does not have any effect on performance. 

        INPUT:
            li              INTEGER, index of the layer under concern
        OUTPUT:
            NONE
        """
        opt_layer_param = {'Ni': None,
                           'di': None,
                           'fin_pi': None,
                           'fout_pi': None,
                           'T_img_i': None}
        opt_time = float('inf')
        for Ni in self.algo_conf['N avail']:
            if Ni-self.cnn['l_kern'][li]+1 <= 0:
                continue
            self.Nmax = Ni
            self.layer['Ni'] = np.array([Ni])
            self.algo_engine.algo_params['N avail'] = np.array([Ni])
            self.algo_engine.count(layer_list=[li])
            self.layer['di'] = np.array([self.algo_engine.algo_params['batch fold'][li]])
            for fin_pi in range(1,self.cnn['f_in'][li]+1,self.dse_range['range f_in']['stride']):
                fout_pi = int(np.floor(self.meta['memory']/self.meta['byte per cpx word']/Ni**2/fin_pi))
                if fout_pi <= 0:
                    continue
                _num_fout_rounds = np.ceil(self.cnn['f_out'][li]/fout_pi)
                fout_pi = int(np.ceil(self.cnn['f_out'][li]/_num_fout_rounds))
                self.layer['fin_pi'] = np.array([fin_pi])
                self.layer['fout_pi'] = np.array([fout_pi])
                max_pw_fft = int(math.log(Ni,2))
                for pw_1dfft in range(1,max_pw_fft+1):
                    for pw_1difft in range(1,max_pw_fft+1):
                        for pw_2dfft in range(1,max_pw_fft+1):
                            for pw_2difft in range(1,max_pw_fft+1):
                                self.network['q_1dfft'] = 2**pw_1dfft
                                self.network['q_1difft']= 2**pw_1difft
                                self.network['q_2dfft'] = 2**pw_2dfft
                                self.network['q_2difft']= 2**pw_2difft
                                for pw_q_hac in range(1,max_pw_fft+1):
                                    self.network['q_hac'] = 2**pw_q_hac
                                    # this is for approximation as well
                                    P_hac = np.floor(self.meta['dsp']/self.arch['cpx mac dsp']*self.network['q_hac']/Ni**2)
                                    if P_hac == 0.:
                                        continue
                                    self.network['P_hac'] = P_hac
                                    cur_time = self.total_clk_per_layer(np.array([li])).sum()
                                    if opt_time < cur_time:
                                        continue
                                    opt_time = cur_time
                                    self.copy_layer_param(opt_layer_param)
        self.Nmax = None
        return opt_layer_param
        
    def layer_design_space_exploration(self):
        """
        Design space exploration 
        """
        self.network['P_fft'] = 1
        self.network['P_ifft'] = 1
        self.layer['T_img_i'] = np.array([1])       # Value of T_img_i does not matter.
        opt_layers = {'Ni': [],
                      'di': [],
                      'fin_pi': [],
                      'fout_pi': [],
                      'T_img_i': []}
        for li in range(self.num_layers):
            _opt_layer_li_param = self.layer_i_param(li)
            for k in self.layer.keys():
                opt_layers[k].append(_opt_layer_li_param[k][0])
            printf('finish layer dse for {}', li)
            self.print_layer_conf(_opt_layer_li_param,li)
        for k in self.layer.keys():
            opt_layers[k] = np.array(opt_layers[k])
        self.layer = opt_layers
        self.layer['T_img_i'] = np.array([1]*self.num_layers)
        self.Nmax = self.layer['Ni'].max()
            
            
    def total_clk_per_layer(self,layer_idx=None,mode='realistic'):
        """
        time in terms of number of clk
        layer_idx: should be an ndarray specifying the layer idx
        
        if mode == 'ideal':
            get the theoritical performance ignoring communication constraint.
            this is the computation bound. 
        """
        if mode == 'ideal':
            ideal_output_rate = self.meta['dsp']/self.arch['cpx mac dsp']/self.layer['fin_pi'] #### TODO: need to x2 cuz of the imaginary channel
            ideal_time_per_round = self.layer['fout_pi']*self.layer['T_img_i']*self.layer['Ni']**2/ideal_output_rate
            ideal_num_rounds = self.cnn['f_in']*self.cnn['f_out']/self.layer['fin_pi']/self.layer['fout_pi']\
                *(self.cnn['l_img']+self.cnn['l_kern']-1)**2/self.layer['T_img_i']/self.layer['Ni']**2
            #printf('IDEAL\noutput rate: {}\ntime per round: {}\nnum nodes: {}\n',\
            #       ideal_output_rate[0:-3],ideal_time_per_round[0:-3],ideal_num_rounds[0:-3],type='DEBUG')
            return ideal_time_per_round*ideal_num_rounds
        if layer_idx is None:
            layer_idx = np.arange(self.num_layers)
        l_img_d = self.layer['di']*self.cnn['l_img'][layer_idx]\
                + (self.layer['di']-1)*(self.cnn['l_kern'][layer_idx]-1)
        time_per_round = self.layer['T_img_i']*self.layer['fout_pi']*self.layer['Ni']**2/self.output_rate(layer_idx)[0]
        _num_fin_part = np.ceil(self.cnn['f_in'][layer_idx]/self.layer['fin_pi'])
        _num_fout_part = np.ceil(self.cnn['f_out'][layer_idx]/self.layer['fout_pi'])
        _num_tiles_2d = np.ceil(l_img_d/(self.layer['Ni']-self.cnn['l_kern'][layer_idx]+1))**2
        num_rounds = _num_fin_part*_num_fout_part\
            *_num_tiles_2d/self.layer['T_img_i']\
            /self.layer['di']**2
        #printf('REALISTIC\noutput rate: {}\ntime per round: {}\nnum nodes: {}\n',\
        #       self.output_rate(layer_idx)[0][0:-3],time_per_round[0:-3],num_rounds[0:-3],type='DEBUG')
        return time_per_round*num_rounds


    def copy_layer_param(self,temp_layer):
        """
        Util function
        """
        temp_layer['Ni'] = self.layer['Ni']
        temp_layer['di'] = self.layer['di']
        temp_layer['fin_pi'] = self.layer['fin_pi']
        temp_layer['fout_pi'] = self.layer['fout_pi']
        temp_layer['T_img_i'] = self.layer['T_img_i']

    def copy_network_param(self,temp_network):
        """
        Util function
        """
        temp_network['q_1dfft'] = self.network['q_1dfft']
        temp_network['q_2dfft'] = self.network['q_2dfft']
        temp_network['q_1difft'] = self.network['q_1difft']
        temp_network['q_2difft'] = self.network['q_2difft']
        temp_network['P_fft'] = self.network['P_fft']
        temp_network['P_ifft'] = self.network['P_ifft']
        temp_network['P_hac'] = self.network['P_hac']
        temp_network['q_hac'] = self.network['q_hac']

    def design_space_exploration(self):
        self.layer_design_space_exploration()
        opt_network_param = {'q_1dfft': None,
                     'q_2dfft': None,
                     'q_1difft':None,
                     'q_2difft':None,
                     'P_fft':   None,
                     'P_ifft':  None,
                     'P_hac':   None,
                     'q_hac':   None}
        opt_fout = None
        self.network['P_fft'] = 1
        self.network['P_ifft'] = 1
        opt_time = float('Inf')
        # ==================================================
        # TODO: this range has to be thought about carefully
        # ==================================================
        max_pw_fft = int(math.log(self.Nmax,2))
        min_pw_fft = max_pw_fft - int(math.log(self.layer['Ni'].min(),2))
        for pw_1dfft in range(min_pw_fft,max_pw_fft+1):
            for pw_1difft in range(min_pw_fft,max_pw_fft+1):
                for pw_2dfft in range(min_pw_fft,max_pw_fft+1):
                    for pw_2difft in range(min_pw_fft,max_pw_fft+1):
                        self.network['q_1dfft'] = 2**pw_1dfft
                        self.network['q_1difft'] = 2**pw_1difft
                        self.network['q_2dfft'] = 2**pw_2dfft
                        self.network['q_2difft'] = 2**pw_2difft
                        _start_P_hac = self.dse_range['range P_hac']['start']
                        _end_P_hac = self.dse_range['range P_hac']['end']
                        _stride_P_hac = self.dse_range['range P_hac']['stride']
                        for i_P_hac in range(_start_P_hac,_end_P_hac,_stride_P_hac):
                            self.network['P_hac'] = i_P_hac
                            for pw_q_hac in range(1,max_pw_fft+1):
                                self.network['q_hac'] = 2**pw_q_hac
                                res_fft = self.consumption_fft()
                                res_hac = self.consumption_hac()
                                res_ifft = self.consumption_ifft()
                                res_img_buf = self.consumption_img_buf()
                                # set self.layer['fout_pi'] based on the remaining buffer budget
                                remain_mem = self.meta['memory']\
                                    -res_fft['mem']-res_hac['mem']-res_ifft['mem']-res_img_buf['mem']
                                fout_pi = np.floor(remain_mem/self.meta['byte per cpx word']\
                                    /self.layer['Ni']**2/self.layer['fin_pi']).astype(np.int)
                                #if pw_q_hac == max_pw_fft:
                                #    import pdb; pdb.set_trace()
                                if np.any(fout_pi <= 0):
                                    continue
                                _num_fout_rounds = np.ceil(self.cnn['f_out']/fout_pi)
                                fout_pi = np.ceil(self.cnn['f_out']/_num_fout_rounds)
                                self.layer['fout_pi'] = fout_pi
                                res_kern_buf = self.consumption_kern_buf()
                                if self.meta['alm'] <\
                                    res_fft['alm']+res_hac['alm']+\
                                    res_ifft['alm']+res_img_buf['alm']+\
                                    res_kern_buf['alm']:
                                    continue
                                cur_time = self.total_clk_per_layer().sum()
                                if opt_time < cur_time:
                                    continue
                                opt_time = cur_time
                                self.copy_network_param(opt_network_param)
                                opt_fout = [i for i in self.layer['fout_pi']]
        self.network = opt_network_param
        self.layer['fout_pi'] = np.array(opt_fout)


    def clk_to_time(self):
        return self.total_clk_per_layer()/self.meta['clk rate']

    def clk_to_GOPS_i(self):
        """
        Throughput per layer
        """
        time_per_layer = self.clk_to_time()
        return self.ops_per_layer/time_per_layer/1e9        


    def import_param(self, fname=None):
        """
        NEED WORK FOR LOADED_PARAM
        """
        if fname is None:
            fname = './{}/{}_{}.yaml'.format(self.log_dir,self.cnn['name'],self.algo_conf['name'])
        with open(fname) as f:
            loaded_param = yaml.load(f)
            self.layer = loaded_param['layer']
            ### need to check type of loaded_param['layer']
            self.network = loaded_param['network']
        for l in self.layer:
            self.layer[l] = np.array(self.layer[l])
        self.Nmax = max(self.layer['Ni'])
        # set the optimal d value -- use a single d for all layers
        min_clk = float('Inf')
        if self.layer['di'][0] < 0:
            max_di = -self.layer['di'][0]
            for di in range(1,max_di):
                self.layer['di'] = np.array([di]*len(self.layer['di']))
                cur_clk = self.total_clk_per_layer().sum()
                if min_clk > cur_clk:
                    best_d = di
                    min_clk = cur_clk
            self.layer['di'] = np.array([best_d]*len(self.layer['di']))
            #printf('best d is: {}',best_d)

    def export_param(self):
        """
        NEED WORK HERE
        """
        fname = '{}_{}.yaml'.format(self.cnn['name'],self.algo_conf['name'])
        msg = ''
        msg += 'network:\n'
        for k,v in self.network.items():
            msg += '    {}: {}\n'.format(k,v)
        msg += 'layer:\n'
        for k,v in self.layer.items():
            msg += '    {}:\n'.format(k)
            for vi in v:
                msg += '     - {}\n'.format(vi)
        print_to_file(fname,msg,type=None,log_dir=self.log_dir,mode='w')


    def print_layer_conf(self, _opt_layer_li_param,li):
        #import pdb;pdb.set_trace()
        num_param = len(_opt_layer_li_param)
        row_regex_s = '  '.join(['{:>10s}']*num_param)
        row_regex_d = '  '.join(['{:>10d}']*num_param)
        s = stringf('LAYER {} CONF', li, type=None, separator='.')
        s += '\n'
        s += row_regex_s.format('Ni','di','fin_pi','fout_pi','T_img_i')
        s += '\n'
        s += '-'*(10*num_param+2*(num_param-1)) + '\n'
        s += row_regex_d.format(_opt_layer_li_param['Ni'][0],_opt_layer_li_param['di'][0],
                              _opt_layer_li_param['fin_pi'][0],_opt_layer_li_param['fout_pi'][0],
                              _opt_layer_li_param['T_img_i'][0])
        s += '\n'
        printf(s,type=None,separator='*')


    def str_performance(self):
        s  = stringf('CONFIG-LAYER', type=None,separator='=')
        s += '\n'
        for k,v in self.layer.items():
            s += stringf('{:>10s}: {:<s}',k,str(v),type=None)
            s += '\n'
        s += stringf('CONFIG-NETWORK',type=None,separator='=')
        s += '\n'
        for k,v in self.network.items():
            s += stringf('{:>10s}: {:<10d}',k,int(v),type=None)
            s += '\n'
        s += stringf('THROUGHPUT',type=None,separator='=')
        s += '\n'
        time_per_layer = self.clk_to_time()
        GOPS_per_layer = self.clk_to_GOPS_i()
        s += stringf('layer  time (ms)   ops (G)      GOPS',type=None,separator='-')
        s += '\n'
        for li in range(self.num_layers):
            #import pdb; pdb.set_trace()
            s += stringf('{:5d}     {:5.1f}     {:5.3f}     {:6.1f}\n',
                li,time_per_layer[li]*1e3,self.ops_per_layer[li]/1e9,
                GOPS_per_layer[li],type=None)
        s += '-'*36 + '\n'
        s += stringf('TOTAL     {:5.1f}     {:5.3f}     {:6.1f}\n\n',
            time_per_layer.sum()*1e3,self.ops_per_layer.sum()/1e9,
            self.ops_per_layer.sum()/time_per_layer.sum()/1e9,type=None)
        s += stringf('CONSUMPTION',type=None,separator='=')
        s += '\n'
        res_fft = self.consumption_fft()
        res_hac = self.consumption_hac()
        res_ifft = self.consumption_ifft()
        res_img_buf = self.consumption_img_buf()
        res_kern_buf = self.consumption_kern_buf()
        module_dict = { 'fft':res_fft,
                        'hac':res_hac,
                        'ifft':res_ifft,
                        'img_buf':res_img_buf,
                        'kern_buf':res_kern_buf}
        s += stringf('  module       alm    (%)   mem (MB)   (%)',type=None,separator='-')
        s += '\n'
        for n,r in module_dict.items():
            s += stringf('{:>8s}    {:6d} ({:4.2f})   {:6.5f} ({:4.2f})\n',
                n,int(r['alm']),r['alm']/self.meta['alm'],r['mem']/1e6,r['mem']/self.meta['memory'],
                type=None)
        tot_alm = res_fft['alm']+res_hac['alm']+res_ifft['alm']+res_img_buf['alm']+res_kern_buf['alm']
        tot_mem = res_fft['mem']+res_hac['mem']+res_ifft['mem']+res_img_buf['mem']+res_kern_buf['mem']
        s += '-'*42 + '\n'
        s += stringf('{:>8s}    {:6d} ({:4.2f})   {:6.5f} ({:4.2f})\n\n',
            'TOTAL',int(tot_alm),tot_alm/self.meta['alm'],
            tot_mem/1e6,tot_mem/self.meta['memory'],type=None)
        return s
